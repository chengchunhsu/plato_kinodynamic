import os
import torch
import h5py
os.environ["MUJOCO_GL"] = "egl"
import mujoco
import numpy as np
import mediapy as media
from robosuite.utils.transform_utils import convert_quat, quat2axisangle, axisangle2quat
from plato_copilot.kinodynamics.dynamics_model import DynamicsDeltaOutputModel, safe_device

def replay_state(mj_model, mj_data, qpos, qvel, mocap_pos, mocap_quat):
    mujoco.mj_setState(mj_model, mj_data, qpos, mujoco.mjtState.mjSTATE_QPOS)
    mujoco.mj_setState(mj_model, mj_data, qvel, mujoco.mjtState.mjSTATE_QVEL)
    mujoco.mj_setState(mj_model, mj_data, np.squeeze(mocap_pos), mujoco.mjtState.mjSTATE_MOCAP_POS)
    mujoco.mj_setState(mj_model, mj_data, np.squeeze(mocap_quat), mujoco.mjtState.mjSTATE_MOCAP_QUAT)
    mujoco.mj_step1(mj_model, mj_data)

experiment_path = "experiments/run_001"

dynamics_model = safe_device(DynamicsDeltaOutputModel())
dynamics_model.load_state_dict(torch.load(os.path.join(experiment_path, "dynamics_model_delta.pth")))
dynamics_model.eval()
device = dynamics_model.device

idx = 10
dataset_path = "dataset"
with h5py.File(os.path.join(dataset_path, "training_data.hdf5"), "r") as f:
    key = list(f["data"].keys())[idx]
    # 1) block pose: (T,6)
    obj_pose = f["data"][key]['object_pose'][()]  # (T,6)
    # 2) tool pose: (T,6)
    tool_pose = f["data"][key]['mocap_pose'][()]  # (T,6)
    # 3) force/torque feedback: (T,6)
    force_fb = f["data"][key]['force_feedback'][()]  # (T,6)
    # 4) actions: (T,6)
    actions = f["data"][key]['actions'][()]  # (T,6)
    traj_idx = f["data"][key].attrs["traj_idx"]

seq_len = 10
interval      = 50
rollout_len   = 100

scene_option = mujoco.MjvOption()
scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = False
with h5py.File(f'{dataset_path}/data.hdf5', 'r') as f:
    xml = f['data'][traj_idx].attrs['xml']
    block_name = f['data'][traj_idx].attrs['block_name']

    mj_model = mujoco.MjModel.from_xml_string(xml)
    mj_data = mujoco.MjData(mj_model)
    renderer = mujoco.Renderer(mj_model, 480, 640)

    mujoco.mj_resetData(mj_model, mj_data)
    mujoco.mj_step(mj_model, mj_data)
    qpos_list = f['data'][traj_idx]['qpos']
    qvel_list = f['data'][traj_idx]['qvel']
    mocap_pos_list = f['data'][traj_idx]['mocap_pos']
    mocap_quat_list = f['data'][traj_idx]['mocap_quat']

    joint_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, f"{block_name}_freejoint")
    qpos_addr_tuple = (mj_model.jnt_qposadr[joint_id], mj_model.jnt_qposadr[joint_id] + 7)

    tgt_joint_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, f"vis_block_freejoint")
    tgt_qpos_addr_tuple = (mj_model.jnt_qposadr[tgt_joint_id], mj_model.jnt_qposadr[tgt_joint_id] + 7)

    obj_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, block_name)
    print(block_name, obj_id, joint_id)

    reference_frames = []
    for (qpos, qvel, mocap_pos, mocap_quat) in zip(qpos_list, qvel_list, mocap_pos_list, mocap_quat_list):
        replay_state(mj_model, mj_data, qpos, qvel, mocap_pos, mocap_quat)
        xpos = np.copy(mj_data.xpos[obj_id])
        xquat = np.copy(mj_data.xquat[obj_id])
        renderer.update_scene(mj_data, camera=0, scene_option=scene_option)
        pixels = renderer.render()
        reference_frames.append(pixels)

    media.set_show_save_dir('/tmp')
    media.show_video(reference_frames, fps=60, title="Ground-truth tower")

    for start in range(seq_len, obj_pose.shape[0] - rollout_len, interval):
        print(f"\nModel rollout from t={start} to t={start + rollout_len}")
        rollout_frames = []

        # seed histories from GT [start-seq_len … start)
        seed0 = start - seq_len
        block_hist = [obj_pose[i] for i in range(seed0, start)]  # list of (6,)
        tool_hist = [tool_pose[i] for i in range(seed0, start)]  # (6,)
        force_hist = [force_fb[i] for i in range(seed0, start)]  # (6,)
        action_hist = [actions[i] for i in range(seed0, start)]  # (6,)

        for step in range(rollout_len):
            t = start + step

            # build model input from the *histories*
            x_state = np.concatenate([
                np.stack(block_hist),  # (seq_len,6)
                np.stack(tool_hist),  # (seq_len,6)
                np.stack(force_hist),  # (seq_len,6)
            ], axis=1)  # → (seq_len,18)
            a_seq = np.stack(action_hist)  # (seq_len,6)

            data = {
                "x_seq": torch.from_numpy(x_state[None]).float().to(device),  # (1, seq_len,18)
                "a_seq": torch.from_numpy(a_seq[None]).float().to(device),  # (1, seq_len, 6)
            }

            # predict next block pose (pos+axis–angle)
            with torch.no_grad():
                pred = dynamics_model.predict(data).cpu().numpy()[0]  # (6,)
            tgt_qpos = np.concatenate((pred[:3], convert_quat(axisangle2quat(pred[3:]), to="wxyz")), axis=-1)

            # replay the true robot & tool state at time t
            replay_state(
                mj_model, mj_data,
                qpos_list[t],
                qvel_list[t],
                mocap_pos_list[t],
                mocap_quat_list[t]
            )

            # override vis_block
            mj_data.qpos[tgt_qpos_addr_tuple[0]:tgt_qpos_addr_tuple[1]] = tgt_qpos
            mj_data.qvel[tgt_qpos_addr_tuple[0]:tgt_qpos_addr_tuple[1]] = 0.0
            mujoco.mj_forward(mj_model, mj_data)

            # render and store frame
            renderer.update_scene(mj_data, camera=0, scene_option=scene_option)
            rollout_frames.append(renderer.render())

            # slide the window: drop oldest, append newest
            block_hist.pop(0)
            block_hist.append(pred)  # use model’s 6-D axis–angle pose

            tool_hist.pop(0)
            tool_hist.append(tool_pose[t])

            force_hist.pop(0)
            force_hist.append(force_fb[t])

            action_hist.pop(0)
            action_hist.append(actions[t])

        # display this rollout
        # media.show_video(
        #     rollout_frames,
        #     fps=60,
        #     title=f"Model roll-out @ t={start}→{start + rollout_len}"
        # )
        
        # save this rollout to folder
        output_folder = os.path.join(experiment_path, "rollout_videos")
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, f"model_rollout_t{start}_{start + rollout_len}.mp4")
        media.write_video(
            output_path,
            rollout_frames,
            fps=60
        )
        print(f"Saved rollout video to: {output_path}")