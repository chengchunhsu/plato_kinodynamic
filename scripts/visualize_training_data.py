import h5py
import os
import torch
import numpy as np
os.environ["MUJOCO_GL"] = "egl"

import mujoco
from IPython.display import HTML

import mediapy as media
from plato_copilot.vision.o3d_utils import O3DPointCloud
from plato_copilot.vision.plotly_utils import plotly_draw_3d_pcd

from robosuite.utils.transform_utils import convert_quat, quat2axisangle, axisangle2quat


trajs = []
dataset_path = "./dataset"
with h5py.File(os.path.join(dataset_path, "training_data.hdf5"), "r") as f:
    for key in f["data"].keys():
        trajs.append((
            torch.tensor(f["data"][key]["object_pose"][()]),
            torch.tensor(f["data"][key]["actions"][()]),
            f["data"][key].attrs["traj_idx"],
            f["data"][key].attrs["geom_attributes"],
        ))


traj_idx = 37
max_len = 10

predicted_y = []
# dynamics_model.use_low_noise()
for i in range(90):
    object_pose = trajs[traj_idx][0][i:i+max_len].unsqueeze(0)
    action = trajs[traj_idx][1][i:i+max_len].unsqueeze(0)
    data = {
        "x_seq": object_pose.float(),
        "a_seq": action.float()
    }
    # y = dynamics_model.predict(data)
    # predicted_y.append(y[0].detach().cpu().numpy())
# predicted_y = np.squeeze(np.stack(predicted_y))
print(trajs[traj_idx][3])
# print(np.round(predicted_y, 2))
from plato_copilot.vision.plotly_utils import plotly_draw_3d_pcd
plotly_draw_3d_pcd(trajs[traj_idx][0])
# plotly_draw_3d_pcd(predicted_y[:, :3])


def replay_state(mj_model, mj_data, qpos, qvel, mocap_pos, mocap_quat):
    mujoco.mj_setState(mj_model, mj_data, qpos, mujoco.mjtState.mjSTATE_QPOS)
    mujoco.mj_setState(mj_model, mj_data, qvel, mujoco.mjtState.mjSTATE_QVEL)
    mujoco.mj_setState(mj_model, mj_data, np.squeeze(mocap_pos), mujoco.mjtState.mjSTATE_MOCAP_POS)
    mujoco.mj_setState(mj_model, mj_data, np.squeeze(mocap_quat), mujoco.mjtState.mjSTATE_MOCAP_QUAT)
    mujoco.mj_step1(mj_model, mj_data)

average_len = []
scene_option = mujoco.MjvOption()
scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = False
with h5py.File(f'{dataset_path}/data.hdf5', 'r') as f:
    frames = []
    # demo = f"traj_{traj_idx}"
    demo = trajs[traj_idx][2]
    xml = f['data'][demo].attrs['xml']
    block_name = f['data'][demo].attrs['block_name']

    mj_model = mujoco.MjModel.from_xml_string(xml)
    mj_data = mujoco.MjData(mj_model)
    renderer = mujoco.Renderer(mj_model, 480, 640)

    mujoco.mj_resetData(mj_model, mj_data)
    mujoco.mj_step(mj_model, mj_data)
    qpos_list = f['data'][demo]['qpos']
    qvel_list = f['data'][demo]['qvel']
    mocap_pos_list = f['data'][demo]['mocap_pos']
    mocap_quat_list = f['data'][demo]['mocap_quat']

    joint_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, f"{block_name}_freejoint")
    qpos_addr_tuple = (mj_model.jnt_qposadr[joint_id], mj_model.jnt_qposadr[joint_id] + 7)

    tgt_joint_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, f"vis_block_freejoint")
    tgt_qpos_addr_tuple = (mj_model.jnt_qposadr[tgt_joint_id], mj_model.jnt_qposadr[tgt_joint_id] + 7)

    obj_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, block_name)
    print(block_name, obj_id, joint_id)

    qpos_list = f['data'][demo]['qpos']
    qvel_list = f['data'][demo]['qvel']
    mocap_pos_list = f['data'][demo]['mocap_pos']
    mocap_quat_list = f['data'][demo]['mocap_quat']
    target_block_pos = []
    target_block_quat = []
    reference_frames = []
    for (qpos, qvel, mocap_pos, mocap_quat) in zip(qpos_list, qvel_list, mocap_pos_list, mocap_quat_list):
        replay_state(mj_model, mj_data, qpos, qvel, mocap_pos, mocap_quat)
        xpos = np.copy(mj_data.xpos[obj_id])
        xquat = np.copy(mj_data.xquat[obj_id])
        target_block_pos.append(xpos)
        target_block_quat.append(xquat)
        renderer.update_scene(mj_data, camera=0, scene_option=scene_option)            
        pixels = renderer.render()
        reference_frames.append(pixels)

    # y_list = [predicted_y[0]] * max_len + predicted_y.tolist()
    
    # for qpos, qvel, xpos, xquat, y, mocap_pos, mocap_quat in zip(qpos_list, qvel_list, target_block_pos, target_block_quat, y_list, mocap_pos_list, mocap_quat_list):
    #     # qpos = qpos_list[0]
    #     # qvel = qvel_list[0]
    #     qpos = np.zeros_like(qpos)
        
    #     reference_qpos = np.concatenate((xpos, xquat), axis=-1)
    #     qpos[qpos_addr_tuple[0]:qpos_addr_tuple[1]] = reference_qpos

    #     tgt_qpos = np.concatenate((y[:3], convert_quat(axisangle2quat(y[3:]), to="wxyz")), axis=-1)
    #     qpos[tgt_qpos_addr_tuple[0]:tgt_qpos_addr_tuple[1]] = tgt_qpos

    #     replay_state(mj_model, mj_data, qpos, qvel, mocap_pos, mocap_quat)
        
    #     renderer.update_scene(mj_data, camera=0, scene_option=scene_option)            
    #     pixels = renderer.render()
    #     frames.append(pixels)

    
    # count = 0
    # import cv2
    # os.makedirs("../experiments/demos/", exist_ok=True)

    # for qpos, qvel, xpos, xquat, y, mocap_pos, mocap_quat in zip(qpos_list, qvel_list, target_block_pos, target_block_quat, y_list, mocap_pos_list, mocap_quat_list):
    #     # qpos = qpos_list[0]
    #     # qvel = qvel_list[0]
    #     qpos = np.zeros_like(qpos)
        
    #     reference_qpos = np.concatenate((xpos, xquat), axis=-1)
    #     qpos[qpos_addr_tuple[0]:qpos_addr_tuple[1]] = reference_qpos

    #     tgt_qpos = np.concatenate((y[:3], convert_quat(axisangle2quat(y[3:]), to="wxyz")), axis=-1)
    #     qpos[tgt_qpos_addr_tuple[0]:tgt_qpos_addr_tuple[1]] = tgt_qpos

    #     replay_state(mj_model, mj_data, qpos, qvel, mocap_pos, mocap_quat)
        
    #     renderer.update_scene(mj_data, camera=0, scene_option=scene_option)            
    #     pixels = renderer.render()
    #     cv2.imwrite(f"../experiments/demos/{count}.png", pixels[..., ::-1])
    #     count += 1
    #     if count > 50:
    #         break
            
# Save videos to output folder
output_folder = "experiments/training_videos"
os.makedirs(output_folder, exist_ok=True)
output_path = os.path.join(output_folder, "training_data_visualization.mp4")
media.write_video(
    output_path,
    reference_frames,
    fps=60
)
print(f"Saved training data video to: {output_path}")

# media.set_show_save_dir('/tmp')
# media.show_video(frames, fps=60, title="Block predicted vs expected")
# media.show_video(reference_frames, fps=60, title="Tower expected")