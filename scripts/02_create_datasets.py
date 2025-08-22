import h5py
import os
import numpy as np
os.environ["MUJOCO_GL"] = "egl"

import mujoco
from IPython.display import HTML

import mediapy as media
from plato.vision.o3d_utils import O3DPointCloud
from plato.vision.plotly_utils import plotly_draw_3d_pcd

from robosuite.utils.transform_utils import quat2axisangle, convert_quat
import argparse

def replay_state(mj_model, mj_data, qpos, qvel, mocap_pos, mocap_quat):
    mujoco.mj_setState(mj_model, mj_data, qpos, mujoco.mjtState.mjSTATE_QPOS)
    mujoco.mj_setState(mj_model, mj_data, qvel, mujoco.mjtState.mjSTATE_QVEL)
    mujoco.mj_setState(mj_model, mj_data, np.squeeze(mocap_pos), mujoco.mjtState.mjSTATE_MOCAP_POS)
    mujoco.mj_setState(mj_model, mj_data, np.squeeze(mocap_quat), mujoco.mjtState.mjSTATE_MOCAP_QUAT)
    mujoco.mj_step1(mj_model, mj_data)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="datasets/jenga_sim_data/example/", help="Path to the dataset")
    args = parser.parse_args()

    dataset_path = args.dataset_path
    scene_option = mujoco.MjvOption()
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = False

    with h5py.File(f'{dataset_path}/training_data.hdf5', 'w') as training_data_file:
        data_grp = training_data_file.create_group("data")

        count = 0
        with h5py.File(f'{dataset_path}/data.hdf5', 'r') as f:
            frames = []
            for demo in list(f['data'].keys()):
                # print(demo)
                xml = f['data'][demo].attrs['xml']
                block_name = f['data'][demo].attrs['block_name']

                mj_model = mujoco.MjModel.from_xml_string(xml)
                mj_data = mujoco.MjData(mj_model)

                joint_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, f"{block_name}_freejoint")
                qpos_addr_tuple = (mj_model.jnt_qposadr[joint_id], mj_model.jnt_qposadr[joint_id] + 7)

                qpos = f['data'][demo]['qpos']
                target_pos = qpos[:, qpos_addr_tuple[0]:qpos_addr_tuple[1]]

                object_pos = target_pos[:, :3]
                object_quat = target_pos[:, 3:]
                object_axisAngle = np.array([quat2axisangle(convert_quat(q, to="xyzw")) for q in object_quat])
                # print(block_name, target_pos[0])
                # renderer = mujoco.Renderer(mj_model, 480, 640)
                ep_grp = data_grp.create_group(f"demo_{count}")
                ep_grp.attrs["traj_idx"] = demo
                ep_grp.create_dataset("object_pose", data=np.concatenate([object_pos, object_axisAngle], axis=1))
                # ep_grp.create_dataset("object_pos", data=target_pos[:, :3])
                # ep_grp.create_dataset("object_quat", data=target_pos[:, 3:])
                ep_grp.create_dataset("actions", data=f['data'][demo]['actions'])
                count = count + 1
            print(target_pos.shape)

        training_data_file.close()

if __name__ == "__main__":
    main()