import mujoco

import init_path
from plato_copilot.games.jenga.simulation.jenga_generator import JengaXMLGenerator
from easydict import EasyDict
import os

from datetime import datetime
import functools
import jax
import random
from jax import numpy as jp
import numpy as np
from typing import Any, Dict, Sequence, Tuple, Union

# mjx implementation. Commented out for now. 
# from brax import base
# from brax import envs
# from brax import math
# from brax.base import Base, Motion, Transform
# from brax.envs.base import Env, PipelineEnv, State
# from brax.mjx.base import State as MjxState
# from brax.training.agents.ppo import train as ppo
# from brax.training.agents.ppo import networks as ppo_networks
# from brax.io import html, mjcf, model

from etils import epath
from flax import struct
from matplotlib import pyplot as plt
# from ml_collections import config_dict
import mujoco
from mujoco import viewer
# from mujoco import mjx
from tqdm import trange
import h5py
import json

import argparse

from plato_copilot.utils.log_utils import get_copilot_logger
logger = get_copilot_logger()


def sample_environment_variables(sampling_ranges: EasyDict):
    env_vars = {}
    for key, value in sampling_ranges.items():
        env_vars[key] = np.random.uniform(value[0], value[1])

    geom_attributes = {
        'group': "2",
        'friction': ' '.join(
            map(str, [env_vars['friction_sliding'], env_vars['friction_torsional'], env_vars['friction_rolling']])),
        'density': str(env_vars['density']),
        'solref': "0.02 1",
        'solimp': " ".join(map(str, [env_vars['sol_contact_normal'], env_vars['sol_contact_tangential'], env_vars['sol_damping'], env_vars['sol_stiffness'], 2])),
        'material': "geom",
    }

    return geom_attributes

def sample_jenga_block_sizes():
    width = np.random.uniform(0.02, 0.1)
    length = width * 3
    width_height_ratio = np.random.uniform(0.7, 1.3)
    return width, length, width / width_height_ratio

def main():
    parser = argparse.ArgumentParser(description='Generate Jenga data')
    parser.add_argument('--process-id', type=int, default=0, help='process id')
    parser.add_argument('--data-folder', required=True, help='data folder')
    parser.add_argument('--num-rollouts', type=int, default=4, help='number of rollouts')
    args = parser.parse_args()

    debug = False
    render = False

    config = EasyDict({
        "floor_pos": np.array([0, 0, 0.28]),
        'num_layer': 8,
        "mocap": {
            "pos": np.array([-0.2, 0.01, 0.60]),
            "quat": np.array([0.707, 0, 0.707, 0]),
            "size": np.array([0.015, 0.1]),
            "density": str(500),
            "rgba": np.array([0.8, 0.8, 0.8, 0.1])
        }
    })

    sampling_ranges = EasyDict({
        "friction_sliding": [0.25, 0.35],
        "friction_torsional": [0.004, 0.006],
        "friction_rolling": [0.00009, 0.00011],
        "density": [0.9, 1.1],
        "sol_contact_normal": [0.9, 1],
        "sol_contact_tangential": [0.9, 1],
        "sol_damping": [0.004, 0.006],
        "sol_stiffness": [0.85, 0.95]
    })

    # Example usage
    xml_gen = JengaXMLGenerator(debug=debug)
    xml_gen.generate_xml(config)
    xml = xml_gen.prettify()
    scene_option = mujoco.MjvOption()
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = False

    duration = 10  # (seconds)
    framerate = 50 # (HZ)
    timestep = 0.002
    num_steps = int(1 / framerate / timestep)

    num_rollouts = args.num_rollouts
    print(mujoco.MjvCamera.lookat)
    # get the camera position
    print(mujoco.MjvCamera.distance)

    data_example_file = h5py.File(f'{args.data_folder}/{args.process_id:03d}.hdf5', 'w')
    data_grp = data_example_file.create_group('data')

    use_straight_line = False

    for i in trange(num_rollouts):
        frames = []

        geom_attributes = sample_environment_variables(sampling_ranges)
        # logger.info(f"Process id {args.process_id}, Rollout {i}, geom_attributes: {geom_attributes}")
        block_sizes = sample_jenga_block_sizes()

        xml_gen.generate_xml(config, geom_attributes=geom_attributes, block_sizes=block_sizes)
        xml = xml_gen.prettify()
        mj_model = mujoco.MjModel.from_xml_string(xml)
        mj_data = mujoco.MjData(mj_model)
        renderer = mujoco.Renderer(mj_model, 480, 640)

        mocap_body_id = mujoco.mj_name2id(
            mj_model,
            mujoco.mjtObj.mjOBJ_BODY,
            "mocap"
        )

        mujoco.mj_resetData(mj_model, mj_data)
        # mocap_y = np.random.uniform(0.01, 0.2)
        # mocap_z = np.random.uniform(0.35, 0.6)
        # config.mocap.pos[1] = mocap_y
        # config.mocap.pos[2] = mocap_z
        if render:
            viewer = mujoco.viewer.launch_passive(mj_model, mj_data)

            # Enable wireframe rendering of the entire scene.
            viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_WIREFRAME] = 1
            viewer.sync()

        qpos_list = []
        qvel_list = []
        time_list = []
        mocap_pos_list = []
        mocap_quat_list = []
        actions = []
        force_feedback_list = []

        for _ in range(2000):
            mujoco.mj_step(mj_model, mj_data)
        block_name = random.sample(xml_gen.odd_layer_block_names, 1)[0]
        block_pos = mj_data.xpos[mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, block_name)]
        mocap_y = block_pos[1]
        mocap_z = block_pos[2]

        mj_data.mocap_pos = np.array([[-block_sizes[1] / 2 - 0.05, mocap_y, mocap_z]])

        linear_vel = np.random.uniform(low=[0.02, 0.0, 0.0],
                                       high = [0.05, 0.0, 0.0])  # m/s

        angular_vel = np.random.uniform(low=[0.0, 0.0, -0.05],
                                        high = [0.0, 0.0, 0.05])  # rad/s

        lin_noise_std = 0.001  # m/s
        ang_noise_std = 0.02  # rad/s

        count = 0
        init_time = mj_data.time
        while mj_data.time - init_time < duration:

            qpos_list.append(np.copy(mj_data.qpos))
            qvel_list.append(np.copy(mj_data.qvel))
            time_list.append(mj_data.time)
            mocap_pos_list.append(np.copy(mj_data.mocap_pos))
            mocap_quat_list.append(np.copy(mj_data.mocap_quat))
            mujoco.mj_rnePostConstraint(mj_model, mj_data)
            force_feedback_list.append(np.copy(mj_data.cfrc_ext[mocap_body_id]))

            # if len(frames) < mj_data.time * framerate and render:
            renderer.update_scene(mj_data, camera=0, scene_option=scene_option)
            pixels = renderer.render()
            frames.append(pixels)

            body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, block_name)
            joint_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, f"{block_name}_freejoint")
            freejoint_qpos_addr = mj_model.jnt_qposadr[joint_id]

            lv_noisy = linear_vel  + np.random.normal(0, lin_noise_std,  size=3)
            av_noisy = angular_vel + np.random.normal(0, ang_noise_std, size=3)

            lin_step = lv_noisy * timestep  # tool-frame
            ang_step = av_noisy * timestep  # tool-frame

            # action = action + np.random.uniform(-0.001, 0.001, action.shape) * np.array([1, 1, 0]) * (
            #             len(actions) % 10 == 0)
            # actions.append(action)
            # unit_action = action / freq
            for _ in range(num_steps):
                mujoco.mj_step(mj_model, mj_data)

                # translate: rotate the tool-frame lin_step into world frame
                q = mj_data.mocap_quat[0]  # [w,x,y,z]
                v_body = lin_step  # (3,)
                qw, qx, qy, qz = q
                qv = np.array([qx, qy, qz])
                t = 2.0 * np.cross(qv, v_body)
                v_world = v_body + qw * t + np.cross(qv, t)
                mj_data.mocap_pos += v_world  # updates (1,3)

                # rotate: small-angle quaternion about tool Z
                ω = ang_step  # [0,0,ωz*dt]
                θ = np.linalg.norm(ω)
                if θ > 1e-8:
                    axis = ω / θ
                    dq = np.array([
                        np.cos(θ / 2.0),
                        *(axis * np.sin(θ / 2.0))
                    ])
                else:
                    dq = np.array([1.0, 0, 0, 0])

                # right-multiply into current tool quat
                qc = mj_data.mocap_quat[0]
                w1, x1, y1, z1 = qc
                w0, x0, y0, z0 = dq
                new_q = np.array([
                    w1 * w0 - x1 * x0 - y1 * y0 - z1 * z0,
                    w1 * x0 + x1 * w0 + y1 * z0 - z1 * y0,
                    w1 * y0 - x1 * z0 + y1 * w0 + z1 * x0,
                    w1 * z0 + x1 * y0 - y1 * x0 + z1 * w0,
                ])
                mj_data.mocap_quat[0] = new_q / np.linalg.norm(new_q)

                if render:
                    viewer.user_scn.ngeom = i
                    viewer.sync()

            #  compute the 7-D action (tool frame)
            action = np.concatenate([lin_step * num_steps, ang_step * num_steps])
            actions.append(action)

            count += 1

        print(f'demo_{i}')
        demo_grp = data_grp.create_group(f'demo_{i}')
        demo_grp.create_dataset('qpos', data=qpos_list)
        demo_grp.create_dataset('qvel', data=qvel_list)
        demo_grp.create_dataset('mocap_pos', data=mocap_pos_list)
        demo_grp.create_dataset('mocap_quat', data=mocap_quat_list)
        demo_grp.create_dataset('force_feedback', data=force_feedback_list)
        demo_grp.create_dataset('actions', data=actions)
        demo_grp.attrs["xml"] = xml
        demo_grp.attrs["block_name"] = block_name
        demo_grp.attrs["geom_attributes"] = json.dumps(geom_attributes)
        # info = {
        #     "joint_id": joint_id,
        #     "freejoint_qpos_addr": freejoint_qpos_addr
        # }
        # demo_grp.attrs["info"] = json.dumps(info)

    data_example_file.close()


if __name__ == "__main__":
    main()
