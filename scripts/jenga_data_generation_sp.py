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

    duration = 4  # (seconds)
    framerate = 60  # (Hz)

    num_rollouts = args.num_rollouts
    print(mujoco.MjvCamera.lookat)
    # get the camera position
    print(mujoco.MjvCamera.distance)

    render = True

    data_example_file = h5py.File(f'{args.data_folder}/{args.process_id:03d}.hdf5', 'w')
    data_grp = data_example_file.create_group('data')

    use_straight_line = True

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

        mujoco.mj_resetData(mj_model, mj_data)
        # mocap_y = np.random.uniform(0.01, 0.2)
        # mocap_z = np.random.uniform(0.35, 0.6)
        # config.mocap.pos[1] = mocap_y
        # config.mocap.pos[2] = mocap_z
        qpos_list = []
        qvel_list = []
        time_list = []
        mocap_pos_list = []
        mocap_quat_list = []
        actions = []
        freq = 20

        for _ in range(2000):
            mujoco.mj_step(mj_model, mj_data)
        block_name = random.sample(xml_gen.odd_layer_block_names, 1)[0]
        block_pos = mj_data.xpos[mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, block_name)]
        mocap_y = block_pos[1]
        mocap_z = block_pos[2]

        mocap_init_pos = np.array([[-block_sizes[1] - 0.05, mocap_y, mocap_z]])
        mj_data.mocap_pos = mocap_init_pos

        count = 0
        init_time = mj_data.time
        while mj_data.time - init_time < duration:

            qpos_list.append(np.copy(mj_data.qpos))
            qvel_list.append(np.copy(mj_data.qvel))
            time_list.append(mj_data.time)
            mocap_pos_list.append(np.copy(mj_data.mocap_pos))
            mocap_quat_list.append(np.copy(mj_data.mocap_quat))

            # if len(frames) < mj_data.time * framerate and render:
            renderer.update_scene(mj_data, camera=0, scene_option=scene_option)
            pixels = renderer.render()
            frames.append(pixels)

            straight_line_action = np.array([[block_sizes[0] * 2, 0., -0.001]]) * 0.05

            curve_action_seq = [np.array([0.1, 0., -0.001]) * 0.05] * 200
            for j in range(30, len(curve_action_seq)):
                curve_action_seq[j] = curve_action_seq[j] + np.array([0., 0.1, 0.0]) * 0.05

            if use_straight_line:
                action = straight_line_action
            else:
                action = curve_action_seq[count]
            # action = np.array([[0.1, 0., -0.001]]) * 0.05

            body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, block_name)
            joint_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, f"{block_name}_freejoint")
            freejoint_qpos_addr = mj_model.jnt_qposadr[joint_id]

            action = action + np.random.uniform(-0.001, 0.001, action.shape) * np.array([1, 1, 0]) * (
                        len(actions) % 10 == 0)
            actions.append(action)
            unit_action = action / freq
            for _ in range(freq):
                mujoco.mj_step(mj_model, mj_data)
                mj_data.mocap_pos = mj_data.mocap_pos + unit_action
            count += 1

        # mjx implementation. Commented out for now. 
        # while mj_data.time < duration:
        #     mujoco.mj_step(mj_model, mj_data)
        #     qpos_list.append(np.copy(mj_data.qpos))
        #     qvel_list.append(np.copy(mj_data.qvel))
        #     time_list.append(mj_data.time)
        #     mocap_pos_list.append(np.copy(mj_data.mocap_pos))
        #     mocap_quat_list.append(np.copy(mj_data.mocap_quat))
        #     if len(frames) < mj_data.time * framerate:
        #         # renderer.update_scene(mj_data, camera=0, scene_option=scene_option)
        #         # pixels = renderer.render()
        #         # frames.append(pixels)
        #         mj_data.mocap_pos += np.array([[0.7, 0., -0.001]]) / (duration * framerate)

        print(f'demo_{i}')
        demo_grp = data_grp.create_group(f'demo_{i}')
        demo_grp.create_dataset('qpos', data=qpos_list)
        demo_grp.create_dataset('qvel', data=qvel_list)
        demo_grp.create_dataset('mocap_pos', data=mocap_pos_list)
        demo_grp.create_dataset('mocap_quat', data=mocap_quat_list)
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
