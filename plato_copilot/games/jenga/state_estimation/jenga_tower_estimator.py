import os
import cv2
import numpy as np
from PIL import Image
from easydict import EasyDict
from fontTools.ttx import process
from jinja2.compiler import generate

from plato_copilot.vision.img_utils import *
from plato_copilot.vision.point_cloud_utils import *
from plato_copilot.games.jenga.state_estimation.jenga_block_detector import JengaBlockDetector
from plato_copilot.games.jenga.jenga_tower_state import JengaTowerState
from plato_copilot.utils.log_utils import get_copilot_logger

logger = get_copilot_logger()

# camera_intrinsics = EasyDict(
#
#     {"fx": 614.0392456054688, "fy": 614.16455078125, "cx": 329.4415283203125, "cy": 241.68116760253906})

class JengaTowerEstimator():
    def __init__(self, camera_intrinsics, config):
        # TODO: 1880-2330
        self.config = config
        self.jenga_tower_states = JengaTowerState()

        # JengaBlockDetector detects individual faces. We need to gather this information and esitmate individual blocks.
        self.jenga_block_detector = JengaBlockDetector()
        # self.sam = SAM()
        # self.pt_tracker = PTTracker()
        self.rgb_sequence = []
        self.depth_sequence = []

        self.camera_matrix = np.array(
            [[camera_intrinsics.fx, 0, camera_intrinsics.cx], [0, camera_intrinsics.fy, camera_intrinsics.cy],
             [0, 0, 1]],
            dtype=np.float32)

        logger.debug("Define configuration for running segment-anything")
        sam_config = EasyDict({
            "sam_checkpoint": "../third_party/sam_checkpoints/sam_vit_b_01ec64.pth",
            "model_type": "vit_b",
            "device": "cuda",

        })
        logger.debug("Create JengaBlockDetector object for detecting jenga blocks at the initial observation")
        self.detector = JengaBlockDetector(self.camera_matrix, sam_config, num_runs=100)

    def reset(self):
        self.rgb_sequence = []
        self.depth_sequence = []

    def obs(self, img_obs):
        logger.debug("Save image observations into a history buffer for processing tracking.")

        logger.debug("Update rgb sequence")
        self.rgb_sequence.append(img_obs["rgb"])
        logger.debug("Update depth sequence")
        self.depth_sequence.append(img_obs["depth"])
        # if not track:
        #     jenga_block_poses = self.detect(img_obs)
        # else:
        #     previous_jenga_block_pose_history = jenga_block_poses
        #     new_annotation = self.track(img_obs)

    def detect(self):

        logger.debug("Detecting Jenga blocks faces from the initial observation")

        # using faces to reconstruct and estimate the poses of each jenga block.
        logger.debug("Estimate poses of the Jenga blocks based on the detected faces")

        logger.debug("Update jenga_tower_state")

        # self.jenga_block_poses = self.estimate_poses(self.faces)

    def add_files(self, extension, folder):
        files = [f for f in os.listdir(folder) if f.endswith(extension)]
        for file in files:
            if extension == ".tiff":
                depth_image = np.array(load_depth(os.path.join(folder, file)), dtype=np.float32)
                self.depth_sequence.append(depth_image)
            else:
                img = np.array(Image.open(os.path.join(folder, file)))
                self.rgb_sequence.append(img)

    def generate_point_prompts(self, coord, layer, layer_orientation, detection_info):
        # along vector currently always points from left to right
        block_height = detection_info["probable_block_height"]
        block_width = detection_info["block_width"] / 3.0
        primary_face_top_left = detection_info["primary_face_top_left"]
        primary_face_along = detection_info["primary_face_basis"][0] # this is left to right
        respective_face_along = primary_face_along # this points from the middle to edge of the primary face
        primary_face_down = detection_info["primary_face_basis"][1]


        if detection_info["primary_face_side"]:
            respective_face_along = -respective_face_along

        if layer_orientation:
            # horizontal
            tower_middle_column_top = primary_face_top_left + primary_face_along * block_width * 1.5
            layer_z_vector = primary_face_down * block_height * (layer + 0.5)
            layer_start_point = tower_middle_column_top + layer_z_vector

            return generate_point_prompt(layer_start_point, primary_face_along, primary_face_down, top_pt=(layer == 0), bot_pt=(layer == detection_info["num_layers"] - 1))
        else:
            # vertical

            pass


    def determine_location(self, tower: JengaTowerState, block_id, detection_info):
        # given the tower, coordinate, and tower info
        # figure out a mask for the given block
        logger.warn("Assuming block is on the primary face for now!")
        block_coordinate = tower.get_block_coordinates(block_id)
        num_layers = detection_info["num_layers"]
        top_layer_orientation = detection_info["top_horizontal_to_camera"]

        layer = num_layers - block_coordinate[2]
        block = block_coordinate[0] if layer % 2 == 1 else block_coordinate[1]

        # determine whether layer is horizontal or vertical
        # True -> horizontal, False -> vertical
        layer_orientation = top_layer_orientation if layer % 2 == 0 else not top_layer_orientation

        # get the non top and bottom faces of the block
        block_faces_x, block_faces_y, _ = tower.get_block_faces(block_id)

        assert len(block_faces_x) == 2
        assert len(block_faces_y) == 2

        # based on the tower orientation, get the one or two faces that can be seen
        primary_face_basis = detection_info["primary_face_basis"]
        non_primary_face_basis = detection_info["non_primary_face_basis"]

        final_faces = []
        if layer_orientation:
            # horizontal
            if top_layer_orientation:
                max_x_face = max(block_faces_x.keys())
                final_faces.append(block_faces_x[max_x_face])

                max_y_face = max(block_faces_y.keys())
                if abs(max_y_face - tower.block_width) < 0.01:
                    final_faces.append(block_faces_y[max_y_face])
            else:
                # get the max-x face, there are two keys in the block_faces_x dictionary
                max_x_face = max(block_faces_x.keys())
                if abs(max_x_face - tower.block_width) < 0.01:
                    final_faces.append(block_faces_x[max_x_face])

                min_y_face = min(block_faces_y.keys())
                final_faces.append(block_faces_y[min_y_face])
        else:
            # vertical
            if top_layer_orientation:
                max_x_face = max(block_faces_x.keys())
                if abs(max_x_face - tower.block_width) < 0.01:
                    final_faces.append(block_faces_y[max_x_face])

                max_y_face = max(block_faces_y.keys())
                final_faces.append(block_faces_x[max_y_face])
            else:
                max_x_face = max(block_faces_x.keys())
                final_faces.append(block_faces_x[max_x_face])

                min_y_face = min(block_faces_y.keys())
                if abs(min_y_face - 0) < 0.01:
                    final_faces.append(block_faces_y[min_y_face])

        # TODO: turn these faces into masks in the image

    def track_block(self, block_coordinate, video_folder):
        logger.warn("Tracking currently is for a full video, not a live stream yet")
        logger.warn("Currently only tracking blocks visible on the primary face")
        # load the images from the folder into the array
        self.add_files(".jpg", video_folder)
        self.add_files(".tiff", video_folder)

        processed = 0

        # first do a detect on one image and get the necessary tower specs
        detection_result = self.detector.segment(rgb_image=self.rgb_sequence[processed], depth_image=self.depth_sequence[processed])
        self.jenga_tower_states = self.detector.get_jenga_tower(rgb_image=self.rgb_sequence[processed], depth_image=self.depth_sequence[processed])

        # generate the mask for the block of interest
        mask = self.determine_location(self.jenga_tower_states, block_coordinate, detection_result)

        # pass through SAM2 video predictor and keep track of mask


        # use open3d library to do 3d back projection
        pass


    def track(self):
        logger.debug("Track the Jenga blocks within a video stream.")

        logger.debug("Step 1: Track the segmentation of jenga blocks in the video. ")
        logger.debug("Step 2: Track the keypoints of the jenga blocks in the video. ")

        logger.debug("Step 3: Filter keypoints that fall outside of the segmentation masks")
        logger.debug("Step 4: Back-project the keypoints to 3D space")

        logger.debug("Step 5: Estimate the jenga block poses from the 3D trajectories")

        # Cutie and Track-Any-Point tracker

        # self.tracked_faces = self.cutie(img_os, self.sam_annotation)
        # self.tracked_jenga_block_poses = self.track_pose(self.jenga_block_poses,
        #                                                                                         self.tracked_faces)
        # return self.tracked_jenga_block_poses