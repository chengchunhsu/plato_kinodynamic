import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from easydict import EasyDict
from fontTools.ttx import process
from jinja2.compiler import generate
from sam2.modeling import sam2_base
from sam2.build_sam import build_sam2, build_sam2_video_predictor

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
    def __init__(self, camera_intrinsics, config, video_dir=None):
        self.config = config
        self.jenga_tower_states = JengaTowerState()

        # JengaBlockDetector detects individual faces. We need to gather this information and esitmate individual blocks.
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

        checkpoint = "../third_party/sam_checkpoints/sam2.1_hiera_large.pt"
        model_cfg = "sam2.1_hiera_l.yaml"
        self.predictor = build_sam2_video_predictor(model_cfg, checkpoint)
        self.video_dir = video_dir

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
        files.sort()
        for file in files:
            if extension == ".tiff":
                depth_image = np.array(load_depth(os.path.join(folder, file)), dtype=np.float32)
                self.depth_sequence.append(depth_image)
            else:
                img = np.array(Image.open(os.path.join(folder, file)))
                self.rgb_sequence.append(img)

    def get_closest_block(self, img_coord_of_block, block_masks):
        # get the closest block to the camera
        mn_dist = 10000
        closest_block = (-1, -1)  # layer and block number
        for layer in block_masks:
            for block in block_masks[layer]:
                total_dist = 0
                total_cnt = 0
                for coord in block_masks[layer][block]:
                    dist = np.linalg.norm(coord[:2] - img_coord_of_block)
                    if dist < 50:
                        total_dist += dist
                        total_cnt += 1

                if total_cnt == 0:
                    continue

                dist_avg = total_dist / total_cnt

                if dist_avg < mn_dist:
                    mn_dist = dist_avg
                    closest_block = (layer, block)

        return closest_block

    def track_block(self, img_coord_of_block, video_folder):
        logger.warn("Tracking currently is for a full video, not a live stream yet")
        logger.warn("Currently only tracking blocks visible on the primary face")
        # load the images from the folder into the array
        self.add_files(".jpg", video_folder)
        self.add_files(".tiff", video_folder)

        processed = 0

        # first get the masks of all the blocks we can see
        logger.info("Getting the masks of all the blocks in the first frame of the video")
        block_masks, seg_img = self.detector.segment_all_faces(rgb_image=self.rgb_sequence[processed], depth_image=self.depth_sequence[processed])

        # get the closest block to the camera
        logger.info("Getting the closest block coordinate to the user selected block")
        closest_block = self.get_closest_block(img_coord_of_block, block_masks)
        logger.debug(f"Closest block is: {closest_block}")

        # get the mask of the closest block for the image
        logger.info("Getting the mask of the closest block to the user selected block")
        mask = np.zeros(self.rgb_sequence[processed].shape[:2], dtype=np.float32)
        for coord in block_masks[closest_block[0]][closest_block[1]]:
            mask[int(coord[1]), int(coord[0])] = 1

        # pass through SAM2 video predictor and keep track of mask
        logger.info("Tracking the block through the video")
        video_segments = {}  # video_segments contains the per-frame segmentation results

        # array that stores the masks of the selected block throughout the video
        block_states = []
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            state = self.predictor.init_state(video_folder)

            # add new prompts and instantly get the output on the same frame
            frame_idx, object_ids, masks = self.predictor.add_new_mask(frame_idx=0, obj_id=1, inference_state=state, mask=mask) #(state, None):

            # propagate the prompts to get masklets throughout the video
            for frame_idx, object_ids, masks in self.predictor.propagate_in_video(state):
                video_segments[frame_idx] = {
                    out_obj_id: (masks[i] > 0.0).cpu().numpy()
                    for i, out_obj_id in enumerate(object_ids)
                }

        # visualize the segmentation results
        vis_frame_stride = 1
        plt.close("all")
        for out_frame_idx in range(1, len(self.rgb_sequence), vis_frame_stride):
            plt.figure(figsize=(6, 4))
            plt.title(f"frame {out_frame_idx}")
            plt.imshow(Image.open(os.path.join(video_folder, "{:09}".format(out_frame_idx) + ".jpg")))
            for out_obj_id, out_mask in video_segments[out_frame_idx].items():
                show_mask(out_mask, plt.gca(), obj_id=out_obj_id)

            plt.savefig(f"../test_tmp/video_sam_frames/seg_{out_frame_idx}.png", dpi=300, bbox_inches='tight')
            plt.close()

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