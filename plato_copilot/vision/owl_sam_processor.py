import cv2
import numpy as np
import torch
from PIL import Image
import sys

from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import time
from functools import partial
from plato_copilot import PLATO_COPILOT_ROOT_PATH
from plato_copilot.vision.sam_operator import EfficientSAMOperator
from plato_copilot.vision.owl_detector import OwlDetector

from plato_copilot.vision.plotly_utils import *
from plato_copilot.vision.img_utils import *
import os


def resize_and_crop_depth_map(depth_map, new_size, bbox):
    """
    Resize a depth map and crop it using the given bounding box.
    
    Parameters:
    - depth_map: 2D NumPy array (H x W) representing depth in millimeters/meters
    - new_size: tuple (new_width, new_height)
    - bbox: tuple (x, y, width, height) in the resized image coordinates
    
    Returns:
    - cropped_depth: Cropped region of resized depth map
    """
    # Resize depth map (using nearest-neighbor to preserve depth integrity)
    resized_depth = cv2.resize(depth_map, new_size, interpolation=cv2.INTER_NEAREST)
    
    # Unpack bbox
    x, y, w, h = bbox
    
    # Ensure bbox is within image bounds
    x = max(0, x)
    y = max(0, y)
    w = min(w, new_size[0] - x)
    h = min(h, new_size[1] - y)
    
    # Crop the depth map
    cropped_depth = resized_depth[y:y+h, x:x+w]
    
    return cropped_depth

class OwlSAMProcessor:
    def __init__(self,
                 model_name="xl0",
                 weight_url=os.path.join(PLATO_COPILOT_ROOT_PATH, "../third_party/efficientvit/assets/checkpoints/sam/xl0.pt")):
        self.owl_detector = OwlDetector()
        self.sam_operator = EfficientSAMOperator(model_type=model_name, checkpoint=weight_url)
        self.sam_operator.init()

        self.last_overlay_img = None
        self.last_final_mask = None
        self.last_bbox = None

    def __call__(self, original_image, threshold=0.1, alpha=0.0):
        raw_image = resize_sam_image(original_image)
        texts = [["a photo of jenga tower"]]
        self.owl_detector.detect(Image.fromarray(raw_image), texts, threshold=threshold)
        cropped_images = self.owl_detector.get_cropped_image()

        # cv2.imwrite("cropped.jpg", cropped_images[0])

        bbox_list = self.owl_detector.boxes

        if len(bbox_list) == 0:
            return self.last_overlay_img, self.last_final_mask
        bbox = bbox_list[0].detach().cpu().numpy()
        self.last_bbox = bbox

        masks, _ = self.sam_operator.segment_image_from_bbox_prompt(raw_image, bbox)
        tower_mask = masks[0]
        cropped_tower_mask = Image.fromarray(tower_mask).crop(bbox)

        raw_cropped_img = resize_sam_image(cropped_images[0])
        complete_mask = self.sam_operator.segment_image(raw_cropped_img)

        merged_mask = complete_mask["merged_mask"] * cropped_tower_mask
        merged_mask = cv2.resize(merged_mask, (cropped_images[0].shape[1], cropped_images[0].shape[0]), cv2.INTER_NEAREST)

        bbox_int = np.round(bbox, 0).astype(int)
        canvas = np.zeros((raw_image.shape[0], raw_image.shape[1]), dtype=np.uint8)
        canvas[bbox_int[1]:bbox_int[3], bbox_int[0]:bbox_int[2]] = merged_mask

        canvas = cv2.resize(canvas, (original_image.shape[1], original_image.shape[0]), cv2.INTER_NEAREST)
        self.last_overlay_img = overlay_xmem_mask_on_image(original_image, canvas, rgb_alpha=alpha)

        self.last_final_mask = canvas  # This assumes final_mask is what you intend to use

        return self.last_overlay_img, self.last_final_mask

    def use_owl(self, original_image):
        raw_image = resize_sam_image(original_image)
        texts = [["a photo of jenga tower"]]
        self.owl_detector.detect(Image.fromarray(raw_image), texts)
        bbox_list = self.owl_detector.boxes

        if len(bbox_list) == 0:
            return self.last_overlay_img, self.last_final_mask
        bbox = bbox_list[0].detach().cpu().numpy()
        return bbox


    def get_cropped(self, original_image, original_depth=None):
        raw_image = resize_sam_image(original_image)
        if original_depth is not None:
            raw_depth = resize_sam_depth(original_depth)

        texts = [["a photo of jenga tower"]]
        self.owl_detector.detect(Image.fromarray(raw_image), texts)

        # cropped_images = self.owl_detector.get_cropped_image()

        boxes = self.owl_detector.boxes
        cropped_images = []
        cropped_depths = []
        for box in boxes:
            box = [round(i) for i in box.tolist()]
            cropped_image = Image.fromarray(raw_image).crop((box[0], box[1], box[2], box[3]))
            cropped_images.append(np.array(cropped_image))
            if original_depth is not None:
                cropped_depth = crop_depth(raw_depth, box)
                cropped_depths.append(np.array(cropped_depth))

        if original_depth is not None:
            return cropped_images, cropped_depths
        else:
            return cropped_images

