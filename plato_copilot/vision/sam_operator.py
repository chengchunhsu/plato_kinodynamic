"""This wrapper for segment-anything adopted from GROOT Zhu et al. 2023"""

import cv2  # type: ignore

import argparse
import json
import os
import numpy as np
import yaml
import pprint
import torch

from functools import partial
from PIL import Image

from typing import Any, Dict, List
from easydict import EasyDict
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

from efficientvit.models.efficientvit.sam import EfficientViTSamAutomaticMaskGenerator, EfficientViTSamPredictor
from efficientvit.models.utils import build_kwargs_from_config
from efficientvit.sam_model_zoo import create_sam_model
from efficientvit.apps.utils import parse_unknown_args

from plato_copilot import PLATO_COPILOT_ROOT_PATH
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

class SAMOperator():
    def __init__(self,
                 model_type="vit_b",
                 checkpoint=os.path.join(PLATO_COPILOT_ROOT_PATH, "../" "third_party/sam_checkpoints/sam_vit_b_01ec64.pth"),
                 sam_config_file=os.path.join(PLATO_COPILOT_ROOT_PATH, "vision_model_configs", "sam_config.yaml"),
                 device="cuda:0",
                 output_mode="binary_mask",
                 half_mode=True) -> None:
        with open(sam_config_file, 'r') as stream:
            self.config = EasyDict(yaml.safe_load(stream))

        self.model_type = model_type
        self.checkpoint = checkpoint
        self.device = device
        self.sam = None
        self.output_mode = output_mode

        self.half_mode = half_mode

        self.autocast_dtype = torch.float32
        if self.half_mode:
            self.autocast_dtype = torch.half
        self.autocast_ctx = partial(torch.cuda.amp.autocast, enabled=True, dtype=self.autocast_dtype)


    def init(self):
        self.sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint)
        self.sam.to(device=self.device)

        self.generator = SamAutomaticMaskGenerator(self.sam, output_mode=self.output_mode, **self.config)

    def print_config(self):
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(self.config)


    def write_masks_to_folder(self, masks: List[Dict[str, Any]], path: str) -> None:
        header = "id,area,bbox_x0,bbox_y0,bbox_w,bbox_h,point_input_x,point_input_y,predicted_iou,stability_score,crop_box_x0,crop_box_y0,crop_box_w,crop_box_h"  # noqa
        metadata = [header]
        for i, mask_data in enumerate(masks):
            mask = mask_data["segmentation"]
            filename = f"{i}.png"
            cv2.imwrite(os.path.join(path, filename), mask * 255)
            mask_metadata = [
                str(i),
                str(mask_data["area"]),
                *[str(x) for x in mask_data["bbox"]],
                *[str(x) for x in mask_data["point_coords"][0]],
                str(mask_data["predicted_iou"]),
                str(mask_data["stability_score"]),
                *[str(x) for x in mask_data["crop_box"]],
            ]
            row = ",".join(mask_metadata)
            metadata.append(row)
        metadata_path = os.path.join(path, "metadata.csv")
        with open(metadata_path, "w") as f:
            f.write("\n".join(metadata))

        return

    def show_anns(self, anns):
        if len(anns) == 0:
            return
        sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
        img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
        img[:,:,3] = 0
        for ann in sorted_anns:
            m = ann['segmentation']
            color_mask = np.concatenate([np.random.random(3), [0.35]])
            img[m] = color_mask
        return img

    def merge_segmentation_masks(self, anns):
        sorted_anns = sorted(anns, key=(lambda x: x["area"]), reverse=True)
        arrays = [ann["segmentation"] for ann in sorted_anns]
        merged_mask = np.zeros_like(arrays[0]).astype(np.uint8)
        instance_id = 1

        for array in arrays:
            merged_mask[np.where(array)] = instance_id
            instance_id += 1

        return merged_mask


    def save_merged_mask(self, merged_mask, filepath, verbose=True):
        # Convert the merged_mask NumPy array to a PIL Image
        mask_image = Image.fromarray(merged_mask.astype(np.uint8))
        # Save the image to the specified filepath
        mask_image.save(filepath)
        if verbose:
            print(f"Saved merged mask to {filepath}.")

    def save_overall_vis_mask(self, overall_mask, filepath, verbose=True):
        cv2.imwrite(filepath, (overall_mask * 255).astype(np.uint8))
        if verbose:
            print(f"Saved overall mask to {filepath}.")

    def get_individual_masks_from_raw(self, raw_data):
        masks = []
        for i, mask_data in enumerate(raw_data):
            mask = mask_data["segmentation"]
            masks.append(mask * 255)
        return masks

    def segment_image(self,
                      image,
                      merge_masks=True,
                      overall_vis_mask=True,
                      ):
        with self.autocast_ctx():
            print(self.generator, self.generator.generate)
            masks = self.generator.generate(image)
            print(f"Generated {len(masks)} masks")
            if merge_masks:
                merged_mask = self.merge_segmentation_masks(masks)

            if overall_vis_mask:
                overall_mask = self.show_anns(masks)

            return {
                "raw_data": masks,
                "merged_mask": merged_mask,
                "overall_mask": overall_mask,
            }

    def segment_images_from_a_folder(self,
                                     input_folder,
                                     output_folder,
                                     individual_masks=True,
                                     merge_masks=True,
                                     overall_vis_mask=True,
                                     save_masks=True,
                                     verbose=True):
        """sequentially segment images from a folder

        Args:
            input_folder (str): make sure the input folder path is configured correctly so that it is relative to the place you are running the script.
            output_folder (str): ake sure the output folder path is configured correctly so that it is relative to the place you are running the script.
            merge_masks (bool, optional): _description_. Defaults to True.
            overall_vis_mask (bool, optional): _description_. Defaults to True.
            save_masks (bool, optional): _description_. Defaults to True.
            verbose (bool, optional): _description_. Defaults to True.
        """
        targets = [os.path.join(input_folder, f) for f in os.listdir(input_folder)]

        mask_results = []
        for t in targets:
            image_name = t.split("/")[-1].split(".")[0]
            print(f"Processing '{t}'...")
            image = cv2.imread(t)
            if image is None:
                print(f"Could not load '{t}' as an image, skipping...")
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            mask_result_dict = self.segment_image(image,
                                                  merge_masks=merge_masks, overall_vis_mask=overall_vis_mask)

            if merge_masks and save_masks:
                self.save_overall_vis_mask(mask_result_dict["overall_mask"], os.path.join(output_folder, f"{image_name}_vis.png"), verbose=verbose)
            if overall_vis_mask and save_masks:
                self.save_merged_mask(mask_result_dict["merged_mask"], os.path.join(output_folder, image_name), verbose=verbose)
            if individual_masks and save_masks:
                base = os.path.basename(t)
                base = os.path.splitext(base)[0]
                save_base = os.path.join(output_folder, base)
                self.write_masks_to_folder(mask_result_dict["masks"], save_base)
            mask_results.append(mask_result_dict)
        if verbose:
            print("SAM operator Done")
        return mask_results



def draw_binary_mask(raw_image: np.ndarray, binary_mask: np.ndarray, mask_color=(0, 0, 255)) -> np.ndarray:
    color_mask = np.zeros_like(raw_image, dtype=np.uint8)
    color_mask[binary_mask == 1] = mask_color
    mix = color_mask * 0.5 + raw_image * (1 - 0.5)
    binary_mask = np.expand_dims(binary_mask, axis=2)
    canvas = binary_mask * mix + (1 - binary_mask) * raw_image
    canvas = np.asarray(canvas, dtype=np.uint8)
    return canvas

def load_image(data_path: str, mode="rgb") -> np.ndarray:
    img = Image.open(data_path)
    if mode == "rgb":
        img = img.convert("RGB")
    return np.array(img)


def draw_bbox(
    image: np.ndarray,
    bbox: list[list[int]],
    color: str or list[str] = "g",
    linewidth=1,
    tmp_name=".tmp.png",
) -> np.ndarray:
    dpi = 300
    oh, ow, _ = image.shape
    plt.close()
    plt.figure(1, figsize=(oh / dpi, ow / dpi))
    plt.imshow(image)
    if isinstance(color, str):
        color = [color for _ in bbox]
    for (x0, y0, x1, y1), c in zip(bbox, color):
        plt.gca().add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, lw=linewidth, edgecolor=c, facecolor=(0, 0, 0, 0)))
    plt.axis("off")
    plt.savefig(tmp_name, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.0)
    image = cv2.resize(load_image(tmp_name), dsize=(ow, oh))
    os.remove(tmp_name)
    plt.close()
    return image


def draw_scatter(
    image: np.ndarray,
    points: list[list[int]],
    color: str or list[str] = "g",
    marker="*",
    s=10,
    ew=0.25,
    tmp_name=".tmp.png",
) -> np.ndarray:
    dpi = 300
    oh, ow, _ = image.shape
    plt.close()
    plt.figure(1, figsize=(oh / dpi, ow / dpi))
    plt.imshow(image)
    if isinstance(color, str):
        color = [color for _ in points]
    for (x, y), c in zip(points, color):
        plt.scatter(x, y, color=c, marker=marker, s=s, edgecolors="white", linewidths=ew)
    plt.axis("off")
    plt.savefig(tmp_name, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.0)
    image = cv2.resize(load_image(tmp_name), dsize=(ow, oh))
    os.remove(tmp_name)
    plt.close()
    return image



class EfficientSAMOperator(SAMOperator):
    def __init__(
            self,
            model_type,
            checkpoint,
            sam_config_file=os.path.join(PLATO_COPILOT_ROOT_PATH, "vision_model_configs", "sam_config.yaml"),
            device="cuda:0", 
            output_mode="binary_mask",
            half_mode=True
    ):
        super().__init__(
            model_type,
            checkpoint,
            sam_config_file=sam_config_file,
            device=device,
            output_mode=output_mode,
            half_mode=half_mode
        )

    def init(self):

        efficientvit_sam = create_sam_model(self.model_type, 
                                            True, 
                                            self.checkpoint).cuda().eval()        
        self.predictor = EfficientViTSamPredictor(efficientvit_sam)
        self.generator = EfficientViTSamAutomaticMaskGenerator(
            efficientvit_sam,
            points_per_side=16,
            pred_iou_thresh=0.8,
            stability_score_thresh=0.85,
            min_mask_region_area=0,
        )

    def get_mask_results(self,
                         masks,
                         merge_masks=True,
                         overall_vis_mask=True):
        merged_mask = None
        overall_mask = None

        if merge_masks:
            merged_mask = self.merge_segmentation_masks(masks)

        if overall_vis_mask:
            overall_mask = self.show_anns(masks)

        return {
            "raw_data": masks,
            "merged_mask": merged_mask,
            "overall_mask": overall_mask,
        }

    def segment_image(self, 
                      image,
                      merge_masks=True,
                      overall_vis_mask=True,
                      ):
        print(image.shape)
        with self.autocast_ctx():
            print(self.generator)
            masks = self.generator.generate(image)
            print(f"Generated {len(masks)} masks")
            # merged_mask = None
            # overall_mask = None

            return self.get_mask_results(
                masks,
                merge_masks=merge_masks,
                overall_vis_mask=overall_vis_mask
            )
            # if merge_masks:
            #     merged_mask = self.merge_segmentation_masks(masks)

            # if overall_vis_mask:
            #     overall_mask = self.show_anns(masks)

            # return {
            #     "raw_data": masks,
            #     "merged_mask": merged_mask,
            #     "overall_mask": overall_mask,
            # }
        
    def segment_image_from_point_prompt(self, 
                                        image,
                                        points,
                                        multimask=False,
                                        merge_masks=True,
                                        overall_vis_mask=True):
        point_coords = [(x, y) for x, y, _ in points]
        point_labels = [l for _, _, l in points]        
        with self.autocast_ctx():
            self.predictor.set_image(image)
            masks, _, _ = self.predictor.predict(
                point_coords=np.array(point_coords),
                point_labels=np.array(point_labels),
                multimask_output=multimask,
            )
        plots = [
            draw_scatter(
                draw_binary_mask(image, binary_mask, (0, 0, 255)),
                point_coords,
                color=["g" if l == 1 else "r" for l in point_labels],
                s=10,
                ew=0.25,
                tmp_name="tmp.png",
            )
            for binary_mask in masks
        ]
        # mask_results = self.get_mask_results(
        #                     masks,
        #                     merge_masks=merge_masks,
        #                     overall_vis_mask=overall_vis_mask
        #                 )
        plots = self.cat_images(plots, axis=1)
        return masks, plots
    
    def segment_image_from_bbox_prompt(self, 
                                        image,
                                        bbox,
                                        multimask=False,
                                        merge_masks=True,
                                        overall_vis_mask=True):
     
        with self.autocast_ctx():
            self.predictor.set_image(image)
            masks, _, _ = self.predictor.predict(
                point_coords=None,
                point_labels=None,
                box=np.array(bbox),
                multimask_output=multimask,
            )

        plots = [
            draw_bbox(
                draw_binary_mask(image, binary_mask, (0, 0, 255)),
                [bbox],
                color="g",
                tmp_name="tmp.png",
            )
            for binary_mask in masks
        ]
        # mask_results = self.get_mask_results(
        #                     masks,
        #                     merge_masks=merge_masks,
        #                     overall_vis_mask=overall_vis_mask
        #                 )
        plots = self.cat_images(plots, axis=1)
        return masks, plots

    def cat_images(self, image_list: list[np.ndarray], axis=1, pad=20) -> np.ndarray:
        shape_list = [image.shape for image in image_list]
        max_h = max([shape[0] for shape in shape_list]) + pad * 2
        max_w = max([shape[1] for shape in shape_list]) + pad * 2

        for i, image in enumerate(image_list):
            canvas = np.zeros((max_h, max_w, 3), dtype=np.uint8)
            h, w, _ = image.shape
            crop_y = (max_h - h) // 2
            crop_x = (max_w - w) // 2
            canvas[crop_y : crop_y + h, crop_x : crop_x + w] = image
            image_list[i] = canvas

        image = np.concatenate(image_list, axis=axis)
        return image


