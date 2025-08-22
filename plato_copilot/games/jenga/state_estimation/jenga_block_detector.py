import copy
import math
import random
from collections import Counter

import cv2
import numpy as np
import torch
from PIL import Image
import sys
import argparse

from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D
import time
from functools import partial
import sys

from scipy.spatial import ConvexHull, convex_hull_plot_2d

import matplotlib
from segment_anything import sam_model_registry, SamPredictor
from sam2.modeling import sam2_base
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from easydict import EasyDict
from PIL import Image
from matplotlib import pyplot as plt
from plato_copilot.vision.owl_sam_processor import OwlSAMProcessor
from sklearn.cluster import KMeans

from plato_copilot.vision.o3d_utils import *
from plato_copilot.vision.plotly_utils import *
from plato_copilot.vision.img_utils import *
from plato_copilot.vision.point_cloud_utils import *
from plato_copilot.games.jenga.jenga_tower_state import *

from plato_copilot.utils.vis_utils import overlay_segmentation, save_block_masks_post
from plato_copilot.utils.log_utils import get_copilot_logger
logger = get_copilot_logger()

class JengaBlockDetector:
    def __init__(self, camera_matrix, sam_config, num_runs=100):
        self.camera_matrix = camera_matrix
        self.num_runs = num_runs
        self.num_layers = -1

        # set up SAM
        sam_checkpoint = "../third_party/sam_checkpoints/sam_vit_b_01ec64.pth"
        model_type = "vit_b"

        device = "cuda"

        sam = sam_model_registry[sam_config.model_type](checkpoint=sam_config.sam_checkpoint)
        sam.to(device=sam_config.device)

        # print(sam2_base.__file__)
        checkpoint = "../third_party/sam_checkpoints/sam2.1_hiera_large.pt"
        model_cfg = "sam2.1_hiera_l.yaml"
        # self.predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))
        self.predictor = SamPredictor(sam)
        logger.warn("currently only support initialization of jenga blocks.")

    def orient_image_pcd(self, rgb_image, depth_image, camera_matrix):
        pcd = scene_pcd_only_fn(rgb_image, np.array(depth_image[:, :, 0], dtype=np.float32), camera_matrix, np.eye(4).astype(np.float32))

        normal_pcd = pcd_normal_vectors(pcd.pcd)
        normal_pcd.paint_uniform_color([0, 0, 0])

        normal_pcd.orient_normals_towards_camera_location(camera_location=np.array([0, 0, 0], dtype=np.float32))

        pcd = O3DPointCloud()
        pcd.create_from_pcd(normal_pcd)

        # plotly_draw_3d_pcd(np.asarray(normal_pcd.points))

        # get the estimated plane model

        plane_model = pcd.plane_estimation(verbose=False)["plane_model"]
        transformation_matrix = estimate_rotation(plane_model, z_up=False)

        temp_pcd = copy.deepcopy(normal_pcd)
        temp_pcd.transform(transformation_matrix)

        # estimate plane again
        pcd = O3DPointCloud()
        pcd.create_from_pcd(temp_pcd)
        plane_model = pcd.plane_estimation(verbose=False)["plane_model"]

        translate_z = plane_model[3] / plane_model[2]
        transformation_matrix[2, 3] = translate_z

        # translate the point cloud to the plane
        temp_pcd = copy.deepcopy(normal_pcd)
        temp_pcd.transform(transformation_matrix)

        final_pcd = pcd_normal_vectors(temp_pcd)

        vis_info = {"normal_pcd": np.asarray(normal_pcd.points)}

        return final_pcd, transformation_matrix, vis_info


    def cluster_surface_normals(self, final_pcd, overlay_masks, depth_image, camera_matrix, transformation_matrix):
        transformed_kdtree = get_kdtree_flann(final_pcd)

        normal_vectors = set()
        for y in range(len(overlay_masks)):
            for x in range(len(overlay_masks[0])):
                if overlay_masks[y][x] == 0:
                    continue

                points = image_to_world_coords([[x, y]], depth_image, camera_matrix, transformation_matrix)[0]

                normal_vector, idx = find_closest_point(final_pcd, transformed_kdtree, points)
                normal_vectors.add(
                    tuple([round(normal_vector[0], 3), round(normal_vector[1], 3), round(normal_vector[2], 3), idx]))

        normal_vectors_with_idx = [list(normal_vector) for normal_vector in normal_vectors]
        normal_vectors = [normal_vector[:3] for normal_vector in normal_vectors]
        normal_idx = [normal_vector[3] for normal_vector in normal_vectors_with_idx]

        vector_labels, centroids, centroid_labels = kmeans_3d(normal_vectors, n_clusters=3)

        kmeans_points = ([v[0] for v in normal_vectors], [v[1] for v in normal_vectors],
                                [v[2] for v in normal_vectors])
        # plot_3d_kmeans_clusters([v[0] for v in normal_vectors], [v[1] for v in normal_vectors],
        #                         [v[2] for v in normal_vectors], num_clusters=3)

        max_z_centroid = np.argmax(np.abs(centroids[:, 2]))
        max_z_label = centroid_labels[max_z_centroid]

        non_z_centroids = np.delete(centroids, max_z_centroid, axis=0)
        angle_between_non_z_centroids = angle_between_vecs(non_z_centroids[0], non_z_centroids[1])

        label_sz = {}
        for label in vector_labels:
            if label == max_z_label:
                continue

            label_sz[label] = label_sz.get(label, 0) + 1

        primary_face = max(label_sz, key=label_sz.get)
        non_primary_face = min(label_sz, key=label_sz.get)
        print("Non primary face count: ", non_primary_face)

        primary_face_indices = [i for i, label in enumerate(vector_labels) if label == primary_face]
        non_primary_face_indices = [i for i, label in enumerate(vector_labels) if label == non_primary_face]

        primary_face_points = final_pcd.select_by_index([normal_idx[i] for i in primary_face_indices])
        primary_face_pcd = O3DPointCloud()
        primary_face_pcd.create_from_pcd(primary_face_points)

        non_primary_face_points = final_pcd.select_by_index([normal_idx[i] for i in non_primary_face_indices])
        non_primary_face_pcd = O3DPointCloud()
        non_primary_face_pcd.create_from_pcd(non_primary_face_points)

        # plotly_draw_3d_pcd(np.asarray(primary_face_points.points))
        vis_info = {
            "kmeans_points": kmeans_points,
            "primary_face_points": np.asarray(primary_face_points.points)}


        return primary_face_pcd, vis_info, None if angle_between_non_z_centroids < 60. else non_primary_face_pcd


    def get_plane_of_primary_face(self, primary_face_pcd, final_pcd):
        if primary_face_pcd is None:
            return None, None, None, None
        primary_face_plane = primary_face_pcd.plane_estimation(verbose=False)["plane_model"]

        primary_face_keep_idx = []

        for i, pt in enumerate(primary_face_pcd.get_points()):
            if abs(evaluate_plane(primary_face_plane, pt)) < 0.01:
                primary_face_keep_idx.append(i)

        primary_face_points = primary_face_pcd.pcd.select_by_index(primary_face_keep_idx)
        primary_face_x = []
        for pt in primary_face_points.points:
            primary_face_x.append(pt[0])
        primary_face_leftmost_x = np.percentile(np.asarray(primary_face_x), 1)
        primary_face_rightmost_x = np.percentile(np.asarray(primary_face_x), 99)

        primary_face_keep_idx = []
        for i, pt in enumerate(final_pcd.points):  # only keep points on the plane
            if pt[0] < primary_face_leftmost_x or pt[0] > primary_face_rightmost_x:
                continue
            if abs(evaluate_plane(primary_face_plane, pt)) < 0.01:  # TODO: Change this threshold to be dynamic
                primary_face_keep_idx.append(i)

        primary_face_points = final_pcd.select_by_index(primary_face_keep_idx)
        primary_face_pcd = O3DPointCloud()
        primary_face_pcd.create_from_pcd(primary_face_points)

        # plotly_draw_3d_pcd(np.asarray(primary_face_points.points))
        vis_info = {"primary_face_points": np.asarray(primary_face_points.points)}

        return primary_face_pcd, primary_face_plane, primary_face_points, vis_info


    def convex_hull(self, rgb_image, primary_face_points, camera_matrix, transformation_matrix):
        primary_face_np_array = np.asarray(primary_face_points.points)
        primary_face_image_coords = world_to_image_coords(primary_face_np_array, camera_matrix, transformation_matrix)

        # remove last dimension
        primary_face_image_coords = primary_face_image_coords[:, :2]

        # turn into integers
        primary_face_image_coords = np.round(primary_face_image_coords).astype(int)

        # construct the convex hull of these points
        hull = ConvexHull(primary_face_image_coords)
        hull_vertices = []

        # print("Hull: ", hull.vertices)

        # draw lines connecting each hull on the rgb image
        for i in range(len(hull.vertices)):
            hull_vertices.append(primary_face_image_coords[hull.vertices[i]])
            start = hull.vertices[i]
            end = hull.vertices[(i + 1) % len(hull.vertices)]
            cv2.line(rgb_image, (int(primary_face_image_coords[start][0]), int(primary_face_image_coords[start][1])),
                     (int(primary_face_image_coords[end][0]), int(primary_face_image_coords[end][1])), (255, 0, 0), 2)

        # print("Hull Vertices: ", hull_vertices)

        # get leftmost and rightmost image point on hull

        leftmost_x = math.inf
        rightmost_x = -math.inf
        for pt in hull_vertices:
            leftmost_x = min(leftmost_x, pt[0])
            rightmost_x = max(rightmost_x, pt[0])

        # print("Leftmost X: ", leftmost_x)

        # get height of tower

        leftmost_y_max = -math.inf
        leftmost_y_min = math.inf

        rightmost_y_max = -math.inf
        rightmost_y_min = math.inf
        for pt in hull_vertices:
            if leftmost_x - 30 < pt[0] < leftmost_x + 30:
                leftmost_y_max = max(leftmost_y_max, pt[1])
                leftmost_y_min = min(leftmost_y_min, pt[1])

            if rightmost_x - 30 < pt[0] < rightmost_x + 30:
                rightmost_y_max = max(rightmost_y_max, pt[1])
                rightmost_y_min = min(rightmost_y_min, pt[1])

        # print(leftmost_y_min, leftmost_y_max, rightmost_y_min, rightmost_y_max)

        bbox = [leftmost_x, min(leftmost_y_min, rightmost_y_min), rightmost_x, max(leftmost_y_max, rightmost_y_max)]
        cropped_primary_face = Image.fromarray(rgb_image).crop((bbox[0], bbox[1], bbox[2], bbox[3]))

        return rgb_image, cropped_primary_face, bbox


    def construct_tower_basis(self, primary_face_pcd, primary_face_points, primary_face_plane):
        if primary_face_pcd is None:
            return None, None, None, None, None, None
        # go through z values, and get length of tower
        face_min_z = min(primary_face_pcd.get_points()[:, 2])
        face_max_z = max(primary_face_pcd.get_points()[:, 2])
        face_min_x = min(primary_face_pcd.get_points()[:, 0])
        face_max_x = max(primary_face_pcd.get_points()[:, 0])
        steps = 50.

        margin_z = 0.005
        margin_x = 0.005

        tower_length = 0

        face_left_point = None
        face_right_point = None

        for z in np.linspace(face_min_z, face_max_z, int(steps)):
            min_x = math.inf
            max_x = -math.inf
            left_pt = None
            right_pt = None

            for pt in primary_face_points.points:
                if abs(pt[2] - z) > margin_z:
                    continue

                if pt[0] < min_x:
                    min_x = pt[0]
                    left_pt = pt

                if pt[0] > max_x:
                    max_x = pt[0]
                    right_pt = pt

            if max_x != -math.inf and min_x != math.inf:
                if np.linalg.norm(left_pt - right_pt) > tower_length:
                    face_left_point = left_pt
                    face_right_point = right_pt
                    # print("Left pt: ", left_pt, "Right pt: ", right_pt)
                    tower_length = np.linalg.norm(left_pt - right_pt)

        # print("Tower Length: ", tower_length)
        logger.debug(f"Tower Length: {tower_length}")

        three_d_tower_height = 0

        for x in np.linspace(face_min_x, face_max_x, int(steps)):
            min_z = math.inf
            max_z = -math.inf
            bottom_pt = None
            top_pt = None

            for pt in primary_face_points.points:
                if abs(pt[0] - x) > margin_x:
                    continue

                if pt[2] < min_z:
                    min_z = pt[2]
                    bottom_pt = pt

                if pt[2] > max_z:
                    max_z = pt[2]
                    top_pt = pt

            if max_z != -math.inf and min_z != math.inf:
                if np.linalg.norm(top_pt - bottom_pt) > three_d_tower_height:
                    three_d_tower_height = np.linalg.norm(top_pt - bottom_pt)

        logger.debug(f"3D tower height: {three_d_tower_height}")

        # Jenga Tower Bounding Box
        tower_top_left = [face_left_point[0], face_left_point[1], face_max_z]

        primary_face_out_of_plane = np.array([primary_face_plane[0], primary_face_plane[1], primary_face_plane[2]])
        primary_face_down = project_onto_plane(primary_face_plane, np.array([0, 0, -1], dtype=np.float32))
        primary_face_along_plane = np.cross(primary_face_out_of_plane, primary_face_down)

        # normalize
        primary_face_out_of_plane /= np.linalg.norm(primary_face_out_of_plane)
        primary_face_down /= np.linalg.norm(primary_face_down)
        primary_face_along_plane /= np.linalg.norm(primary_face_along_plane)

        # flip the direction of primary face along plane to be in the direction of left to right if necessary
        left_to_right = np.array(face_right_point) - np.array(face_left_point)
        if np.dot(primary_face_along_plane, left_to_right) < 0:
            primary_face_along_plane = -primary_face_along_plane

        # print(primary_face_plane)
        logger.debug(f"Primary Out of Plane: {primary_face_out_of_plane}")
        logger.debug(f"Primary Face Along Plane: {primary_face_along_plane}")
        logger.debug(f"Primary Face Down: {primary_face_down}")

        return tower_length, three_d_tower_height, tower_top_left, primary_face_along_plane, primary_face_down, primary_face_out_of_plane

    def determine_layer_count(self, rgb_image, bbox, transformation_matrix, cropped_primary_face,
                              primary_face_basis, tower_info):
        # assume there is a maximum of 20 layers
        three_d_tower_height = tower_info["three_d_tower_height"]
        tower_middle_column_top = tower_info["tower_middle_column_top"]
        tower_length = tower_info["tower_length"]

        primary_face_down = primary_face_basis["primary_face_down"]
        primary_face_along_plane = primary_face_basis["primary_face_along_plane"]
        primary_face_out_of_plane = primary_face_basis["primary_face_out_of_plane"]

        accept_threshold = 0.25 # accepts block if num_runs * accept_threshold votes

        num_layer_estimate = (0, 0) # (number of layers, estimated block height)
        for num_layer in range(1, 16):
            layer_orientation_horizontal = 0
            layer_orientation_vertical = 0

            num_runs = 150
            # 0 is horizontal at the top, 1 is vertical at the top
            layer_height_candidates = []

            for run in range(num_runs):

                total_point_prompts = []
                total_point_pos_neg = []

                # block_height = 0.024270 # three_d_tower_height / num_layer + (random.random() * 0.005) - 0.0025
                block_height = three_d_tower_height / num_layer + (random.random() * 0.01) - 0.005
                # print("Block Height: ", block_height)

                layer_areas = []

                images = []
                for layer in range(int(num_layer)):
                    layer_z_vector = primary_face_down * block_height * (layer + 0.5)
                    layer_start_point = tower_middle_column_top + layer_z_vector

                    # generate points
                    point_prompts, point_pos_neg = generate_point_prompt(layer_start_point,
                                                                         primary_face_along_plane * (tower_length / 3.),
                                                                         primary_face_down * block_height,
                                                                         top_pt=(layer == 0),
                                                                         bot_pt=(layer == num_layer - 1))

                    # convert to image coordinates
                    point_prompts_image = world_to_image_coords(point_prompts, self.camera_matrix,
                                                                transformation_matrix)
                    point_prompts_image = point_prompts_image[:, :2]

                    total_point_prompts.extend(point_prompts_image)
                    total_point_pos_neg.extend(point_pos_neg)

                    # round point prompts image coordinates to integers
                    point_prompts_owl = np.round(point_prompts_image).astype(int)

                    for i in range(len(point_prompts_owl)):
                        point_prompts_owl[i][0] -= bbox[0]
                        point_prompts_owl[i][1] -= bbox[1]

                    good_points = True
                    for i in range(len(point_prompts_owl)):
                        if (point_prompts_owl[i][0] < 0 or point_prompts_owl[i][0] >=
                                np.asarray(cropped_primary_face).shape[1] or
                                point_prompts_owl[i][1] < 0 or point_prompts_owl[i][1] >=
                                np.asarray(cropped_primary_face).shape[0]):
                            layer_score = -1
                            good_points = False
                            break

                    if not good_points:
                        break

                    point_prompts_owl = np.array(point_prompts_owl)
                    point_pos_neg = np.array(point_pos_neg)

                    masks, scores, logits = self.predictor.predict(
                        point_coords=point_prompts_owl,
                        point_labels=point_pos_neg,
                        multimask_output=False
                    )

                    best_mask = masks[np.argmax(scores)]
                    layer_areas.append(np.sum(best_mask))
                    layer_score = np.max(scores)

                if layer_score < 0.4:
                    continue

                # compare even layer means and odd layer means
                even_layer_means = np.mean(layer_areas[::2])
                odd_layer_means = np.mean(layer_areas[1::2])

                # if means are too close, do not count it, must be at least 1-2x different
                # print("Layer Means: ", block_height, even_layer_means, odd_layer_means)
                # print(abs(even_layer_means - odd_layer_means), 0.85 * min(even_layer_means, odd_layer_means))
                if abs(even_layer_means - odd_layer_means) < 0.85 * min(even_layer_means, odd_layer_means):
                    # grid_image = create_grid_picture(images, 4, 3)
                    # cv2.imwrite(f"test_output_images/negative/{num_layer}_{run}_{block_height}.jpg", grid_image)
                    continue

                # grid_image = create_grid_picture(images, 4, 3)
                # cv2.imwrite(f"test_output_images/{num_layer}_{run}_{block_height}.jpg", grid_image)

                if even_layer_means > odd_layer_means:
                    layer_orientation_horizontal += 1
                else:
                    layer_orientation_vertical += 1

                layer_height_candidates.append(block_height)

            logger.debug(f"Layers: {num_layer} has {len(layer_height_candidates)} votes")
            if len(layer_height_candidates) >= num_runs * accept_threshold:
                num_layer_estimate = (num_layer, np.mean(layer_height_candidates))

        return num_layer_estimate

    def segment(self, rgb_image, depth_image):
        # initial segment
        owl_sam_processor = OwlSAMProcessor()
        overlay_img, overlay_masks = owl_sam_processor(rgb_image, threshold=0.07)


        # re-orient the rgb point cloud to have table on the x-y plane
        logger.info("Finding and orienting the jenga tower point cloud")
        final_pcd, transformation_matrix, vis_info = self.orient_image_pcd(rgb_image, depth_image, self.camera_matrix)

        # cluster the surface normals to get the primary faces of the surface
        logger.info("Clustering surface normals")
        primary_face_pcd, vis_info, non_primary_face_pcd = self.cluster_surface_normals(final_pcd, overlay_masks, depth_image, self.camera_matrix, transformation_matrix)

        # plotly_draw_3d_pcd(np.asarray(primary_face_pcd.pcd.points))

        # construct the full primary face plane
        logger.info("Constructing the primary face plane")
        primary_face_pcd, primary_face_plane, primary_face_points, vis_info = self.get_plane_of_primary_face(primary_face_pcd, final_pcd)

        # get the convex hull of the primary face and crop the image
        logger.info("Constructing the convex hull of the primary face points")
        rgb_image, cropped_primary_face, bbox = self.convex_hull(rgb_image, primary_face_points, self.camera_matrix, transformation_matrix)

        # get the tower dimensions and tower basis from the current point data
        logger.info("Estimating the tower dimensions")
        tower_length, three_d_tower_height, tower_top_left, primary_face_along_plane, primary_face_down, primary_face_out_of_plane = (
            self.construct_tower_basis(primary_face_pcd, primary_face_points, primary_face_plane))

        # plotly_draw_3d_pcd(np.asarray(primary_face_points.points))

        self.predictor.set_image(np.asarray(cropped_primary_face))


        tower_middle_column_top = tower_top_left + primary_face_along_plane * (tower_length / 3.) * 1.5

        layer_orientation_horizontal = 0
        layer_orientation_vertical = 0

        logger.debug("Estimating the number of layers")
        num_layer_estimate = self.determine_layer_count(rgb_image, bbox, transformation_matrix, cropped_primary_face,
                                        {
                                            "primary_face_along_plane": primary_face_along_plane,
                                            "primary_face_down": primary_face_down,
                                            "primary_face_out_of_plane": primary_face_out_of_plane
                                        },
                                        {
                                            "tower_length": tower_length,
                                            "three_d_tower_height": three_d_tower_height,
                                            "tower_middle_column_top": tower_middle_column_top
                                        })

        logger.info(f"Estimated Number of Layers: {num_layer_estimate[0]}, Estimated block height: {num_layer_estimate[1]}")
        self.num_layers = num_layer_estimate[0]
        num_runs = 100
        # 0 is horizontal at the top, 1 is vertical at the top
        layer_height_candidates = []
        grid_image = None

        # TODO: Record intermediate results from the runs.
        grid_images = []
        for run in range(num_runs):

            total_point_prompts = []
            total_point_pos_neg = []

            block_height = three_d_tower_height / self.num_layers + (random.random() * 0.01) - 0.005
            layer_areas = []
            images = []

            for layer in range(int(self.num_layers)):
                layer_z_vector = primary_face_down * block_height * (layer + 0.5)
                layer_start_point = tower_middle_column_top + layer_z_vector

                # generate points
                point_prompts, point_pos_neg = generate_point_prompt(layer_start_point,
                                                                     primary_face_along_plane * (tower_length / 3.),
                                                                     primary_face_down * block_height,
                                                                     top_pt=(layer == 0), bot_pt=(layer == self.num_layers - 1))

                # convert to image coordinates
                point_prompts_image = world_to_image_coords(point_prompts, self.camera_matrix, transformation_matrix)
                point_prompts_image = point_prompts_image[:, :2]

                total_point_prompts.extend(point_prompts_image)
                total_point_pos_neg.extend(point_pos_neg)

                # round point prompts image coordinates to integers
                point_prompts_owl = np.round(point_prompts_image).astype(int)

                for i in range(len(point_prompts_owl)):
                    point_prompts_owl[i][0] -= bbox[0]
                    point_prompts_owl[i][1] -= bbox[1]

                good_points = True
                for i in range(len(point_prompts_owl)):
                    if (point_prompts_owl[i][0] < 0 or point_prompts_owl[i][0] >= np.asarray(cropped_primary_face).shape[
                        1] or
                            point_prompts_owl[i][1] < 0 or point_prompts_owl[i][1] >=
                            np.asarray(cropped_primary_face).shape[0]):
                        layer_score = -1
                        good_points = False
                        break

                if not good_points:
                    break

                point_prompts_owl = np.array(point_prompts_owl)
                point_pos_neg = np.array(point_pos_neg)

                masks, scores, logits = self.predictor.predict(
                    point_coords=point_prompts_owl,
                    point_labels=point_pos_neg,
                    multimask_output=False
                )

                best_mask = masks[np.argmax(scores)]
                layer_areas.append(np.sum(best_mask))
                layer_score = np.max(scores)

                temp_image = copy.deepcopy(rgb_image)
                temp_image_2 = copy.deepcopy(rgb_image)

                # turn best_mask to grayscale
                temp_mask = np.zeros((best_mask.shape[0], best_mask.shape[1], 3), dtype=np.uint8)
                temp_mask[:, :, 0] = best_mask * 255

                # map onto image mask
                image_mask = np.zeros((rgb_image.shape[0], rgb_image.shape[1], 3), dtype=np.uint8)
                image_mask[bbox[1]:bbox[3], bbox[0]:bbox[2]] = temp_mask

                # draw the points on the image, positive are green, negative are red
                for i, point in enumerate(point_prompts_image):
                    if point_pos_neg[i] == 1:
                        cv2.circle(temp_image, (int(point[0]), int(point[1])), 5, (0, 255, 0), -1)
                    else:
                        cv2.circle(temp_image, (int(point[0]), int(point[1])), 5, (0, 0, 255), -1)

                # Blend the mask onto the original image
                result_image = cv2.addWeighted(temp_image, 1, image_mask, 0.5, 0)
                images.append(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))

                if layer == self.num_layers - 1:
                    for i, point in enumerate(total_point_prompts):
                        if total_point_pos_neg[i] == 1:
                            cv2.circle(temp_image_2, (int(point[0]), int(point[1])), 5, (0, 255, 0), -1)
                        else:
                            cv2.circle(temp_image_2, (int(point[0]), int(point[1])), 5, (0, 0, 255), -1)

            if layer_score < 0.4:
                continue

            # compare even layer means and odd layer means
            even_layer_means = np.mean(layer_areas[::2])
            odd_layer_means = np.mean(layer_areas[1::2])

            # if means are too close, do not count it, must be at least 1-2x different
            # print("Layer Means: ", even_layer_means, odd_layer_means)
            # print(abs(even_layer_means - odd_layer_means), 1.25 * min(even_layer_means, odd_layer_means))

            threshold = 0.85
            logger.debug(f"Layer Means: {even_layer_means}, {odd_layer_means}, difference: {abs(even_layer_means - odd_layer_means)}"
                         f", threshold: {threshold * min(even_layer_means, odd_layer_means)}")
            if abs(even_layer_means - odd_layer_means) < threshold * min(even_layer_means, odd_layer_means):
                continue

            # grid_image = create_grid_picture(images, 4, 3)
            # grid_images.append(grid_image)

            if even_layer_means > odd_layer_means:
                layer_orientation_horizontal += 1
            else:
                layer_orientation_vertical += 1

            layer_height_candidates.append(block_height)

        # for (ix, im) in enumerate(grid_images):
        #     cv2.imwrite(f"test_output_images/{ix}.jpg", im)

        logger.info(f"Block length: {tower_length}")
        logger.info(f"Probable Layer Heights: {np.mean(layer_height_candidates)}")
        logger.info(f"Horizontal Votes: {layer_orientation_horizontal}")
        logger.info(f"Vertical Votes: {layer_orientation_vertical}")
        if layer_orientation_horizontal > layer_orientation_vertical:
            logger.info("Horizontal Orientation At the Top")
        else:
            logger.info("Vertical Orientation At the Top")

        out_of_screen = np.array([0, -1, primary_face_out_of_plane[2]], dtype=np.float32)
        direction = np.dot(primary_face_out_of_plane, out_of_screen)
        if direction < 0:
            primary_face_out_of_plane = -primary_face_out_of_plane

        primary_face_angle = np.arctan(primary_face_out_of_plane[0] / primary_face_out_of_plane[1])
        primary_face_left = True  # true means left, false means right
        if primary_face_angle < 0:
            primary_face_left = False

        logger.info(f"Tower's primary face is on the {'left' if primary_face_left else 'right'} side")

        left_to_right = np.array([1, 0, 0])

        # calculate non_primary_face basis
        non_primary_face_out_of_plane = np.cross(primary_face_out_of_plane, np.array([0, 0, 1], dtype=np.float32))
        non_primary_face_along_plane = np.cross(primary_face_along_plane, np.array([0, 0, 1], dtype=np.float32))
        if np.dot(non_primary_face_along_plane, left_to_right) < 0:
            non_primary_face_along_plane = -non_primary_face_along_plane
        non_primary_face_down = np.cross(non_primary_face_out_of_plane, non_primary_face_along_plane)
        if np.dot(non_primary_face_down, primary_face_down) < 0:
            non_primary_face_down = -non_primary_face_down

        # normalize everything
        non_primary_face_out_of_plane /= np.linalg.norm(non_primary_face_out_of_plane)
        non_primary_face_along_plane /= np.linalg.norm(non_primary_face_along_plane)
        non_primary_face_down /= np.linalg.norm(non_primary_face_down)


        non_primary_face_top_left = tower_top_left + primary_face_along_plane * (tower_length / 3.) if primary_face_left \
            else tower_top_left - non_primary_face_along_plane * (tower_length / 3.)

        direction = np.dot(non_primary_face_out_of_plane, out_of_screen)
        if direction < 0:
            non_primary_face_out_of_plane = -non_primary_face_out_of_plane

        non_primary_face_left = not primary_face_left


        logger.info(f"Primary face basis: {(primary_face_along_plane, primary_face_down, primary_face_out_of_plane)}")
        logger.info(f"Non primary face basis: {(non_primary_face_along_plane, non_primary_face_down, non_primary_face_out_of_plane)}")
        logger.info(f"Primary face top left point: {tower_top_left}")
        logger.info(f"Non primary face top left point: {non_primary_face_top_left}")
        detection_result = {
            "block_length": tower_length,
            "num_layers": self.num_layers,
            "transformation_matrix": transformation_matrix,
            "probable_layer_height": np.mean(layer_height_candidates),
            "primary_face_top_left": tower_top_left,
            "non_primary_face_top_left": non_primary_face_top_left,
            "horizontal_votes": layer_orientation_horizontal,
            "vertical_votes": layer_orientation_vertical,
            "top_horizontal_to_camera": layer_orientation_horizontal > layer_orientation_vertical,
            "primary_face_basis": (primary_face_along_plane, primary_face_down, primary_face_out_of_plane),
            "non_primary_face_basis": (non_primary_face_along_plane, non_primary_face_down, non_primary_face_out_of_plane),
            "primary_face_side": primary_face_left,
            "non_primary_face_side": non_primary_face_left
        }

        return detection_result, grid_images

    def segment_all_faces(self, rgb_image, depth_image):
        detection_result, _ = self.segment(rgb_image, depth_image)

        num_layers = detection_result["num_layers"]
        block_height = detection_result["probable_layer_height"]
        block_width = detection_result["block_length"]
        primary_face_basis = detection_result["primary_face_basis"]
        non_primary_face_basis = detection_result["non_primary_face_basis"]
        primary_face_side = detection_result["primary_face_side"]
        top_layer_orientation = detection_result["top_horizontal_to_camera"]
        primary_face_top_left = detection_result["primary_face_top_left"]
        non_primary_face_top_left = detection_result["non_primary_face_top_left"]
        transformation_matrix = detection_result["transformation_matrix"]
        block_masks_pre = {}
        block_masks_post = {}

        # currently the along the plane vectors are from left to right, need to change to be from the midpoint
        # to the edge respective to the side of primary face
        horizontal_with = {
            (1, 1): 0, # top layer horizontal and primary face left
            (1, 0): 2, # top layer horizontal and primary face right
            (0, 1): 2, # top layer vertical and primary face left
            (0, 0): 0  # top layer vertical and primary face right
        }

        cnt = 0
        starts = [(0, block_width / 3., 0, block_height), (block_width / 3., 2./3 * block_width, 0, block_height), (2./3 * block_width, block_width, 0, block_height)]
        for layer in range(num_layers):
            # this is referring to the primary face
            layer_orientation = top_layer_orientation if layer % 2 == 0 else not top_layer_orientation
            start_point = primary_face_top_left + primary_face_basis[1] * layer * block_height
            start_point_non_primary = start_point + -non_primary_face_basis[0] * block_width
            if primary_face_side:
                start_point_non_primary = start_point + primary_face_basis[0] * block_width

            # print("Start point is: ", start_point, primary_face_basis[1] * layer * block_height)
            block_faces_mask = []
            horizontal_face_mask = None
            if layer_orientation:
                # horizontal
                horizontal_face_mask = generate_mask_points_basis(start_point, primary_face_basis, 0, block_width, 0, block_height, density=300)

                # get the non primary side's blocks
                for i in range(3):
                    block_faces_mask.append([])
                    block_faces_mask[i] = generate_mask_points_basis(start_point_non_primary, non_primary_face_basis, starts[i][0], starts[i][1], starts[i][2], starts[i][3])

            else:
                # vertical
                for i in range(3):
                    block_faces_mask.append([])
                    block_faces_mask[i] = generate_mask_points_basis(start_point, primary_face_basis, starts[i][0], starts[i][1], starts[i][2], starts[i][3])

                # get the non primary side's horizontal blocks
                horizontal_face_mask = generate_mask_points_basis(start_point_non_primary, non_primary_face_basis, 0, block_width, 0, block_height, density=300)


            # print(block_faces_mask[horizontal_with[(top_layer_orientation, primary_face_side)]])
            for horizontals in horizontal_face_mask:
                block_faces_mask[horizontal_with[(layer_orientation, primary_face_side)]].append(horizontals)

            # if cnt == 0:
            #     block1_pcd = create_o3d_from_points_and_color(block_faces_mask[0])
            #     block2_pcd = create_o3d_from_points_and_color(block_faces_mask[1])
            #     block3_pcd = create_o3d_from_points_and_color(block_faces_mask[2])
            #
            #     plotly_draw_3d_pcd(np.asarray(block1_pcd.points))
            #     plotly_draw_3d_pcd(np.asarray(block2_pcd.points))
            #     plotly_draw_3d_pcd(np.asarray(block3_pcd.points))
            #     cnt+=1
            block_masks_pre[layer] = block_faces_mask

        # print(block_masks_pre.keys())
        # turn the masks from world coords to camera coords
        for layer in block_masks_pre:
            block_masks_post[layer] = {}
            for i in range(3):
                block_masks_post[layer][i] = []
                image_coords = world_to_image_coords(np.array(block_masks_pre[layer][i]), self.camera_matrix, transformation_matrix)

                for image_coord in image_coords:
                    block_masks_post[layer][i].append(image_coord)

        # segmented_overlay_image = np.zeros((rgb_image.shape[0], rgb_image.shape[1], 3), dtype=np.uint8)

        # # print(block_masks_post.keys())
        # colors = 0
        # for layer in block_masks_post:
        #     for i in range(3):
        #         for coord in block_masks_post[layer][i]:
        #             if coord[0] < 0 or coord[0] >= segmented_overlay_image.shape[1] or coord[1] < 0 or coord[1] >= segmented_overlay_image.shape[0]:
        #                 continue

        #             # assign new color for every mask
        #             segmented_overlay_image[int(coord[1]), int(coord[0])] = id_to_rgb(colors)
        #         colors += 1

        # for y in range(len(segmented_overlay_image)):
        #     for x in range(len(segmented_overlay_image[0])):
        #         if np.all(segmented_overlay_image[y][x] == 0):
        #             segmented_overlay_image[y][x] = rgb_image[y][x]

        save_block_masks_post(block_masks_post, 'block_masks_post.pkl')

        segmented_overlay_image = overlay_segmentation(rgb_image, block_masks_post)
        return block_masks_post, segmented_overlay_image

    def get_jenga_tower(self, rgb_image, depth_image):
        detection_result, grid_images = self.segment(rgb_image, depth_image)

        """
        Orientation not needed right now, will be used later
        """

        tower = JengaTowerState(layers=self.num_layers, block_height=detection_result["probable_layer_height"] * 100, block_length=detection_result["block_length"] * 100,
                           block_width=detection_result["block_length"]/3. * 100)

        return tower
