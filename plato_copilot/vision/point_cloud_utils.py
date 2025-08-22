import copy
import math
import random
from collections import Counter, defaultdict

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

from easydict import EasyDict
from PIL import Image
from matplotlib import pyplot as plt
from plato_copilot.vision.plotly_utils import *
from plato_copilot.vision.owl_sam_processor import OwlSAMProcessor
from sklearn.cluster import KMeans


def kmeans_3d(points, n_clusters=3):
    # Initialize KMeans with 3 clusters
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(points)

    # Get the labels for each point
    labels = kmeans.labels_

    centroids = kmeans.cluster_centers_
    centroid_labels = kmeans.predict(centroids)

    return labels, centroids, centroid_labels

def plot_3d_kmeans_clusters(x_vals, y_vals, z_vals, num_clusters=2):
    # Combine x_vals, y_vals, and z_vals into a 2D feature array
    features = np.column_stack((x_vals, y_vals, z_vals))

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(features)
    cluster_labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    # Create a 3D scatter plot using matplotlib
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    cluster_points = []
    cluster_colors = []
    colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [0, 255, 255], [255, 0, 255]]
    for cluster_label in range(num_clusters):
        cluster_mask = (cluster_labels == cluster_label)
        for cluster_ix in range(len(cluster_mask)):
            if cluster_mask[cluster_ix]:
                cluster_points.append([x_vals[cluster_ix], y_vals[cluster_ix], z_vals[cluster_ix]])
                cluster_colors.append(colors[cluster_label])

    cluster_points = np.array(cluster_points)
    cluster_colors = np.array(cluster_colors)

    plotly_draw_3d_pcd(cluster_points, cluster_colors, marker_size=3, title="K-means Clustering")
def project_onto_plane(plane_model, vector):
    # Extract plane coefficients
    a, b, c, d = plane_model
    # Normal vector
    normal_vector = np.array([a, b, c])
    # Magnitude of the normal vector
    norm_squared = np.dot(normal_vector, normal_vector)

    # Dot product of the input vector with the normal vector
    dot_product = np.dot(vector, normal_vector)

    # Calculate the projection vector
    projection = vector - (dot_product / norm_squared) * normal_vector

    return projection

def angle_between_vecs(u, v):
    dot_product = np.dot(u, v)

    # Compute the norms
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)

    # Compute the cosine of the angle
    cos_theta = dot_product / (norm_u * norm_v)

    # Clip the cosine value
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    theta = np.arccos(cos_theta)
    theta_degrees = np.degrees(theta)
    return theta_degrees

def generate_point_prompt(start_point, along_vector, down_vector, top_pt=False, bot_pt=False):
    positive = [start_point]
    negative = [start_point + along_vector, start_point - along_vector, start_point + down_vector,
                start_point - down_vector]

    if top_pt:
        negative = negative[:len(negative) - 1]

    if bot_pt:
        negative = negative[:len(negative) - 2] + [negative[-1]]

    return np.array(positive + negative), np.array([1, 0, 0, 0]) if top_pt or bot_pt else np.array([1, 0, 0, 0, 0])


def generate_faces(vertices, num_points_u=10, num_points_v=10):
    """
    Generates the 6 faces of a rectangular prism and returns a dense grid of points for each face,
    regardless of the input vertex ordering.

    Parameters:
    - vertices: List of 8 vertices, each vertex is a tuple or list of (x, y, z).
    - num_points_u: Number of points along one edge of the face (controls density).
    - num_points_v: Number of points along the other edge of the face.

    Returns:
    - faces_points: A list containing 6 elements, each element is a numpy array of shape
      (num_points_u * num_points_v, 3), representing the grid points over each face.
    """
    # Ensure vertices are numpy arrays
    vertices = np.array(vertices)

    coord_points_x = defaultdict(list)
    coord_points_y = defaultdict(list)
    coord_points_z = defaultdict(list)

    for vertex in vertices:
        x, y, z = vertex
        # round to 4 decimal places to avoid floating point errors
        x = round(x, 4)
        y = round(y, 4)
        z = round(z, 4)

        coord_points_x[x].append(vertex)
        coord_points_y[y].append(vertex)
        coord_points_z[z].append(vertex)

    # assert here that each dictionary has 2 keys
    assert len(coord_points_x) == 2
    assert len(coord_points_y) == 2
    assert len(coord_points_z) == 2

    face_points_x = {}
    face_points_y = {}
    face_points_z = {}

    for coord_points in coord_points_x:
        for coord, points in coord_points.items():
            assert(len(points) == 4)
            # Found a face
            v0, v1, v2, v3 = points
            face_points_x[coord] = generate_face_points(v0, v1, v2, v3, num_points_u, num_points_v)

    for coord_points in coord_points_y:
        for coord, points in coord_points.items():
            assert (len(points) == 4)
            # Found a face
            v0, v1, v2, v3 = points
            face_points_y[coord] = generate_face_points(v0, v1, v2, v3, num_points_u, num_points_v)

    for coord_points in coord_points_z:
        for coord, points in coord_points.items():
            assert (len(points) == 4)
            # Found a face
            v0, v1, v2, v3 = points
            face_points_z[coord] = generate_face_points(v0, v1, v2, v3, num_points_u, num_points_v)

    return face_points_x, face_points_y, face_points_z


def generate_face_points(v0, v1, v2, v3, num_points_u, num_points_v):
    """
    Generates a grid of points over a quadrilateral face defined by vertices v0, v1, v2, v3.

    Parameters:
    - v0, v1, v2, v3: Corner vertices of the face.
    - num_points_u, num_points_v: Number of points along two edges.

    Returns:
    - points: Numpy array of shape (num_points_u * num_points_v, 3)
    """
    # Create parameter grids
    u = np.linspace(0, 1, num_points_u)
    v = np.linspace(0, 1, num_points_v)
    uu, vv = np.meshgrid(u, v)

    # Flatten the grids
    uu = uu.flatten()
    vv = vv.flatten()

    # Bilinear interpolation over the face
    points = (1 - uu)[:, np.newaxis] * (1 - vv)[:, np.newaxis] * v0 + \
             uu[:, np.newaxis] * (1 - vv)[:, np.newaxis] * v1 + \
             uu[:, np.newaxis] * vv[:, np.newaxis] * v2 + \
             (1 - uu)[:, np.newaxis] * vv[:, np.newaxis] * v3

    return points

def generate_mask_points_basis(start_point, face_basis, x_start, x_stop, y_start, y_stop, density=100.):
    x_vals = np.linspace(x_start, x_stop, int(density))
    y_vals = np.linspace(y_start, y_stop, int(density))

    mask_points = []
    for x in x_vals:
        for y in y_vals:
            mask_points.append(start_point + face_basis[0] * x + face_basis[1] * y)

    return mask_points