import pickle
import numpy as np
from matplotlib import pyplot as plt

# Save block_masks_post to a file
def save_block_masks_post(block_masks_post, filename):
    with open(filename, 'wb') as f:
        pickle.dump(block_masks_post, f)
# Load block_masks_post from a file
def load_block_masks_post(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def get_color_map(N, cmap='rainbow'):
    """Generates N visually distinct colors."""
    colormap = plt.get_cmap(cmap)
    return (colormap(np.linspace(0, 1, N))[:, :3] * 255).astype(np.uint8)

def overlay_segmentation(rgb_image, block_masks_post, alpha=0.75):
    h, w = rgb_image.shape[:2]
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    mask = np.zeros((h, w), dtype=bool)

    # Count how many mask groups there are
    total_masks = sum(
        len(block_masks_post[layer][i]) > 0
        for layer in block_masks_post
        for i in range(3)
    )
    colors = get_color_map(total_masks)
    color_index = 0

    for layer in block_masks_post:
        for i in range(3):
            coords = np.array(block_masks_post[layer][i], dtype=int)
            if coords.size == 0:
                continue

            valid = (0 <= coords[:, 0]) & (coords[:, 0] < w) & (0 <= coords[:, 1]) & (coords[:, 1] < h)
            coords = coords[valid]

            y = coords[:, 1]
            x = coords[:, 0]

            overlay[y, x] = colors[color_index]
            mask[y, x] = True
            color_index += 1

    # Alpha blending
    blended = rgb_image.copy()
    blended[mask] = (alpha * overlay[mask] + (1 - alpha) * rgb_image[mask]).astype(np.uint8)
    return blended

def overlay_segmentation_by_layer(rgb_image, block_masks_post, alpha=0.5, x_offset=0, w_threshold=None, h_threshold=None, ignore_layer_list=[]):
    h, w = rgb_image.shape[:2]
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    mask = np.zeros((h, w), dtype=bool)

    layers = list(block_masks_post.keys())
    colors = get_color_map(len(layers))
    colors = colors[:, [2, 1, 0]]

    for idx, layer in enumerate(layers):
        if layer in ignore_layer_list:
            continue

        layer_color = colors[idx]
        for i in range(3):  # assuming 3 groups per layer
            coords = np.array(block_masks_post[layer][i], dtype=int)
            if coords.size == 0:
                continue

            w = w if w_threshold is None else w_threshold
            h_lower = 0 if h_threshold is None else h_threshold
            valid = (0 <= coords[:, 0]) & (coords[:, 0] < w) & (h_lower <= coords[:, 1]) & (coords[:, 1] < h)
            coords = coords[valid]

            y = coords[:, 1]
            x = coords[:, 0] + x_offset

            overlay[y, x] = layer_color
            mask[y, x] = True

    # Alpha blending
    blended = rgb_image.copy()
    blended[mask] = (alpha * overlay[mask] + (1 - alpha) * rgb_image[mask]).astype(np.uint8)



    # Define parameters
    center = (711, 532)
    radius = 10
    fill_color = colors[3] / 1.     # fill
    outline_color = colors[3] - 100. # outline
    print(fill_color)
    outline_thickness = 3

    # Draw filled circle
    import cv2
    cv2.circle(blended, center, radius, fill_color, -1)

    # Draw outline on top
    cv2.circle(blended, center, radius, outline_color, outline_thickness)


    return blended