import numpy as np
import cv2
import colorsys

from PIL import Image

class ImageProcessor():
    def __init__(self):
        pass

    def get_fx_fy_dict(self, img_size=224):
        if img_size == 224:
            fx_fy_dict = {
                "k4a": {
                    0: {"fx": 0.35, "fy": 0.35},
                    1: {"fx": 0.35, "fy": 0.35},
                    2: {"fx": 0.4, "fy": 0.6}
                },
                "rs": {
                    0: {"fx": 0.49, "fy": 0.49},
                    1: {"fx": 0.49, "fy": 0.49},
                }
            }
        # elif img_size == 128:
        #     fx_fy_dict = {0: {"fx": 0.2, "fy": 0.2}, 1: {"fx": 0.2, "fy": 0.2}, 2: {"fx": 0.2, "fy": 0.3}}
        # elif img_size == 84:
        #     fx_fy_dict = {0: {"fx": 0.13, "fy": 0.13}, 1: {"fx": 0.13, "fy": 0.13}, 2: {"fx": 0.15, "fy": 0.225}}
        return fx_fy_dict

    def resize_img(
                self,
                img: np.ndarray,
                camera_type: str, 
                img_w: int=224, 
                img_h: int=224, 
                offset_w: int=0, 
                offset_h: int=0,
                fx: float=None,
                fy: float=None) -> np.ndarray:
        if camera_type == "k4a":
            if fx is None:
                fx = 0.2
            if fy is None:
                fy = 0.2
            resized_img = cv2.resize(img, (0, 0), fx=fx, fy=fy, interpolation = cv2.INTER_NEAREST)
            w = resized_img.shape[0]
            h = resized_img.shape[1]

        if camera_type == "rs":
            if fx is None:
                fx = 0.2
            if fy is None:
                fy = 0.3
            resized_img = cv2.resize(img, (0, 0), fx=fx, fy=fy, interpolation = cv2.INTER_NEAREST)
            w = resized_img.shape[0]
            h = resized_img.shape[1]

        resized_img = resized_img[w//2-img_w//2:w//2+img_w//2, h//2-img_h//2:h//2+img_h//2, ...]
        return resized_img

    def resize_intrinsics(
            self,
            original_image_size: np.ndarray,
            intrinsic_matrix: np.ndarray,
            camera_type: str,
            img_w: int=224,
            img_h: int=224,
            fx: float=None,
            fy: float=None) -> np.ndarray:
        if camera_type == "k4a":
            if fx is None:
                fx = 0.2
            if fy is None:
                fy = 0.2
        elif camera_type == "rs":
            if fx is None:
                fx = 0.2
            if fy is None:
                fy = 0.3
            
        fake_image = np.zeros((original_image_size[0], original_image_size[1], 3))

        resized_img = cv2.resize(fake_image, (0, 0), fx=fx, fy=fy)
        new_intrinsic_matrix = intrinsic_matrix.copy()
        w, h = resized_img.shape[0], resized_img.shape[1]
        new_intrinsic_matrix[0, 0] = intrinsic_matrix[0, 0] * fx
        new_intrinsic_matrix[1, 1] = intrinsic_matrix[1, 1] * fy
        new_intrinsic_matrix[0, 2] = intrinsic_matrix[0, 2] * fx
        new_intrinsic_matrix[1, 2] = intrinsic_matrix[1, 2] * fy
        new_intrinsic_matrix[0, 2] = new_intrinsic_matrix[0, 2] - (w//2-img_w//2)
        new_intrinsic_matrix[1, 2] = new_intrinsic_matrix[1, 2] - (h//2-img_h//2)
        return new_intrinsic_matrix

def create_grid_picture(images, num_rows, num_cols):
    # create a grid of images with 10 pixels of padding between each image horizontally and vertically
    grid_image = np.ones(
        (images[0].shape[0] * num_rows + 10 * (num_rows - 1), images[0].shape[1] * num_cols + 10 * (num_cols - 1), 3),
        dtype=np.uint8) * 255

    for i, image in enumerate(images):
        row = i // num_cols
        col = i % num_cols

        grid_image[row * images[0].shape[0] + 10 * row: (row + 1) * images[0].shape[0] + 10 * row,
        col * images[0].shape[1] + 10 * col: (col + 1) * images[0].shape[1] + 10 * col] = image

    return grid_image

def load_depth(depth_img_name):
    return cv2.imread(depth_img_name, cv2.IMREAD_UNCHANGED)

def evaluate_plane(plane_model, point):
    return plane_model[0] * point[0] + plane_model[1] * point[1] + plane_model[2] * point[2] + plane_model[3]

def image_to_world_coords(image_coords, depth_data, camera_matrix, transformation_matrix):
    # image coords is a list of (x, y)
    points = np.array(np.concatenate([image_coords, np.ones((len(image_coords), 1))], axis=1), dtype=np.float32)
    points = np.dot(np.linalg.inv(camera_matrix), points.T).T

    for i in range(len(points)):
        points[i] *= depth_data[image_coords[i][1], image_coords[i][0], 0] / 1000.

    # add 1 to the last column
    points = np.concatenate([points, np.ones((len(points), 1))], axis=1)

    points = np.dot(transformation_matrix, points.T).T

    return points[:, :3]


def world_to_image_coords(world_coords, camera_matrix, transformation_matrix):
    # print("World coords shape: ", world_coords.shape)
    world_coords = np.concatenate(
        [world_coords, np.ones((world_coords.shape[0], 1))], axis=1)

    # transform to camera coordinates
    camera_coordinates = np.dot(np.linalg.inv(transformation_matrix),
                                np.array(world_coords).T).T

    # remove last dimension
    camera_coordinates = camera_coordinates[:, :3]

    # divide by z
    camera_coordinates /= camera_coordinates[:, 2].reshape(-1, 1)

    # transform to image coordinates
    image_coords = np.dot(camera_matrix, camera_coordinates.T).T

    return image_coords


def id_to_rgb(id, hue_cycle=50, saturation=0.7, lightness=0.5):
    """
    Convert an integer ID to an RGB color.

    Parameters:
    - id (int): The unique identifier.
    - saturation (float): Saturation component (0 to 1).
    - lightness (float): Lightness component (0 to 1).

    Returns:
    - tuple: RGB color as (R, G, B), each ranging from 0 to 255.
    """
    # Number of distinct hues you want before cycling
    hue = (id * 137.508) % hue_cycle  # 137.508 is the golden angle in degrees

    # Convert hue to [0,1] for colorsys
    h = hue / hue_cycle
    s = saturation
    l = lightness

    # Convert HSL to RGB
    r, g, b = colorsys.hls_to_rgb(h, l, s)

    # Scale RGB to [0,255]
    return int(r * 255), int(g * 255), int(b * 255)

def get_palette(palette="davis"):
    davis_palette = b'\x00\x00\x00\x80\x00\x00\x00\x80\x00\x80\x80\x00\x00\x00\x80\x80\x00\x80\x00\x80\x80\x80\x80\x80@\x00\x00\xc0\x00\x00@\x80\x00\xc0\x80\x00@\x00\x80\xc0\x00\x80@\x80\x80\xc0\x80\x80\x00@\x00\x80@\x00\x00\xc0\x00\x80\xc0\x00\x00@\x80\x80@\x80\x00\xc0\x80\x80\xc0\x80@@\x00\xc0@\x00@\xc0\x00\xc0\xc0\x00@@\x80\xc0@\x80@\xc0\x80\xc0\xc0\x80\x00\x00@\x80\x00@\x00\x80@\x80\x80@\x00\x00\xc0\x80\x00\xc0\x00\x80\xc0\x80\x80\xc0@\x00@\xc0\x00@@\x80@\xc0\x80@@\x00\xc0\xc0\x00\xc0@\x80\xc0\xc0\x80\xc0\x00@@\x80@@\x00\xc0@\x80\xc0@\x00@\xc0\x80@\xc0\x00\xc0\xc0\x80\xc0\xc0@@@\xc0@@@\xc0@\xc0\xc0@@@\xc0\xc0@\xc0@\xc0\xc0\xc0\xc0\xc0 \x00\x00\xa0\x00\x00 \x80\x00\xa0\x80\x00 \x00\x80\xa0\x00\x80 \x80\x80\xa0\x80\x80`\x00\x00\xe0\x00\x00`\x80\x00\xe0\x80\x00`\x00\x80\xe0\x00\x80`\x80\x80\xe0\x80\x80 @\x00\xa0@\x00 \xc0\x00\xa0\xc0\x00 @\x80\xa0@\x80 \xc0\x80\xa0\xc0\x80`@\x00\xe0@\x00`\xc0\x00\xe0\xc0\x00`@\x80\xe0@\x80`\xc0\x80\xe0\xc0\x80 \x00@\xa0\x00@ \x80@\xa0\x80@ \x00\xc0\xa0\x00\xc0 \x80\xc0\xa0\x80\xc0`\x00@\xe0\x00@`\x80@\xe0\x80@`\x00\xc0\xe0\x00\xc0`\x80\xc0\xe0\x80\xc0 @@\xa0@@ \xc0@\xa0\xc0@ @\xc0\xa0@\xc0 \xc0\xc0\xa0\xc0\xc0`@@\xe0@@`\xc0@\xe0\xc0@`@\xc0\xe0@\xc0`\xc0\xc0\xe0\xc0\xc0\x00 \x00\x80 \x00\x00\xa0\x00\x80\xa0\x00\x00 \x80\x80 \x80\x00\xa0\x80\x80\xa0\x80@ \x00\xc0 \x00@\xa0\x00\xc0\xa0\x00@ \x80\xc0 \x80@\xa0\x80\xc0\xa0\x80\x00`\x00\x80`\x00\x00\xe0\x00\x80\xe0\x00\x00`\x80\x80`\x80\x00\xe0\x80\x80\xe0\x80@`\x00\xc0`\x00@\xe0\x00\xc0\xe0\x00@`\x80\xc0`\x80@\xe0\x80\xc0\xe0\x80\x00 @\x80 @\x00\xa0@\x80\xa0@\x00 \xc0\x80 \xc0\x00\xa0\xc0\x80\xa0\xc0@ @\xc0 @@\xa0@\xc0\xa0@@ \xc0\xc0 \xc0@\xa0\xc0\xc0\xa0\xc0\x00`@\x80`@\x00\xe0@\x80\xe0@\x00`\xc0\x80`\xc0\x00\xe0\xc0\x80\xe0\xc0@`@\xc0`@@\xe0@\xc0\xe0@@`\xc0\xc0`\xc0@\xe0\xc0\xc0\xe0\xc0  \x00\xa0 \x00 \xa0\x00\xa0\xa0\x00  \x80\xa0 \x80 \xa0\x80\xa0\xa0\x80` \x00\xe0 \x00`\xa0\x00\xe0\xa0\x00` \x80\xe0 \x80`\xa0\x80\xe0\xa0\x80 `\x00\xa0`\x00 \xe0\x00\xa0\xe0\x00 `\x80\xa0`\x80 \xe0\x80\xa0\xe0\x80``\x00\xe0`\x00`\xe0\x00\xe0\xe0\x00``\x80\xe0`\x80`\xe0\x80\xe0\xe0\x80  @\xa0 @ \xa0@\xa0\xa0@  \xc0\xa0 \xc0 \xa0\xc0\xa0\xa0\xc0` @\xe0 @`\xa0@\xe0\xa0@` \xc0\xe0 \xc0`\xa0\xc0\xe0\xa0\xc0 `@\xa0`@ \xe0@\xa0\xe0@ `\xc0\xa0`\xc0 \xe0\xc0\xa0\xe0\xc0``@\xe0`@`\xe0@\xe0\xe0@``\xc0\xe0`\xc0`\xe0\xc0\xe0\xe0\xc0'
    youtube_palette = b'\x00\x00\x00\xec_g\xf9\x91W\xfa\xc8c\x99\xc7\x94b\xb3\xb2f\x99\xcc\xc5\x94\xc5\xabyg\xff\xff\xffes~\x0b\x0b\x0b\x0c\x0c\x0c\r\r\r\x0e\x0e\x0e\x0f\x0f\x0f'
    if palette == "davis":
        return davis_palette
    elif palette == "youtube":
        return youtube_palette

def overlay_xmem_mask_on_image(rgb_img, mask, use_white_bg=False, rgb_alpha=0.7):
    """

    Args:
        rgb_img (np.ndarray):rgb images
        mask (np.ndarray)): binary mask
        use_white_bg (bool, optional): Use white backgrounds to visualize overlap. Note that we assume mask ids 0 as the backgrounds. Otherwise the visualization might be screws up. . Defaults to False.

    Returns:
        np.ndarray: overlay image of rgb_img and mask
    """
    colored_mask = Image.fromarray(mask)
    colored_mask.putpalette(get_palette())
    colored_mask = np.array(colored_mask.convert("RGB"))
    if use_white_bg:
        colored_mask[mask == 0] = [255, 255, 255]
    overlay_img = cv2.addWeighted(rgb_img, rgb_alpha, colored_mask, 1-rgb_alpha, 0)

    return overlay_img


def resize_sam_image(image, sam_model="xl0"):
    raw_image = np.array(image)
    H, W, _ = raw_image.shape
    if sam_model in ["xl0", "xl1"]:
        max_size = 1024
    else:
        max_size = 512

    new_W = W
    new_H = H

    scaling = 1
    if H > W and H > max_size:
        new_H = max_size
        new_W = int(W * (new_H / H))
        scaling = new_H / H
    elif W > max_size:
        new_W = max_size
        new_H = int(H * (new_W / W))
        scaling = new_W / W
    
    raw_image = cv2.resize(raw_image, (new_W, new_H))

    H, W, _ = raw_image.shape

    raw_image = np.array(raw_image)
    return raw_image

def resize_sam_depth(depth_map, sam_model="xl0"):
    H, W = depth_map.shape

    # Determine max size based on SAM model
    max_size = 1024 if sam_model in ["xl0", "xl1"] else 512

    new_W = W
    new_H = H
    scaling = 1

    if H > W and H > max_size:
        new_H = max_size
        new_W = int(W * (new_H / H))
        scaling = new_H / H
    elif W > max_size:
        new_W = max_size
        new_H = int(H * (new_W / W))
        scaling = new_W / W

    # Resize the depth map using nearest-neighbor to preserve depth values
    resized_depth = cv2.resize(depth_map, (new_W, new_H), interpolation=cv2.INTER_NEAREST)
    resized_depth = np.array(resized_depth)
    return resized_depth

def crop_depth(depth, bbox):
    H, W = depth.shape

    # Scale the bbox coordinates to match resized image (if needed)
    x1, y1, x2, y2 = bbox
    x1 = int(np.clip(x1, 0, W - 1))
    y1 = int(np.clip(y1, 0, H - 1))
    x2 = int(np.clip(x2, 0, W))
    y2 = int(np.clip(y2, 0, H))

    # Ensure coordinates are valid
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Invalid bbox: (x2, y2) must be greater than (x1, y1)")

    # Crop the resized depth map
    cropped_depth = depth[y1:y2, x1:x2]
    return cropped_depth
