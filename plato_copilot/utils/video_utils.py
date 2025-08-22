import h5py
import numpy as np
import imageio
import cv2
import os

class VideoWriter():
    def __init__(self, video_path, save_video=False, video_name=None, fps=30, single_video=True):
        self.video_path = video_path
        self.save_video = save_video
        self.fps = fps
        self.image_buffer = {}
        self.single_video = single_video
        self.last_images = {}
        if video_name is None:
            self.video_name = "video.mp4"
        else:
            self.video_name = video_name

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.save(self.video_name)

    def reset(self):
        if self.save_video:
            self.last_images = {}

    def append_image(self, image, idx=0):
        if self.save_video:
            if idx not in self.image_buffer:
                self.image_buffer[idx] = []
            if idx not in self.last_images:
                self.last_images[idx] = None
            self.image_buffer[idx].append(image[::-1])

    def append_vector_obs(self, images):
        if self.save_video:
            for i in range(len(images)):
                self.append_image(images[i], i)

    def save(self, video_name=None, flip=False, bgr=False):
        if video_name is None:
            video_name = self.video_name
        img_convention = 1
        color_convention = 1
        if flip:
            img_convention = -1
        if bgr:
            color_convention = -1
        if self.save_video:
            os.makedirs(self.video_path, exist_ok=True)
            if self.single_video:
                video_name = os.path.join(self.video_path, video_name)
                video_writer = imageio.get_writer(video_name, fps=self.fps)
                for idx in self.image_buffer.keys():
                    for im in self.image_buffer[idx]:
                        video_writer.append_data(im[::img_convention, :, ::color_convention])
                video_writer.close()
            else:
                for idx in self.image_buffer.keys():
                    video_name = os.path.join(self.video_path, f"{idx}.mp4")
                    video_writer = imageio.get_writer(video_name, fps=self.fps)
                    for im in self.image_buffer[idx]:
                        video_writer.append_data(im[::img_convention, :, ::color_convention])
                    video_writer.close()
            print(f"Saved videos to {video_name}.")


def convert_webm_to_mp4(input_path, output_path):
     # Read the .webm file
    cap = cv2.VideoCapture(input_path)

    # Check if the video file has been opened successfully
    if not cap.isOpened():
        print("Error: Couldn't open the video file.")
        return

    # Get video properties
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # or use 'XVID'
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        out.write(frame)

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()