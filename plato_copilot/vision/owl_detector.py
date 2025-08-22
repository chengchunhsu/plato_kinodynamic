import requests
from PIL import Image
import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

class OwlDetector:
    def __init__(self):
        self.processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
        self.model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
        self.image = None
        self.texts = None
        self.results = None
        self.boxes = None
        self.scores = None
        self.labels = None
    
    def detect(self, image, texts, threshold=0.1):
        self.image = image
        self.texts = texts
        inputs = self.processor(text=texts, images=image, return_tensors="pt")
        outputs = self.model(**inputs)
        target_sizes = torch.Tensor([image.size[::-1]])
        self.results = self.processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=threshold)
        self._extract_detection_results()
    
    def _extract_detection_results(self):
        i = 0  # Assuming we're always dealing with one image and its corresponding texts for simplicity
        text = self.texts[i]
        result = self.results[i]
        self.boxes, self.scores, self.labels = result["boxes"], result["scores"], result["labels"]
        for box, score, label in zip(self.boxes, self.scores, self.labels):
            box = [round(i, 2) for i in box.tolist()]
            print(f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}")
    
    def visualize(self):
        fig, ax = plt.subplots(1)
        ax.imshow(self.image)
        for box, score, label in zip(self.boxes, self.scores, self.labels):
            box = [round(i, 2) for i in box.tolist()]
            rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor="r", facecolor="none")
            ax.add_patch(rect)
            ax.text(box[0], box[1], self.texts[0][label], color="white")
        plt.show()
    
    def get_cropped_image(self):
        cropped_images = []
        for box in self.boxes:
            box = [round(i) for i in box.tolist()]
            cropped_image = self.image.crop((box[0], box[1], box[2], box[3]))
            cropped_images.append(np.array(cropped_image))
        return cropped_images
    
