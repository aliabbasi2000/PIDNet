# ------------------------------------------------------------------------------
# Implementation of LoveDA dataset class, compatible with PIDNet
# ------------------------------------------------------------------------------

import os
import cv2
import numpy as np
from PIL import Image

import torch
from .base_dataset import BaseDataset

class LoveDA(BaseDataset):
    def __init__(self, 
                 root, 
                 list_path,
                 num_classes=7,
                 multi_scale=True, 
                 flip=True, 
                 ignore_label=255, 
                 base_size=1024, 
                 crop_size=(512, 512),
                 scale_factor=16,
                 mean=[0.485, 0.456, 0.406], 
                 std=[0.229, 0.224, 0.225],
                 bd_dilate_size=4):

        super(LoveDA, self).__init__(ignore_label, base_size,
                                     crop_size, scale_factor, mean, std)

        self.root = root
        self.list_path = list_path
        self.num_classes = num_classes

        self.multi_scale = multi_scale
        self.flip = flip
        
        # Load image and mask paths from list file
        self.img_list = [line.strip().split() for line in open(os.path.join(root, list_path))]

        # Parse the image and label paths into structured files
        self.files = self.read_files()

        # Define class weights for LoveDA (these are placeholders, adjust if needed)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_weights = torch.FloatTensor([1.0, 2.0, 2.0, 1.5, 1.5, 1.0, 1.2])


        self.bd_dilate_size = bd_dilate_size

    def read_files(self):
        """Parse image and label paths from the list."""
        files = []
        for item in self.img_list:
            image_path, label_path = item
            name = os.path.splitext(os.path.basename(label_path))[0]
            files.append({
                "img": image_path,
                "label": label_path,
                "name": name
            })
        return files

    def __getitem__(self, index):
        """Load and preprocess an image and its corresponding label."""
        item = self.files[index]
        name = item["name"]
        image = cv2.imread(os.path.join(self.root, item["img"]), cv2.IMREAD_COLOR)
        size = image.shape

        label = cv2.imread(os.path.join(self.root, item["label"]), cv2.IMREAD_GRAYSCALE)

        # Generate the transformed sample (image, label, and edge)
        image, label, edge = self.gen_sample(image, label, 
                                             self.multi_scale, self.flip, 
                                             edge_size=self.bd_dilate_size)

        return image.copy(), label.copy(), edge.copy(), np.array(size), name

    def single_scale_inference(self, config, model, image):
        """Perform single-scale inference."""
        pred = self.inference(config, model, image)
        return pred

    def save_pred(self, preds, sv_path, name):
        """Save predictions as PNG images."""
        preds = np.asarray(np.argmax(preds.cpu(), axis=1), dtype=np.uint8)
        for i in range(preds.shape[0]):
            save_img = Image.fromarray(preds[i])
            save_img.save(os.path.join(sv_path, name[i] + '.png'))
