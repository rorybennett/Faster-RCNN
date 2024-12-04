"""
Custom dataset class for the prostate dataset. Up to two classes will exist, prostate and bladder.
"""
import os

import torch
from PIL import Image
from natsort import natsorted
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F


class ProstateDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None, oversampling_factor=1):
        """
        Set the root and transforms variables. Do some minor dataset validation.

        :param root: Directory containing images and labels folders.
        :param transforms: Transformers to be used on this dataset.
        :param oversampling_factor: Increase dataset size by this amount (oversampling).
        """
        self.root = root
        self.transforms = transforms
        self.oversampling_factor = oversampling_factor
        self.imgs = list(natsorted(os.listdir(os.path.join(root, "images"))))
        self.labels = list(natsorted(os.listdir(os.path.join(root, "labels"))))

        self.validate_dataset()

    def __getitem__(self, idx):
        original_idx = idx // self.oversampling_factor
        img_path = os.path.join(self.root, "images", self.imgs[original_idx])
        label_path = os.path.join(self.root, "labels", self.labels[original_idx])
        img = Image.open(img_path).convert("RGB")

        # Read the label file
        boxes = []
        labels = []
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                class_id = int(parts[0])
                x_center, y_center, width, height = map(float, parts[1:])

                # Convert from YOLO format to FasterRCNN format
                img_width, img_height = img.size
                x_min = (x_center - width / 2) * img_width
                y_min = (y_center - height / 2) * img_height
                x_max = (x_center + width / 2) * img_width
                y_max = (y_center + height / 2) * img_height

                boxes.append([x_min, y_min, x_max, y_max])
                # Add 1 as 0 is always background for some reason.
                labels.append(class_id + 1)
        # Convert boxes to the format expected by the model.
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        img = tv_tensors.Image(img)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)
        target = {
            "boxes": tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=F.get_size(img)),
            "labels": labels,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd
        }

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs) * self.oversampling_factor

    def get_image_count(self):
        return len(self.imgs)

    def validate_dataset(self):
        """
        Minor dataset validation:
            1. Ensure there are the same amount of images as labels.
            2. Ensure the labels and images have the same naming convention.
        """
        # 1
        assert len(self.imgs) == len(self.labels), "Dataset input images and labels are of different length."
        # 2
        for i in range(len(self.imgs)):
            if self.imgs[i].split('.')[0] != self.labels[i].split('.')[0]:
                assert False, f"There is a mismatch between imgs and labels at {self.imgs[i]} and {self.labels[i]}."
