"""
Faster RCNN class making use of fasterrcnn_resnet50_fpn_v2. I hope this is set up correctly.

The forward pass, if in evaluation mode, returns both losses and detections for validation during
training.
"""

import torch
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn_v2
from torchvision.transforms import v2


def get_training_transforms(image_size):
    return v2.Compose([
        v2.Resize((image_size, image_size)),
        v2.RandomAffine(degrees=30, shear=15, translate=(0.1, 0.1), scale=(0.6, 1.2)),
        v2.RandomHorizontalFlip(p=0.2),
        v2.RandomErasing(0.5, scale=(0.02, 0.08)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True)
    ])


def get_validation_transforms(image_size):
    return v2.Compose([
        v2.Resize((image_size, image_size)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True)
    ])


class CustomFasterRCNN:
    def __init__(self, num_classes):
        self.model = fasterrcnn_resnet50_fpn_v2(weights=None, num_classes=num_classes)

    def forward(self, images, targets=None):
        # If model.train() call standard training function.
        if self.model.training:
            return self.model(images, targets)
        else:
            # Set the model to training mode temporarily to get losses
            with torch.no_grad():
                self.model.train()
                losses = self.model.forward(images, targets)
                # Set the model back to evaluation mode
                self.model.eval()
                detections = self.model.forward(images)
            return losses, detections
