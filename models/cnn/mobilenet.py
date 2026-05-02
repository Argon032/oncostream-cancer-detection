"""
mobilenet.py
------------
MobileNetV2 model with pretrained ImageNet weights.
Final classification layer is replaced to match the target dataset.

Usage:
    from models.cnn.mobilenet import get_model
    model = get_model(num_classes=4)   # brain: 4 classes
    model = get_model(num_classes=2)   # breast: 2 classes
"""

import torch.nn as nn
from torchvision import models


def get_model(num_classes: int) -> nn.Module:
    """
    Load pretrained MobileNetV2 and replace the final classifier
    layer to output `num_classes` predictions.

    Strategy:
        - Backbone layers are fully fine-tuned (no freezing).
        - The original classifier[1] linear layer is replaced.
        - Dropout added for regularisation on small medical datasets.

    Args:
        num_classes: Number of output classes.
                     4 for brain (Glioma, Meningioma, Pituitary, No Tumor)
                     2 for breast (Benign, Malignant)

    Returns:
        nn.Module: Modified MobileNetV2 ready for training or inference.
    """
    weights = models.MobileNet_V2_Weights.DEFAULT
    model = models.mobilenet_v2(weights=weights)

    # MobileNetV2 classifier: [Dropout(0.2), Linear(1280, 1000)]
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes)
    )

    return model

if __name__ == "__main__":
    import torch
    model = get_model(num_classes=2)
    dummy = torch.randn(2, 3, 224, 224)
    out = model(dummy)
    print(f"MobileNetV2 output shape: {out.shape}")  # (2, 2)
