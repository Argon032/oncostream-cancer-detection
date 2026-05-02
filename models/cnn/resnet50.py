"""
resnet50.py
-----------
ResNet-50 model with pretrained ImageNet weights.
Final classification layer is replaced to match the target dataset.

Usage:
    from models.cnn.resnet50 import get_model
    model = get_model(num_classes=4)   # brain: 4 classes
    model = get_model(num_classes=2)   # breast: 2 classes
"""

import torch.nn as nn
from torchvision import models


def get_model(num_classes: int) -> nn.Module:
    """
    Load pretrained ResNet-50 and replace the final fully-connected
    layer to output `num_classes` predictions.

    Strategy:
        - All backbone layers are kept and fine-tuned (not frozen).
        - Only the final FC layer is re-initialised from scratch.
        - A standard learning rate with moderate weight decay is used
          during training; see train_cnn.py for details.

    Args:
        num_classes: Number of output classes.
                     4 for brain (Glioma, Meningioma, Pituitary, No Tumor)
                     2 for breast (Benign, Malignant)

    Returns:
        nn.Module: Modified ResNet-50 ready for training or inference.
    """
    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes)
    )

    return model


if __name__ == "__main__":
    import torch
    model = get_model(num_classes=4)
    dummy = torch.randn(2, 3, 224, 224)
    out = model(dummy)
    print(f"ResNet-50 output shape: {out.shape}")   # (2, 4)
