"""
vit.py
------
Vision Transformer (ViT-B/16) with pretrained ImageNet weights.
Final classification head is replaced to match the target dataset.

Usage:
    from models.transformer.vit import get_model
    model = get_model(num_classes=4)   # brain: 4 classes
    model = get_model(num_classes=2)   # breast: 2 classes
"""

import torch.nn as nn
from torchvision import models


def get_model(num_classes: int) -> nn.Module:
    """
    Load pretrained ViT-B/16 and replace the classification head.

    Strategy:
        - First 6 transformer encoder blocks are frozen (out of 12).
        - Blocks 6-11 and the head are trainable.
        - Standard single linear head (ViT's design is simpler than Swin).
        - Uses a moderate dropout on the head.

    Args:
        num_classes: Number of output classes.
                     4 for brain (Glioma, Meningioma, Pituitary, No Tumor)
                     2 for breast (Benign, Malignant)

    Returns:
        nn.Module: Modified ViT-B/16 ready for training or inference.
    """
    weights = models.ViT_B_16_Weights.DEFAULT
    model = models.vit_b_16(weights=weights)

    for i, block in enumerate(model.encoder.layers):
        if i < 6:
            for param in block.parameters():
                param.requires_grad = False

    in_features = model.heads.head.in_features
    model.heads.head = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes)
    )

    return model


if __name__ == "__main__":
    import torch
    model = get_model(num_classes=4)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"ViT-B/16  |  Total params: {total:,}  |  Trainable: {trainable:,}")

    dummy = torch.randn(2, 3, 224, 224)
    out = model(dummy)
    print(f"Output shape: {out.shape}")   # (2, 4)
