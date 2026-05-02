"""
swin.py
-------
Swin Transformer (Swin-T) model with pretrained ImageNet weights.
Final classification head is replaced to match the target dataset.

Usage:
    from models.transformer.swin import get_model, get_param_groups
    model = get_model(num_classes=4)
    param_groups = get_param_groups(model, head_lr=1e-4, backbone_lr=1e-5)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.05)
"""

import torch.nn as nn
from torchvision import models


def get_model(num_classes: int) -> nn.Module:
    """
    Load pretrained Swin-T and replace the classification head.

    Unfreezing strategy:
        - stages 0 and 1 (early texture features): FROZEN
        - stages 2 and 3 (semantic features): TRAINABLE
        - head: TRAINABLE (always, re-initialised)

    This gives the model enough trainable capacity to specialise on
    medical images without needing a huge dataset.

    Args:
        num_classes: Number of output classes.
                     4 for brain (Glioma, Meningioma, Pituitary, No Tumor)
                     2 for breast (Benign, Malignant)

    Returns:
        nn.Module: Modified Swin-T ready for training or inference.
    """
    weights = models.Swin_T_Weights.DEFAULT
    model = models.swin_t(weights=weights)

    # Freeze early stages
    layers_to_freeze = [
        model.features[0],  # PatchMerging / PatchEmbedding
        model.features[1],  # Stage 0
        model.features[2],  # Patch Merging
        model.features[3],  # Stage 1
    ]
    for layer in layers_to_freeze:
        for param in layer.parameters():
            param.requires_grad = False

    in_features = model.head.in_features
    model.head = nn.Sequential(
        nn.LayerNorm(in_features),
        nn.Dropout(p=0.2),
        nn.Linear(in_features, num_classes)
    )

    return model


def get_param_groups(model: nn.Module, head_lr: float = 1e-4, backbone_lr: float = 1e-5) -> list:
    """
    Return two parameter groups for differential learning rates (important for transformer fine-tuning):
        - backbone (unfrozen transformer stages): lower LR
        - classification head: higher LR

    Args:
        model: The Swin-T model returned by get_model().
        head_lr: Learning rate for the new classification head.
        backbone_lr: Learning rate for unfrozen backbone layers.

    Returns:
        list of dicts compatible with any torch.optim optimiser.
    """
    head_params = list(model.head.parameters())
    head_param_ids = set(id(p) for p in head_params)

    backbone_params = [
        p for p in model.parameters()
        if p.requires_grad and id(p) not in head_param_ids
    ]

    return [
        {"params": backbone_params, "lr": backbone_lr},
        {"params": head_params,     "lr": head_lr},
    ]


if __name__ == "__main__":
    import torch
    model = get_model(num_classes=4)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Swin-T  |  Total params: {total:,}  |  Trainable: {trainable:,}")

    dummy = torch.randn(2, 3, 224, 224)
    out = model(dummy)
    print(f"Output shape: {out.shape}")   # (2, 4)
