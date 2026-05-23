"""
swin.py
-------
Swin Transformer (Swin-T) model with pretrained ImageNet weights.
Final classification head is replaced to match the target dataset.

Two modes controlled by the `freeze_stages` parameter:

    freeze_stages=True  (original):
        Stages 0 and 1 frozen, stages 2 and 3 trainable.
        Use for breast dataset or if compute is limited.

    freeze_stages=False (improved — recommended for brain):
        No backbone layers frozen. Full model trains with differential
        LRs (backbone at backbone_lr, head at head_lr).
        Gives the model more capacity to adapt to medical images that
        look very different from ImageNet.

Usage:
    from models.transformer.swin import get_model, get_param_groups

    # Original
    model = get_model(num_classes=4, freeze_stages=True)

    # Improved — more trainable capacity
    model = get_model(num_classes=4, freeze_stages=False)

    param_groups = get_param_groups(model, head_lr=1e-4, backbone_lr=1e-5)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.05)
"""

import torch.nn as nn
from torchvision import models


def get_model(num_classes: int, freeze_stages: bool = False) -> nn.Module:
    """
    Load pretrained Swin-T and replace the classification head.

    Args:
        num_classes:   Number of output classes.
                       4 for brain, 2 for breast.
        freeze_stages: If True, freezes early backbone stages 0 and 1
                       (original behaviour).
                       If False, entire backbone is trainable — recommended
                       for brain MRI where features differ significantly
                       from ImageNet pretraining.

    Returns:
        nn.Module: Modified Swin-T ready for training or inference.
    """
    weights = models.Swin_T_Weights.DEFAULT
    model   = models.swin_t(weights=weights)

    if freeze_stages:
        # Original: freeze patch embedding and first two transformer stages
        layers_to_freeze = [
            model.features[0],   # Patch embedding
            model.features[1],   # Stage 0
            model.features[2],   # Patch merging
            model.features[3],   # Stage 1
        ]
        for layer in layers_to_freeze:
            for param in layer.parameters():
                param.requires_grad = False
    # If freeze_stages=False — nothing frozen, full model trains with
    # differential LRs set in get_param_groups()

    # Replace classification head
    in_features = model.head.in_features
    model.head  = nn.Sequential(
        nn.LayerNorm(in_features),
        nn.Dropout(p=0.2),
        nn.Linear(in_features, num_classes)
    )

    return model


def get_param_groups(model: nn.Module,
                     head_lr: float = 1e-4,
                     backbone_lr: float = 1e-5) -> list:
    """
    Return two parameter groups for differential learning rates.

    Backbone (all non-head trainable params) → backbone_lr (lower)
    Classification head                      → head_lr    (higher)

    This protects pretrained weights from being overwritten too fast
    while allowing the head to learn quickly.

    Args:
        model:       Swin-T model from get_model().
        head_lr:     LR for the new classification head.
        backbone_lr: LR for backbone layers.

    Returns:
        List of param group dicts for torch.optim.
    """
    head_params     = list(model.head.parameters())
    head_param_ids  = set(id(p) for p in head_params)

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

    for freeze in [True, False]:
        model    = get_model(num_classes=4, freeze_stages=freeze)
        total    = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"freeze_stages={freeze}  |  Total: {total:,}  |  Trainable: {trainable:,}")

    dummy = torch.randn(2, 3, 224, 224)
    out   = model(dummy)
    print(f"Output shape: {out.shape}")