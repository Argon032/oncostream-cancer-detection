"""
confusion_matrix.py
-------------------
Generates and saves a confusion matrix plot for a trained model
on the test set. Output is saved to results/plots/.

HOW TO USE
----------
Edit the CONFIG block and run:
    python confusion_matrix.py
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from sklearn.metrics import confusion_matrix as sk_confusion_matrix

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
CONFIG = {
    # Model: 'resnet50', 'mobilenet', 'swin', 'vit'
    "model_name":   "vit",
    
    # Dataset: 'brain' or 'breast'
    "dataset":      "brain",
    
    "project_root": ".",
    "batch_size":   16,
    "num_workers":  2,
}
# ─────────────────────────────────────────────

PROJECT_ROOT = os.path.abspath(CONFIG["project_root"])
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_INFO = {
    "brain":  {"num_classes": 4,
               "labels": ["glioma", "meningioma", "pituitary", "no_tumor"]},
    "breast": {"num_classes": 2,
               "labels": ["benign", "malignant"]},
}


def load_model(model_name: str, num_classes: int, dataset: str):
    if model_name in ("resnet50", "mobilenet"):
        sys.path.insert(0, os.path.join(PROJECT_ROOT, "models", "cnn"))
        if model_name == "resnet50":
            from resnet50 import get_model
        else:
            from mobilenet import get_model
    else:
        sys.path.insert(0, os.path.join(PROJECT_ROOT, "models", "transformer"))
        if model_name == "swin":
            from swin import get_model
        else:
            from vit import get_model

    model     = get_model(num_classes=num_classes)
    ckpt_path = os.path.join(PROJECT_ROOT, "results", dataset,
                             f"{model_name}_best.pth")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    return model


@torch.no_grad()
def get_predictions(model, test_loader):
    all_preds, all_labels = [], []
    for images, labels in test_loader:
        images  = images.to(DEVICE)
        outputs = model(images)
        preds   = outputs.argmax(dim=1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.tolist())
    return all_labels, all_preds


def plot_confusion_matrix(cm: np.ndarray, labels: list,
                          model_name: str, dataset: str,
                          save_path: str):
    fig, ax = plt.subplots(figsize=(max(5, len(labels) * 1.8),
                                    max(4, len(labels) * 1.6)))

    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        xlabel="Predicted Label",
        ylabel="True Label",
        title=f"Confusion Matrix — {model_name.upper()} on {dataset} dataset",
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    # Annotate cells
    thresh = cm.max() / 2.0
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Confusion matrix saved → {save_path}")


def main():
    cfg        = CONFIG
    model_name = cfg["model_name"]
    dataset    = cfg["dataset"]
    info       = CLASS_INFO[dataset]
    labels     = info["labels"]

    # Dataloader
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "preprocessing"))
    from dataloader import get_dataloaders

    dataset_path = os.path.join(PROJECT_ROOT, "datasets", dataset)
    _, _, test_loader = get_dataloaders(dataset_path,
                                        batch_size=cfg["batch_size"],
                                        num_workers=cfg["num_workers"])

    # Model and predictions
    model = load_model(model_name, info["num_classes"], dataset)
    all_labels, all_preds = get_predictions(model, test_loader)

    # Confusion matrix
    cm = sk_confusion_matrix(all_labels, all_preds,
                             labels=list(range(len(labels))))

    # Save plot
    plots_dir = os.path.join(PROJECT_ROOT, "results", "plots")
    os.makedirs(plots_dir, exist_ok=True)
    save_path = os.path.join(plots_dir,
                             f"{model_name}_{dataset}_confusion_matrix.png")

    plot_confusion_matrix(cm, labels, model_name, dataset, save_path)


if __name__ == "__main__":
    main()
