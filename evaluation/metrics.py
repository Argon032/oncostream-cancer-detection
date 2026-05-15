"""
metrics.py
----------
Computes per-class and overall evaluation metrics for a trained model
on the test set: accuracy, precision, recall, F1-score.

Results are printed to console and saved as a CSV.

"""

import os
import sys
import csv

import torch
import torch.nn as nn
from sklearn.metrics import (classification_report, accuracy_score,
                             precision_recall_fscore_support)

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
CONFIG = {
    # Model: 'resnet50', 'mobilenet', 'swin', 'vit'
    "model_name": "swin",

    # Dataset: 'brain' or 'breast'
    "dataset": "breast",

    "project_root": ".",
    "batch_size": 16,
    "num_workers": 2,
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


def load_model(model_name: str, num_classes: int):
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
    ckpt_path = os.path.join(PROJECT_ROOT, "results",
                             CONFIG["dataset"], f"{model_name}_best.pth")

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
        images = images.to(DEVICE)
        outputs = model(images)
        preds   = outputs.argmax(dim=1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.tolist())

    return all_labels, all_preds


def main():
    cfg        = CONFIG
    model_name = cfg["model_name"]
    dataset    = cfg["dataset"]
    info       = CLASS_INFO[dataset]
    labels     = info["labels"]

    # Dataloader (test set only)
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "preprocessing"))
    from dataloader import get_dataloaders

    dataset_path = os.path.join(PROJECT_ROOT, "datasets", dataset)
    _, _, test_loader = get_dataloaders(dataset_path,
                                        batch_size=cfg["batch_size"],
                                        num_workers=cfg["num_workers"])

    # Model
    model = load_model(model_name, info["num_classes"])

    # Predictions
    all_labels, all_preds = get_predictions(model, test_loader)

    # Metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None, labels=list(range(len(labels)))
    )

    print(f"\n{'='*60}")
    print(f"  Model   : {model_name.upper()}")
    print(f"  Dataset : {dataset}")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"{'='*60}")
    print(f"\n{'Class':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-" * 60)

    rows = []
    for i, label in enumerate(labels):
        print(f"{label:<20} {precision[i]:>10.4f} {recall[i]:>10.4f} "
              f"{f1[i]:>10.4f} {int(support[i]):>10}")
        rows.append({
            "class":     label,
            "precision": round(float(precision[i]), 4),
            "recall":    round(float(recall[i]),    4),
            "f1":        round(float(f1[i]),        4),
            "support":   int(support[i]),
        })

    # Macro averages
    macro_p  = precision.mean()
    macro_r  = recall.mean()
    macro_f1 = f1.mean()
    print("-" * 60)
    print(f"{'macro avg':<20} {macro_p:>10.4f} {macro_r:>10.4f} {macro_f1:>10.4f}")
    rows.append({
        "class": "macro avg",
        "precision": round(float(macro_p),  4),
        "recall":    round(float(macro_r),  4),
        "f1":        round(float(macro_f1), 4),
        "support":   len(all_labels),
    })

    # Save CSV
    results_path = os.path.join(PROJECT_ROOT, "results", dataset)
    os.makedirs(results_path, exist_ok=True)
    csv_path = os.path.join(results_path, f"{model_name}_metrics.csv")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nMetrics saved → {csv_path}")


if __name__ == "__main__":
    main()
