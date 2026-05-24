"""
metrics.py
----------
Computes and saves full evaluation metrics for a trained model on the
test set. Outputs:

    Console:
        - Per-class precision, recall, F1, support
        - Macro averages
        - AUC-ROC scores (per class + macro average)

    Files saved to results/{dataset}/:
        - {model}_metrics.csv          → precision, recall, F1, AUC per class
        - {model}_roc_curve.png        → ROC curve for each class
        - {model}_f1_bar.png           → per-class F1 bar chart

HOW TO USE
----------
Edit CONFIG and run:
    python metrics.py
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import csv
import torch
import torch.nn as nn

from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
CONFIG = {
    "model_name":   "vit",
    "dataset":      "brain",
    "project_root": ".",
    "batch_size":   32,
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

# Consistent colours per class across all plots
CLASS_COLORS = [
    "#E91E63", "#2196F3", "#4CAF50", "#FF9800",
    "#9C27B0", "#00BCD4",
]


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
def get_predictions(model, test_loader, num_classes: int):
    """Returns true labels, predicted labels, and softmax probabilities."""
    all_labels, all_preds, all_probs = [], [], []

    for images, labels in test_loader:
        images  = images.to(DEVICE)
        outputs = model(images)
        probs   = torch.softmax(outputs, dim=1).cpu().numpy()
        preds   = outputs.argmax(dim=1).cpu().tolist()

        all_labels.extend(labels.tolist())
        all_preds.extend(preds)
        all_probs.extend(probs)

    return (
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs),
    )


def plot_roc_curves(all_labels, all_probs, labels, model_name,
                    dataset, save_path):
    """
    Plots one ROC curve per class (One-vs-Rest) plus the macro average.
    For binary classification, plots a single curve using the positive
    class probability directly (label_binarize returns shape (n,1) for
    2 classes which causes an IndexError if treated as multi-class).
    """
    num_classes = len(labels)
    fig, ax     = plt.subplots(figsize=(8, 6))
    aucs        = []

    if num_classes == 2:
        # Binary — use probability of positive class (index 1) directly
        fpr, tpr, _ = roc_curve(all_labels, all_probs[:, 1])
        auc_score   = roc_auc_score(all_labels, all_probs[:, 1])
        aucs.append(auc_score)
        aucs.append(auc_score)   # same value for both classes
        ax.plot(fpr, tpr,
                label=f"{labels[1]}  (AUC = {auc_score:.4f})",
                color=CLASS_COLORS[0], linewidth=2)
    else:
        # Multi-class — One-vs-Rest
        labels_bin = label_binarize(all_labels, classes=list(range(num_classes)))
        for i, label in enumerate(labels):
            fpr, tpr, _ = roc_curve(labels_bin[:, i], all_probs[:, i])
            auc_score   = roc_auc_score(labels_bin[:, i], all_probs[:, i])
            aucs.append(auc_score)
            ax.plot(fpr, tpr,
                    label=f"{label}  (AUC = {auc_score:.4f})",
                    color=CLASS_COLORS[i % len(CLASS_COLORS)],
                    linewidth=2)

    macro_auc = np.mean(aucs)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random (AUC = 0.5)")

    ax.set(
        title=f"ROC Curves — {model_name.upper()} on {dataset} dataset"
              f"\nMacro AUC = {macro_auc:.4f}",
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        xlim=[0.0, 1.0],
        ylim=[0.0, 1.05],
    )
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"ROC curve saved → {save_path}")
    return aucs, macro_auc


def plot_f1_bar(labels, f1_scores, model_name, dataset, save_path):
    """Bar chart of per-class F1 scores."""
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.8), 5))

    bars = ax.bar(labels, f1_scores,
                  color=CLASS_COLORS[:len(labels)],
                  edgecolor="white", linewidth=0.8)

    # Annotate bars
    for bar, score in zip(bars, f1_scores):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{score:.4f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set(
        title=f"Per-Class F1 Score — {model_name.upper()} on {dataset} dataset",
        xlabel="Class",
        ylabel="F1 Score",
        ylim=[0, 1.1],
    )
    ax.axhline(y=np.mean(f1_scores), color="gray", linestyle="--",
               linewidth=1.2, label=f"Macro avg = {np.mean(f1_scores):.4f}")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"F1 bar chart saved → {save_path}")


def main():
    cfg        = CONFIG
    model_name = cfg["model_name"]
    dataset    = cfg["dataset"]
    info       = CLASS_INFO[dataset]
    labels     = info["labels"]
    num_classes = info["num_classes"]

    results_dir = os.path.join(PROJECT_ROOT, "results", dataset)
    plots_dir   = os.path.join(PROJECT_ROOT, "results", "plots")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir,   exist_ok=True)

    # Dataloader
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "preprocessing"))
    from dataloader import get_dataloaders

    dataset_path = os.path.join(PROJECT_ROOT, "datasets", dataset)
    _, _, test_loader = get_dataloaders(dataset_path,
                                        batch_size=cfg["batch_size"],
                                        num_workers=cfg["num_workers"])

    # Model + predictions
    model = load_model(model_name, num_classes, dataset)
    all_labels, all_preds, all_probs = get_predictions(
        model, test_loader, num_classes
    )

    # ── Core metrics ───────────────────────────────────────────────────────────
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds,
        average=None,
        labels=list(range(num_classes))
    )

    # ── AUC-ROC ────────────────────────────────────────────────────────────────
    if num_classes == 2:
        aucs      = [roc_auc_score(all_labels, all_probs[:, 1])] * 2
        macro_auc = aucs[0]
    else:
        labels_bin = label_binarize(all_labels, classes=list(range(num_classes)))
        aucs       = [
            roc_auc_score(labels_bin[:, i], all_probs[:, i])
            for i in range(num_classes)
        ]
        macro_auc = np.mean(aucs)

    # ── Console output ─────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  Model   : {model_name.upper()}")
    print(f"  Dataset : {dataset}")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Macro AUC-ROC: {macro_auc:.4f}")
    print(f"{'='*65}")
    print(f"\n{'Class':<20} {'Prec':>8} {'Recall':>8} "
          f"{'F1':>8} {'AUC':>8} {'Support':>9}")
    print("-" * 65)

    rows = []
    for i, label in enumerate(labels):
        print(f"{label:<20} {precision[i]:>8.4f} {recall[i]:>8.4f} "
              f"{f1[i]:>8.4f} {aucs[i]:>8.4f} {int(support[i]):>9}")
        rows.append({
            "class":     label,
            "precision": round(float(precision[i]), 4),
            "recall":    round(float(recall[i]),    4),
            "f1":        round(float(f1[i]),        4),
            "auc":       round(float(aucs[i]),      4),
            "support":   int(support[i]),
        })

    print("-" * 65)
    print(f"{'macro avg':<20} {precision.mean():>8.4f} {recall.mean():>8.4f} "
          f"{f1.mean():>8.4f} {macro_auc:>8.4f}")
    rows.append({
        "class":     "macro avg",
        "precision": round(float(precision.mean()), 4),
        "recall":    round(float(recall.mean()),    4),
        "f1":        round(float(f1.mean()),        4),
        "auc":       round(float(macro_auc),        4),
        "support":   len(all_labels),
    })

    # ── Save CSV ───────────────────────────────────────────────────────────────
    csv_path = os.path.join(results_dir, f"{model_name}_metrics.csv")
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nMetrics CSV saved → {csv_path}")

    # ── ROC curve plot ─────────────────────────────────────────────────────────
    plot_roc_curves(
        all_labels, all_probs, labels, model_name, dataset,
        save_path=os.path.join(plots_dir,
                               f"{model_name}_{dataset}_roc_curve.png")
    )

    # ── F1 bar chart ───────────────────────────────────────────────────────────
    plot_f1_bar(
        labels, f1, model_name, dataset,
        save_path=os.path.join(plots_dir,
                               f"{model_name}_{dataset}_f1_bar.png")
    )


if __name__ == "__main__":
    main()