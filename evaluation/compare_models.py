"""
compare_models.py
-----------------
Reads training history CSVs and metrics CSVs for all four models
and generates paper-ready plots and summary tables.

Outputs saved to results/plots/:
    {dataset}_val_accuracy_comparison.png   — val accuracy all models
    {dataset}_val_loss_comparison.png       — val loss all models
    {dataset}_train_loss_comparison.png     — training loss all models
    {dataset}_f1_comparison.png             — per-class F1 grouped bar chart
    {dataset}_auc_comparison.png            — macro AUC bar chart per model

Outputs saved to results/{dataset}/:
    model_comparison_summary.csv            — best val acc, F1, AUC, epochs

HOW TO USE
----------
Edit CONFIG and run:
    python compare_models.py

Run metrics.py for each model first so the metrics CSVs exist.
"""

import os
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
CONFIG = {
    "dataset":      "brain",   # 'brain' or 'breast'
    "project_root": ".",
}
# ─────────────────────────────────────────────

PROJECT_ROOT = os.path.abspath(CONFIG["project_root"])

MODELS = ["resnet50", "mobilenet", "vit", "swin"]

MODEL_COLORS = {
    "resnet50":  "#2196F3",
    "mobilenet": "#4CAF50",
    "vit":       "#FF9800",
    "swin":      "#E91E63",
}

CLASS_INFO = {
    "brain":  ["glioma", "meningioma", "pituitary", "no_tumor"],
    "breast": ["benign", "malignant"],
}


# ── Loaders ────────────────────────────────────────────────────────────────────
def load_history(model_name: str, dataset: str):
    path = os.path.join(PROJECT_ROOT, "results", dataset,
                        f"{model_name}_history.csv")
    if not os.path.exists(path):
        print(f"[WARNING] History not found for {model_name} — skipping.")
        return None
    return pd.read_csv(path)


def load_metrics(model_name: str, dataset: str):
    path = os.path.join(PROJECT_ROOT, "results", dataset,
                        f"{model_name}_metrics.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    return df[df["class"] != "macro avg"].set_index("class")


def load_macro_metrics(model_name: str, dataset: str):
    path = os.path.join(PROJECT_ROOT, "results", dataset,
                        f"{model_name}_metrics.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    macro = df[df["class"] == "macro avg"].iloc[0]
    return macro


# ── Training curve plots ───────────────────────────────────────────────────────
def _plot_curve(histories: dict, y_col: str, title: str,
                ylabel: str, save_path: str, legend_loc: str = "lower right"):
    fig, ax = plt.subplots(figsize=(10, 6))

    for model_name, df in histories.items():
        ax.plot(df["epoch"], df[y_col],
                label=model_name.upper(),
                color=MODEL_COLORS.get(model_name, "gray"),
                linewidth=2)

    ax.set(title=title, xlabel="Epoch", ylabel=ylabel)
    ax.legend(loc=legend_loc)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {save_path}")


# ── Per-class F1 grouped bar chart ─────────────────────────────────────────────
def plot_f1_comparison(metrics_dict: dict, dataset: str,
                       classes: list, save_path: str):
    """
    Grouped bar chart — one group per class, one bar per model.
    Makes it easy to see which model handles each class best.
    """
    models     = list(metrics_dict.keys())
    n_models   = len(models)
    n_classes  = len(classes)
    x          = np.arange(n_classes)
    width      = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(max(8, n_classes * 2.5), 6))

    for i, model_name in enumerate(models):
        df     = metrics_dict[model_name]
        scores = [df.loc[c, "f1"] if c in df.index else 0.0 for c in classes]
        offset = (i - n_models / 2 + 0.5) * width
        bars   = ax.bar(x + offset, scores, width,
                        label=model_name.upper(),
                        color=MODEL_COLORS.get(model_name, "gray"),
                        edgecolor="white", linewidth=0.5)

        for bar, score in zip(bars, scores):
            if score > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.005,
                        f"{score:.2f}",
                        ha="center", va="bottom", fontsize=7)

    ax.set(
        title=f"Per-Class F1 Score Comparison — {dataset} dataset",
        xlabel="Class",
        ylabel="F1 Score",
        xticks=x,
        xticklabels=classes,
        ylim=[0, 1.12],
    )
    ax.legend(loc="lower right")
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {save_path}")


# ── Macro AUC bar chart ────────────────────────────────────────────────────────
def plot_auc_comparison(macro_metrics: dict, dataset: str, save_path: str):
    """
    Simple bar chart of macro AUC-ROC per model.
    Only plotted if AUC column is available in the metrics CSVs.
    """
    models = list(macro_metrics.keys())
    aucs   = [macro_metrics[m]["auc"] if "auc" in macro_metrics[m].index
              else None for m in models]

    if any(a is None for a in aucs):
        print("[INFO] AUC not found in all metrics CSVs — "
              "run updated metrics.py first.")
        return

    fig, ax = plt.subplots(figsize=(max(6, len(models) * 1.8), 5))
    bars = ax.bar(
        [m.upper() for m in models], aucs,
        color=[MODEL_COLORS.get(m, "gray") for m in models],
        edgecolor="white", linewidth=0.8
    )

    for bar, auc in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.002,
                f"{auc:.4f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set(
        title=f"Macro AUC-ROC Comparison — {dataset} dataset",
        xlabel="Model",
        ylabel="Macro AUC-ROC",
        ylim=[0.9, 1.02],
    )
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {save_path}")


# ── Summary table ──────────────────────────────────────────────────────────────
def build_summary(histories: dict, macro_metrics: dict,
                  dataset: str) -> list:
    print(f"\n{'='*70}")
    print(f"  Final Results Summary — {dataset} dataset")
    print(f"{'='*70}")
    print(f"{'Model':<15} {'Best Val Acc':>13} {'Macro F1':>10} "
          f"{'Macro AUC':>11} {'Epochs':>8}")
    print("-" * 70)

    rows = []
    for model_name, df in histories.items():
        best_val = df["val_acc"].max()
        epochs   = len(df)
        macro    = macro_metrics.get(model_name)
        f1_val   = round(float(macro["f1"]),  4) if macro is not None else "-"
        auc_val  = round(float(macro["auc"]), 4) if (
            macro is not None and "auc" in macro.index) else "-"

        print(f"{model_name.upper():<15} {best_val:>13.4f} "
              f"{str(f1_val):>10} {str(auc_val):>11} {epochs:>8}")
        rows.append({
            "model":         model_name,
            "best_val_acc":  round(float(best_val), 4),
            "macro_f1":      f1_val,
            "macro_auc":     auc_val,
            "epochs_trained": epochs,
        })

    return rows


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    dataset = CONFIG["dataset"]
    classes = CLASS_INFO[dataset]

    plots_dir   = os.path.join(PROJECT_ROOT, "results", "plots")
    results_dir = os.path.join(PROJECT_ROOT, "results", dataset)
    os.makedirs(plots_dir,   exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Load available data
    histories     = {m: load_history(m, dataset)
                     for m in MODELS if load_history(m, dataset) is not None}
    metrics_dict  = {m: load_metrics(m, dataset)
                     for m in MODELS if load_metrics(m, dataset) is not None}
    macro_metrics = {m: load_macro_metrics(m, dataset)
                     for m in MODELS
                     if load_macro_metrics(m, dataset) is not None}

    if not histories:
        print("No training histories found. Train the models first.")
        return

    # ── Training curves ────────────────────────────────────────────────────────
    _plot_curve(
        histories, "val_acc",
        title=f"Validation Accuracy — All Models ({dataset} dataset)",
        ylabel="Validation Accuracy",
        save_path=os.path.join(plots_dir,
                               f"{dataset}_val_accuracy_comparison.png"),
        legend_loc="lower right",
    )
    _plot_curve(
        histories, "val_loss",
        title=f"Validation Loss — All Models ({dataset} dataset)",
        ylabel="Validation Loss",
        save_path=os.path.join(plots_dir,
                               f"{dataset}_val_loss_comparison.png"),
        legend_loc="upper right",
    )
    _plot_curve(
        histories, "train_loss",
        title=f"Training Loss — All Models ({dataset} dataset)",
        ylabel="Training Loss",
        save_path=os.path.join(plots_dir,
                               f"{dataset}_train_loss_comparison.png"),
        legend_loc="upper right",
    )

    # ── F1 comparison ──────────────────────────────────────────────────────────
    if metrics_dict:
        plot_f1_comparison(
            metrics_dict, dataset, classes,
            save_path=os.path.join(plots_dir,
                                   f"{dataset}_f1_comparison.png")
        )

    # ── AUC comparison ─────────────────────────────────────────────────────────
    if macro_metrics:
        plot_auc_comparison(
            macro_metrics, dataset,
            save_path=os.path.join(plots_dir,
                                   f"{dataset}_auc_comparison.png")
        )

    # ── Summary table + CSV ────────────────────────────────────────────────────
    rows = build_summary(histories, macro_metrics, dataset)
    csv_path = os.path.join(results_dir, "model_comparison_summary.csv")
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSummary CSV saved → {csv_path}")


if __name__ == "__main__":
    main()