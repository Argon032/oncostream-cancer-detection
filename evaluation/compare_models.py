"""
compare_models.py
-----------------
Loads training history CSVs for all four models and generates:
  1. A val accuracy comparison plot (all models on one chart)
  2. A final results summary table (printed + saved as CSV)

Run this after all four models have been trained on a dataset.

"""

import os
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
CONFIG = {
    "dataset":      "breast",    # 'brain' or 'breast'
    "project_root": ".",
}
# ─────────────────────────────────────────────

PROJECT_ROOT = os.path.abspath(CONFIG["project_root"])

MODELS = ["resnet50", "mobilenet", "vit", "swin"]

# Colours for each model line in the plot
MODEL_COLORS = {
    "resnet50": "#2196F3",   # blue
    "mobilenet": "#4CAF50",  # green
    "vit":      "#FF9800",   # orange
    "swin":     "#E91E63",   # pink/red — Swin stands out as primary model
}


def load_history(model_name: str, dataset: str) -> pd.DataFrame | None:
    csv_path = os.path.join(PROJECT_ROOT, "results", dataset,
                            f"{model_name}_history.csv")
    if not os.path.exists(csv_path):
        print(f"[WARNING] History not found for {model_name} — skipping.")
        return None
    return pd.read_csv(csv_path)


def plot_val_accuracy(histories: dict, dataset: str, save_path: str):
    fig, ax = plt.subplots(figsize=(10, 6))

    for model_name, df in histories.items():
        ax.plot(df["epoch"], df["val_acc"],
                label=model_name.upper(),
                color=MODEL_COLORS.get(model_name, "gray"),
                linewidth=2.5 if model_name == "swin" else 1.5,
                linestyle="-")

    ax.set(
        title=f"Validation Accuracy — All Models ({dataset} dataset)",
        xlabel="Epoch",
        ylabel="Validation Accuracy",
    )
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Val accuracy plot saved → {save_path}")


def plot_val_loss(histories: dict, dataset: str, save_path: str):
    fig, ax = plt.subplots(figsize=(10, 6))

    for model_name, df in histories.items():
        ax.plot(df["epoch"], df["val_loss"],
                label=model_name.upper(),
                color=MODEL_COLORS.get(model_name, "gray"),
                linewidth=2.5 if model_name == "swin" else 1.5)

    ax.set(
        title=f"Validation Loss — All Models ({dataset} dataset)",
        xlabel="Epoch",
        ylabel="Validation Loss",
    )
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Val loss plot saved → {save_path}")


def print_summary_table(histories: dict, dataset: str) -> list:
    print(f"\n{'='*60}")
    print(f"  Final Results Summary — {dataset} dataset")
    print(f"{'='*60}")
    print(f"{'Model':<15} {'Best Val Acc':>14} {'Final Train Acc':>16} {'Epochs':>8}")
    print("-" * 60)

    rows = []
    for model_name, df in histories.items():
        best_val  = df["val_acc"].max()
        final_trn = df["train_acc"].iloc[-1]
        epochs    = len(df)

        print(f"{model_name.upper():<15} {best_val:>14.4f} {final_trn:>16.4f} {epochs:>8}")
        rows.append({
            "model":           model_name,
            "best_val_acc":    round(float(best_val),  4),
            "final_train_acc": round(float(final_trn), 4),
            "epochs_trained":  epochs,
        })

    return rows


def main():
    dataset = CONFIG["dataset"]

    # Load all available histories
    histories = {}
    for model_name in MODELS:
        df = load_history(model_name, dataset)
        if df is not None:
            histories[model_name] = df

    if not histories:
        print("No training histories found. Train the models first.")
        return

    # Plots
    plots_dir = os.path.join(PROJECT_ROOT, "results", "plots")
    os.makedirs(plots_dir, exist_ok=True)

    plot_val_accuracy(
        histories, dataset,
        save_path=os.path.join(plots_dir, f"{dataset}_val_accuracy_comparison.png")
    )
    plot_val_loss(
        histories, dataset,
        save_path=os.path.join(plots_dir, f"{dataset}_val_loss_comparison.png")
    )

    # Summary table
    rows = print_summary_table(histories, dataset)

    # Save summary CSV
    results_dir = os.path.join(PROJECT_ROOT, "results", dataset)
    os.makedirs(results_dir, exist_ok=True)
    summary_path = os.path.join(results_dir, "model_comparison_summary.csv")

    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSummary CSV saved → {summary_path}")


if __name__ == "__main__":
    main()
