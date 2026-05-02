"""
visualize_augmentation.py
--------------------------
Generates a figure showing original vs augmented training images
for both datasets (brain MRI and breast histopathology).

Intended for inclusion in the project report to demonstrate data augmentation in training pipeline.

"""

import os
import random
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from PIL import Image
from torchvision import transforms

#  CONFIG
CONFIG = {
    # Root of the project — change if running from a different directory
    # On Colab:  "/content/drive/MyDrive/oncostream-cancer-detection"
    "project_root": ".",

    # Number of sample images to show per dataset
    "num_samples": 4,

    # Number of augmented versions to show per original image
    "num_augmented": 3,

    # Random seed — set to None for different samples every run
    "seed": None,
}

PROJECT_ROOT = os.path.abspath(CONFIG["project_root"])
PLOTS_DIR    = os.path.join(PROJECT_ROOT, "results", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


#  Augmentation pipeline
augmentation_pipeline = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
])

resize_only = transforms.Resize((224, 224))

AUGMENTATION_STEPS = [
    "Resize → 224×224",
    "Random Horizontal Flip (p=0.5)",
    "Random Rotation (±15°)",
    "Color Jitter (brightness, contrast, saturation)",
]

#  Utility

def collect_image_paths(dataset_folder: str) -> list:
    valid_exts = {".jpg", ".jpeg", ".png"}
    paths = []
    for root, _, files in os.walk(dataset_folder):
        for fname in files:
            if os.path.splitext(fname)[1].lower() in valid_exts:
                paths.append(os.path.join(root, fname))
    return paths

def build_figure(samples_brain: list, samples_breast: list, num_augmented: int) -> plt.Figure:
    """
    Layout:
        Each row = one sample image.
        Column 0 = original (resized).
        Columns 1..num_augmented = independently augmented versions.
        Brain samples are in the top half, breast in the bottom half.
        A coloured section label separates them.
    """
    n_brain  = len(samples_brain)
    n_breast = len(samples_breast)
    n_rows   = n_brain + n_breast
    n_cols   = 1 + num_augmented          # original + augmented

    fig_width  = n_cols * 2.4
    fig_height = n_rows * 2.4 + 1.2 

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))

    if n_rows == 1:
        axes = axes[None, :]

    def render_row(ax_row, img_path, col_labels):
        original = Image.open(img_path).convert("RGB")
        resized  = resize_only(original)

        # Column 0 — original
        ax_row[0].imshow(resized)
        ax_row[0].set_title("Original", fontsize=8, fontweight="bold")
        ax_row[0].axis("off")

        # Sub-label: class name from parent folder
        class_name = os.path.basename(os.path.dirname(img_path))
        ax_row[0].set_xlabel(class_name, fontsize=7, color="#555555",
                             labelpad=2)

        # Columns 1..n — augmented (each independently seeded)
        for col_idx in range(1, n_cols):
            aug = augmentation_pipeline(original)
            ax_row[col_idx].imshow(aug)
            ax_row[col_idx].set_title(f"Aug {col_idx}", fontsize=8)
            ax_row[col_idx].axis("off")

    # Brain rows 
    for i, path in enumerate(samples_brain):
        render_row(axes[i], path, [])

    # Breast rows 
    for i, path in enumerate(samples_breast):
        render_row(axes[n_brain + i], path, [])

    # Section labels (drawn as text on the figure, not on axes) 
    brain_y_top  = 1.0 - (0 / n_rows)
    breast_y_top = 1.0 - (n_brain / n_rows)

    fig.text(0.01, brain_y_top  - 0.01,
             " Brain Tumor MRI (4 classes)",
             fontsize=9, fontweight="bold", color="#1a5276",
             va="top", transform=fig.transFigure)
    fig.text(0.01, breast_y_top - 0.01,
             " Breast Histopathology (2 classes)",
             fontsize=9, fontweight="bold", color="#6e2f1a",
             va="top", transform=fig.transFigure)

    #  Legend for augmentation steps ─
    legend_text = "  Augmentation pipeline:  " + "  →  ".join(AUGMENTATION_STEPS)
    fig.text(0.5, 0.01, legend_text,
             ha="center", va="bottom", fontsize=7.5, color="#333333",
             style="italic", transform=fig.transFigure,
             wrap=True)

    fig.suptitle("Data Augmentation Examples — OncoStream",
                 fontsize=13, fontweight="bold", y=1.01)

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    return fig


#  Main
def main():
    if CONFIG["seed"] is not None:
        random.seed(CONFIG["seed"])

    num_samples  = CONFIG["num_samples"]
    num_aug      = CONFIG["num_augmented"]

    brain_root  = os.path.join(PROJECT_ROOT, "datasets", "brain")
    breast_root = os.path.join(PROJECT_ROOT, "datasets", "breast")

    for folder, name in [(brain_root, "brain"), (breast_root, "breast")]:
        if not os.path.isdir(folder):
            print(f"[WARNING] Dataset folder not found: {folder}")
            print(f"          Download the {name} dataset from Kaggle and place it under datasets/{name}/")

    #  Collect image paths 
    brain_paths  = collect_image_paths(brain_root)  if os.path.isdir(brain_root)  else []
    breast_paths = collect_image_paths(breast_root) if os.path.isdir(breast_root) else []

    if not brain_paths and not breast_paths:
        print("\nNo images found in either dataset folder. "
              "Please download the datasets first.")
        return

    # Sample randomly — different images on each run (unless seed is set)
    samples_brain  = random.sample(brain_paths,  min(num_samples, len(brain_paths)))
    samples_breast = random.sample(breast_paths, min(num_samples, len(breast_paths)))

    print(f"Sampled {len(samples_brain)} brain images and "
          f"{len(samples_breast)} breast images.")

    #  Build and save figure 
    fig = build_figure(samples_brain, samples_breast, num_aug)

    out_path = os.path.join(PLOTS_DIR, "augmentation_examples.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nFigure saved → {out_path}")
    print("Include this image in your paper under the Methodology / "
          "Data Augmentation section.")

if __name__ == "__main__":
    main()
