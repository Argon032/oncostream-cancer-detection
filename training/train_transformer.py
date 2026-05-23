"""
train_transformer.py
--------------------
Training script for transformer-based models: Swin-T and ViT-B/16.

HOW TO USE
----------
Edit the CONFIG block below, then run:
    python train_transformer.py

On Google Colab / Kaggle, mount your drive / attach dataset, update
project_root in CONFIG, and run the cell.

EXPECTED DATASET STRUCTURE
---------------------------
datasets/
  brain/
    train/ val/ test/
      glioma/ meningioma/ pituitary/ no_tumor/
  breast/
    train/ val/ test/
      benign/ malignant/

(Produced by dataset_split.py)

CHECKPOINT FORMAT (for evaluation scripts)
-------------------------------------------
Saved to: results/{dataset}/{model_name}_best.pth

Load with:
    checkpoint = torch.load("swin_best.pth")
    model.load_state_dict(checkpoint["model_state_dict"])

Available keys in checkpoint:
    checkpoint["epoch"]            → epoch at which best val_acc was achieved
    checkpoint["model_state_dict"] → model weights
    checkpoint["val_acc"]          → best validation accuracy
    checkpoint["num_classes"]      → int (4 for brain, 2 for breast)
    checkpoint["class_labels"]     → list e.g. ["glioma", "meningioma", ...]

WHY A SEPARATE SCRIPT FROM train_cnn.py?
-----------------------------------------
Transformers need a different training recipe than CNNs:
  1. AdamW optimiser — weight decay applied correctly for attention layers.
  2. Differential learning rates — Swin backbone gets 10x lower LR than head.
  3. Warmup + cosine annealing — transformers are sensitive to early updates.
  4. Gradient clipping — prevents instability in attention layers.
  5. Optional class weights — addresses class imbalance (important for Swin
     on brain dataset where one class was over-predicted).
"""

import os
import sys
import csv
import time
import copy
import math

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets

# ─────────────────────────────────────────────
#  CONFIG — edit this block before running
# ─────────────────────────────────────────────
CONFIG = {
    # Model: 'swin' or 'vit'
    "model_name": "swin",

    # Dataset: 'brain' or 'breast'
    "dataset": "brain",

    # Root folder of the project
    "project_root": ".",

    "freeze_stages": False,

    # Weight the loss function by inverse class frequency.
    "use_class_weights": True,

    # Training hyperparameters
    "epochs":     20,           
    "batch_size": 8,

    "head_lr":     1e-4,
    "backbone_lr": 1e-5,
    "weight_decay": 0.05,

    "warmup_epochs": 3,

    "patience": 6,

    "num_workers": 2,
}

# ── Resolve paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(CONFIG["project_root"])
DATASET_PATH = os.path.join(PROJECT_ROOT, "datasets", CONFIG["dataset"])
RESULTS_PATH = os.path.join(PROJECT_ROOT, "results",  CONFIG["dataset"])
os.makedirs(RESULTS_PATH, exist_ok=True)

# ── Class labels ───────────────────────────────────────────────────────────────
CLASS_INFO = {
    "brain":  {"num_classes": 4,
               "labels": ["glioma", "meningioma", "pituitary", "no_tumor"]},
    "breast": {"num_classes": 2,
               "labels": ["benign", "malignant"]},
}

# ── Device ─────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ── Import shared dataloader ───────────────────────────────────────────────────
sys.path.insert(0, os.path.join(PROJECT_ROOT, "preprocessing"))
from dataloader import get_dataloaders

#  Class weight computation
def compute_class_weights(dataset_path: str, num_workers: int) -> torch.Tensor:
    """
    Compute inverse-frequency class weights from the training set.
    Returns a tensor of shape (num_classes,) on DEVICE.

    These weights are passed to CrossEntropyLoss so underrepresented
    or frequently mis-predicted classes receive higher gradient signal.
    """
    from torchvision import datasets as tvdatasets
    from torchvision import transforms
    from sklearn.utils.class_weight import compute_class_weight

    # Load train set without augmentation — we only need labels
    plain = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    train_ds = tvdatasets.ImageFolder(
        root=os.path.join(dataset_path, "train"),
        transform=plain
    )

    labels = [label for _, label in train_ds.samples]
    classes = np.unique(labels)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=labels
    )

    print(f"Class weights: { {train_ds.classes[i]: round(w, 4) for i, w in enumerate(weights)} }")
    return torch.FloatTensor(weights).to(DEVICE)


#  Model loading
def load_model(model_name: str, num_classes: int, cfg: dict):
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "models", "transformer"))

    if model_name == "swin":
        from swin import get_model, get_param_groups
        model = get_model(
            num_classes=num_classes,
            freeze_stages=cfg["freeze_stages"]
        ).to(DEVICE)

        param_groups = get_param_groups(
            model,
            head_lr=cfg["head_lr"],
            backbone_lr=cfg["backbone_lr"]
        )
        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=cfg["weight_decay"]
        )

    elif model_name == "vit":
        from vit import get_model
        model = get_model(num_classes=num_classes).to(DEVICE)
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg["head_lr"],
            weight_decay=cfg["weight_decay"]
        )

    else:
        raise ValueError(f"Unknown model: '{model_name}'. Choose 'swin' or 'vit'.")

    return model, optimizer


#  Warmup + Cosine Annealing scheduler (step-level)
def get_scheduler(optimizer, warmup_epochs: int, total_epochs: int,
                  steps_per_epoch: int):
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps  = total_epochs  * steps_per_epoch

    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


#  Training loop (one epoch)
def train_one_epoch(model, loader, criterion, optimizer, scheduler):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        running_loss += loss.item() * images.size(0)
        preds         = outputs.argmax(dim=1)
        correct      += (preds == labels).sum().item()
        total        += labels.size(0)

    return running_loss / total, correct / total


#  Evaluation loop
@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        loss    = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        preds         = outputs.argmax(dim=1)
        correct      += (preds == labels).sum().item()
        total        += labels.size(0)

    return running_loss / total, correct / total


#  Main training routine
def main():
    cfg        = CONFIG
    model_name = cfg["model_name"]
    dataset    = cfg["dataset"]
    info       = CLASS_INFO[dataset]

    print(f"\n{'='*60}")
    print(f"  Model        : {model_name.upper()}")
    print(f"  Dataset      : {dataset}")
    print(f"  Classes      : {info['num_classes']} → {info['labels']}")
    print(f"  Epochs       : {cfg['epochs']}   Warmup: {cfg['warmup_epochs']}")
    print(f"  Head LR      : {cfg['head_lr']}   Backbone LR: {cfg['backbone_lr']}")
    if model_name == "swin":
        print(f"  Freeze stages: {cfg['freeze_stages']}")
        print(f"  Class weights: {cfg['use_class_weights']}")
    print(f"{'='*60}\n")

    # Data
    train_loader, val_loader, test_loader = get_dataloaders(
        DATASET_PATH, cfg["batch_size"], cfg["num_workers"]
    )

    # Loss — with or without class weights
    if model_name == "swin" and cfg["use_class_weights"]:
        weights   = compute_class_weights(DATASET_PATH, cfg["num_workers"])
        criterion = nn.CrossEntropyLoss(weight=weights)
        print("Using weighted CrossEntropyLoss.\n")
    else:
        criterion = nn.CrossEntropyLoss()

    # Model + optimiser
    model, optimizer = load_model(model_name, info["num_classes"], cfg)

    # Scheduler
    scheduler = get_scheduler(
        optimizer,
        warmup_epochs=cfg["warmup_epochs"],
        total_epochs=cfg["epochs"],
        steps_per_epoch=len(train_loader)
    )

    # ── Training loop ──────────────────────────────────────────────────────────
    best_val_acc      = 0.0
    best_model_wts    = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0
    history           = []

    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader,
                                                criterion, optimizer, scheduler)
        val_loss,   val_acc   = evaluate(model, val_loader, criterion)
        elapsed = time.time() - t0

        current_lr = optimizer.param_groups[-1]["lr"]
        print(f"Epoch [{epoch:02d}/{cfg['epochs']}]  "
              f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.4f}  |  "
              f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.4f}  "
              f"LR: {current_lr:.2e}  ({elapsed:.1f}s)")

        history.append({
            "epoch":      epoch,
            "train_loss": round(train_loss, 6),
            "train_acc":  round(train_acc,  6),
            "val_loss":   round(val_loss,   6),
            "val_acc":    round(val_acc,    6),
            "lr":         round(current_lr, 8),
        })

        if val_acc > best_val_acc:
            best_val_acc      = val_acc
            best_model_wts    = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0

            ckpt_path = os.path.join(RESULTS_PATH, f"{model_name}_best.pth")
            torch.save({
                "epoch":            epoch,
                "model_state_dict": best_model_wts,
                "val_acc":          best_val_acc,
                "num_classes":      info["num_classes"],
                "class_labels":     info["labels"],
            }, ckpt_path)
            print(f"  ✓ Best model saved (val_acc={best_val_acc:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg["patience"]:
                print(f"\nEarly stopping after {epoch} epochs "
                      f"(no improvement for {cfg['patience']} epochs).")
                break

    # ── Save history ───────────────────────────────────────────────────────────
    csv_path = os.path.join(RESULTS_PATH, f"{model_name}_history.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    print(f"\nTraining history saved → {csv_path}")

    model.load_state_dict(best_model_wts)
    test_loss, test_acc = evaluate(model, test_loader, criterion)
    print(f"\nTest Results  →  Loss: {test_loss:.4f}  |  Accuracy: {test_acc:.4f}")
    print(f"Best Val Accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()