"""
train_cnn.py
------------
Training script for CNN-based models: ResNet-50 and MobileNetV2.

HOW TO USE
----------
Edit the CONFIG block below, then run:
    python train_cnn.py

On Google Colab / Kaggle, mount your drive / attach dataset, update
DATASET_ROOT in CONFIG, and run the cell.

EXPECTED DATASET STRUCTURE
---------------------------
datasets/
  brain/
    train/
      glioma/ meningioma/ pituitary/ no_tumor/
    val/
      glioma/ meningioma/ pituitary/ no_tumor/
    test/
      glioma/ meningioma/ pituitary/ no_tumor/
  breast/
    train/
      benign/ malignant/
    val/
      benign/ malignant/
    test/
      benign/ malignant/

(This structure is produced by dataset_split.py)

CHECKPOINT FORMAT (for evaluation scripts)
-------------------------------------------
Saved to: results/{dataset}/{model_name}_best.pth
 
Load with:
    checkpoint = torch.load("resnet50_best.pth")
    model.load_state_dict(checkpoint["model_state_dict"])
 
Available keys in checkpoint:
    checkpoint["epoch"]          → epoch at which best val_acc was achieved
    checkpoint["model_state_dict"] → model weights
    checkpoint["val_acc"]        → best validation accuracy
    checkpoint["num_classes"]    → int (4 for brain, 2 for breast)
    checkpoint["class_labels"]   → list e.g. ["glioma", "meningioma", ...]

"""

import os
import sys
import csv
import time
import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# CONFIG — edit this block before running
CONFIG = {
    # Model: 'resnet50' or 'mobilenet'
    "model_name": "resnet50",

    # Dataset: 'brain' or 'breast'
    "dataset": "brain",

    # Root folder of the project (contains datasets/, results/, models/)
    # On Colab:  "/content/drive/MyDrive/oncostream-cancer-detection"
    "project_root": ".",

    # Training hyperparameters
    "epochs": 25,
    "batch_size": 32,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,

    # How many epochs without val improvement before stopping early
    "patience": 6,

    # Number of CPU workers for data loading (0 = main process, safe on Colab)
    "num_workers": 2,
}


# Resolve paths 
PROJECT_ROOT = os.path.abspath(CONFIG["project_root"])
DATASET_PATH = os.path.join(PROJECT_ROOT, "datasets", CONFIG["dataset"])
RESULTS_PATH = os.path.join(PROJECT_ROOT, "results", CONFIG["dataset"])
os.makedirs(RESULTS_PATH, exist_ok=True)

# Class labels per dataset ─
CLASS_INFO = {
    "brain":  {"num_classes": 4,
               "labels": ["glioma", "meningioma", "pituitary", "no_tumor"]},
    "breast": {"num_classes": 2,
               "labels": ["benign", "malignant"]},
}

# Device ─
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# Data transforms
    # ImageNet normalisation stats (standard for pretrained torchvision models)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

val_test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


#  Data loading
def get_dataloaders(dataset_path: str, batch_size: int, num_workers: int):
    train_dataset = datasets.ImageFolder(
        root=os.path.join(dataset_path, "train"),
        transform=train_transforms
    )
    val_dataset = datasets.ImageFolder(
        root=os.path.join(dataset_path, "val"),
        transform=val_test_transforms
    )
    test_dataset = datasets.ImageFolder(
        root=os.path.join(dataset_path, "test"),
        transform=val_test_transforms
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers,
                              pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=True)

    print(f"Train: {len(train_dataset)} images | "
          f"Val: {len(val_dataset)} images | "
          f"Test: {len(test_dataset)} images")
    print(f"Classes: {train_dataset.classes}")
    return train_loader, val_loader, test_loader


#  Model loading
def load_model(model_name: str, num_classes: int) -> nn.Module:
    # Dynamically import from the models/ directory
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "models", "cnn"))

    if model_name == "resnet50":
        from models.cnn.resnet50 import get_model
    elif model_name == "mobilenet":
        from models.cnn.mobilenet import get_model
    else:
        raise ValueError(f"Unknown CNN model: '{model_name}'. "
                         f"Choose 'resnet50' or 'mobilenet'.")

    model = get_model(num_classes=num_classes)
    return model.to(DEVICE)


#  Training loop (one epoch)
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc  = correct / total
    return epoch_loss, epoch_acc


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
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

    return running_loss / total, correct / total


#  Main training routine
def main():
    cfg        = CONFIG
    model_name = cfg["model_name"]
    dataset    = cfg["dataset"]
    info       = CLASS_INFO[dataset]

    print(f"\n{'='*60}")
    print(f"  Model   : {model_name.upper()}")
    print(f"  Dataset : {dataset}")
    print(f"  Classes : {info['num_classes']} → {info['labels']}")
    print(f"  Epochs  : {cfg['epochs']}   LR: {cfg['learning_rate']}")
    print(f"{'='*60}\n")

    # Data
    train_loader, val_loader, test_loader = get_dataloaders(
        DATASET_PATH, cfg["batch_size"], cfg["num_workers"]
    )

    # Model
    model = load_model(model_name, info["num_classes"])

    # Loss & optimiser
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"]
    )

    # LR scheduler: reduce on plateau
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3, verbose=True
    )

    #  Training loop 
    best_val_acc   = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0
    history = []

    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader,
                                                criterion, optimizer)
        val_loss,   val_acc   = evaluate(model, val_loader, criterion)

        scheduler.step(val_acc)
        elapsed = time.time() - t0

        print(f"Epoch [{epoch:02d}/{cfg['epochs']}]  "
              f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.4f}  |  "
              f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.4f}  "
              f"({elapsed:.1f}s)")

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_acc":  round(train_acc,  6),
            "val_loss":   round(val_loss,   6),
            "val_acc":    round(val_acc,    6),
        })

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc   = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0

            ckpt_path = os.path.join(
                RESULTS_PATH, f"{model_name}_best.pth"
            )
            torch.save({
                "epoch":      epoch,
                "model_state_dict": best_model_wts,
                "val_acc":    best_val_acc,
                "num_classes": info["num_classes"],
                "class_labels": info["labels"],
            }, ckpt_path)
            print(f"  ✓ Best model saved (val_acc={best_val_acc:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg["patience"]:
                print(f"\nEarly stopping after {epoch} epochs "
                      f"(no improvement for {cfg['patience']} epochs).")
                break

    #  Save training history 
    csv_path = os.path.join(RESULTS_PATH, f"{model_name}_history.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    print(f"\nTraining history saved → {csv_path}")

    #  Final test evaluation 
    model.load_state_dict(best_model_wts)
    test_loss, test_acc = evaluate(model, test_loader, criterion)
    print(f"\nTest Results  →  Loss: {test_loss:.4f}  |  Accuracy: {test_acc:.4f}")
    print(f"Best Val Accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
