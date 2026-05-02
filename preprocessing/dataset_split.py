import os
import shutil
import random

def split_dataset(raw_path, output_path, train_ratio=0.7, val_ratio=0.15):
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(output_path, split), exist_ok=True)

    classes = os.listdir(raw_path)

    for c in classes:
        class_path = os.path.join(raw_path, c)
        if not os.path.isdir(class_path):
            continue

        files = os.listdir(class_path)
        random.shuffle(files)

        train_split = int(train_ratio * len(files))
        val_split = int((train_ratio + val_ratio) * len(files))

        splits = {
            "train": files[:train_split],
            "val": files[train_split:val_split],
            "test": files[val_split:]
        }

        for split in splits:
            split_class_path = os.path.join(output_path, split, c)
            os.makedirs(split_class_path, exist_ok=True)

            for f in splits[split]:
                src = os.path.join(class_path, f)
                dst = os.path.join(split_class_path, f)
                shutil.copy(src, dst)

        print(f"{c}: {len(files)} images split")

    print("Full dataset split completed!")

def create_val_split(dataset_path, val_ratio=0.2):
    train_path = os.path.join(dataset_path, "train")
    val_path = os.path.join(dataset_path, "val")

    if not os.path.exists(train_path):
        print("Train folder not found!")
        return

    if os.path.exists(val_path) and len(os.listdir(val_path)) > 0:
        print("Validation already exists. Skipping.")
        return

    os.makedirs(val_path, exist_ok=True)

    classes = os.listdir(train_path)

    for c in classes:
        class_train_path = os.path.join(train_path, c)
        class_val_path = os.path.join(val_path, c)

        if not os.path.isdir(class_train_path):
            continue

        os.makedirs(class_val_path, exist_ok=True)

        files = [
            f for f in os.listdir(class_train_path)
            if os.path.isfile(os.path.join(class_train_path, f))
        ]

        random.shuffle(files)

        val_count = int(val_ratio * len(files))
        val_files = files[:val_count]

        for f in val_files:
            src = os.path.join(class_train_path, f)
            dst = os.path.join(class_val_path, f)
            shutil.move(src, dst)

        print(f"{c}: moved {len(val_files)} images to val")

    print("Validation split created!")

# Breast dataset (raw → split)
if os.path.exists("datasets/breast/raw"):
    split_dataset("datasets/breast/raw", "datasets/breast")

# Brain dataset (train → val)
if os.path.exists("datasets/brain"):
    create_val_split("datasets/brain")