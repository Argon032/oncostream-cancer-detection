import os
import shutil
import random

dataset = "breast_dataset"

train_path = "breast_dataset/train"
val_path = "breast_dataset/val"
test_path = "breast_dataset/test"

classes = ["benign", "malignant"]

for c in classes:
    os.makedirs(os.path.join(train_path, c), exist_ok=True)
    os.makedirs(os.path.join(val_path, c), exist_ok=True)
    os.makedirs(os.path.join(test_path, c), exist_ok=True)

    files = os.listdir(os.path.join(dataset, c))
    random.shuffle(files)

    train_split = int(0.7 * len(files))
    val_split = int(0.85 * len(files))

    train_files = files[:train_split]
    val_files = files[train_split:val_split]
    test_files = files[val_split:]

    for f in train_files:
        shutil.copy(os.path.join(dataset, c, f), os.path.join(train_path, c, f))

    for f in val_files:
        shutil.copy(os.path.join(dataset, c, f), os.path.join(val_path, c, f))

    for f in test_files:
        shutil.copy(os.path.join(dataset, c, f), os.path.join(test_path, c, f))

print("Dataset split completed!")