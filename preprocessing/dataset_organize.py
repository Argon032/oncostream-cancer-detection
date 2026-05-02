import os
import shutil
import random
from collections import defaultdict

source = "datasets/breast/BreaKHis_v1/histology_slides/breast"
output_base = "datasets/breast/raw"

benign_path = os.path.join(output_base, "benign")
malignant_path = os.path.join(output_base, "malignant")

os.makedirs(benign_path, exist_ok=True)
os.makedirs(malignant_path, exist_ok=True)

random.seed(42)

# Images grouped by (class, subtype, magnification)
groups = defaultdict(list)

for root, dirs, files in os.walk(source):
    for file in files:
        if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        path = os.path.join(root, file)
        parts = root.lower().split(os.sep)

        label = None
        subtype = None
        mag = None

        # detect class
        if "benign" in parts:
            label = "benign"
        elif "malignant" in parts:
            label = "malignant"

        # detect subtype 
        for p in parts:
            if p in ["adenosis", "fibroadenoma", "tubular_adenoma", "phyllodes_tumor",
                     "ductal", "lobular", "mucinous", "papillary"]:
                subtype = p

        # detect magnification
        for p in parts:
            if p in ["40X", "100X", "200X", "400X"]:
                mag = p

        if label and subtype and mag:
            groups[(label, subtype, mag)].append(path)

# Sampling size
min_group_size = min(len(v) for v in groups.values())
samples_per_group = min(min_group_size, 100)  # 100 per group (adjust if needed)

print("Sampling", samples_per_group, "per group")

# Sample and copy
benign_count = 0
malignant_count = 0

for (label, subtype, mag), paths in groups.items():
    selected = random.sample(paths, samples_per_group)

    for i, src in enumerate(selected):
        filename = f"{label}_{subtype}_{mag}_{i}.png"

        if label == "benign":
            dst = os.path.join(benign_path, filename)
            shutil.copy(src, dst)
            benign_count += 1
        else:
            dst = os.path.join(malignant_path, filename)
            shutil.copy(src, dst)
            malignant_count += 1

print("Done!")
print("Benign:", benign_count)
print("Malignant:", malignant_count)