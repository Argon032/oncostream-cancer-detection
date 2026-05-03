import os
import shutil
import random

source = "datasets/breast/BreaKHis_v1/histology_slides/breast"
output = "datasets/breast/raw"

benign_out = os.path.join(output, "benign")
malignant_out = os.path.join(output, "malignant")

os.makedirs(benign_out, exist_ok=True)
os.makedirs(malignant_out, exist_ok=True)

benign = []
malignant = []

for root, _, files in os.walk(source):
    for file in files:
        if not file.lower().endswith(".png"):
            continue

        full_path = os.path.join(root, file)

        if "benign" in root.lower():
            benign.append(full_path)

        elif "malignant" in root.lower():
            malignant.append(full_path)

print("Found benign:", len(benign))
print("Found malignant:", len(malignant))

if len(benign) == 0 or len(malignant) == 0:
    print("Something is wrong with path.")
    exit()

n = min(len(benign), len(malignant), 2000)

benign_sample = random.sample(benign, n)
malignant_sample = random.sample(malignant, n)

# 🔹 Step 3: copy
for i, img in enumerate(benign_sample):
    shutil.copy(img, os.path.join(benign_out, f"benign_{i}.png"))

for i, img in enumerate(malignant_sample):
    shutil.copy(img, os.path.join(malignant_out, f"malignant_{i}.png"))

print("Done. Final count per class:", n)