import os
import shutil

source = "archive/IDC_regular_ps50_idx5"
benign = "breast_dataset/benign"
malignant = "breast_dataset/malignant"

os.makedirs(benign, exist_ok=True)
os.makedirs(malignant, exist_ok=True)

benign_count = 0
malignant_count = 0
limit = 2000

for folder in os.listdir(source):
    patient_path = os.path.join(source, folder)

    zero_folder = os.path.join(patient_path, "0")
    one_folder = os.path.join(patient_path, "1")

    # Copy benign images
    if os.path.exists(zero_folder):
        for img in os.listdir(zero_folder):
            if benign_count >= limit:
                break
            src = os.path.join(zero_folder, img)
            dst = os.path.join(benign, img)
            shutil.copy(src, dst)
            benign_count += 1

    # Copy malignant images
    if os.path.exists(one_folder):
        for img in os.listdir(one_folder):
            if malignant_count >= limit:
                break
            src = os.path.join(one_folder, img)
            dst = os.path.join(malignant, img)
            shutil.copy(src, dst)
            malignant_count += 1

    if benign_count >= limit and malignant_count >= limit:
        break

print("Dataset organized successfully!")
print("Benign images:", benign_count)
print("Malignant images:", malignant_count)