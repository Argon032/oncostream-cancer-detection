import os
from PIL import Image

dataset = "breast_dataset"

for category in ["benign", "malignant"]:
    folder = os.path.join(dataset, category)

    for file in os.listdir(folder):
        path = os.path.join(folder, file)

        try:
            img = Image.open(path)
            img = img.resize((224, 224))
            img = img.convert("RGB")
            img.save(path)

        except:
            print("Removed broken image:", path)
            os.remove(path)

print("Preprocessing completed")