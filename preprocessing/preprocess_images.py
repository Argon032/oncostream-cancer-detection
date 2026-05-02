import os
from PIL import Image

def preprocess(dataset_path):
    for split in ["train", "val", "test"]:
        split_path = os.path.join(dataset_path, split)

        if not os.path.exists(split_path):
            continue

        for category in os.listdir(split_path):
            folder = os.path.join(split_path, category)

            if not os.path.isdir(folder):
                continue

            for file in os.listdir(folder):
                path = os.path.join(folder, file)

                try:
                    img = Image.open(path)
                    img = img.resize((224, 224))
                    img = img.convert("RGB")
                    img.save(path)
                except:
                    os.remove(path)

    print(f"Preprocessing done for {dataset_path}")


preprocess("datasets/breast")
preprocess("datasets/brain")