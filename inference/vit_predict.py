"""
vit_predict.py
--------------
Inference script for the Vision Transformer (ViT-B/16) model.
Handles both brain (4 classes) and breast (2 classes) datasets.

Usage (standalone):
    from inference.vit_predict import predict
    from PIL import Image

    image = Image.open("scan.jpg")
    pred_class, confidence, all_probs = predict(image, dataset="brain")
"""

import os
import sys
import torch
from torchvision import transforms
from PIL import Image

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

CLASS_INFO = {
    "brain":  {"num_classes": 4,
               "labels": ["glioma", "meningioma", "no_tumor", "pituitary"]},
    "breast": {"num_classes": 2,
               "labels": ["benign", "malignant"]},
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(dataset: str, project_root: str = "."):
    sys.path.insert(0, os.path.join(project_root, "models", "transformer"))
    from vit import get_model

    num_classes = CLASS_INFO[dataset]["num_classes"]
    model = get_model(num_classes=num_classes)

    ckpt_path = os.path.join(project_root, "results", dataset, "vit_best.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found at {ckpt_path}. "
            f"Run train_transformer.py with model_name='vit' and dataset='{dataset}' first."
        )

    checkpoint = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    return model


def predict(image, dataset: str, project_root: str = "."):
    """
    Run ViT inference on a single image.

    Args:
        image:        PIL Image or path string to the image file.
        dataset:      'brain' or 'breast'
        project_root: Root folder of the project.

    Returns:
        pred_class  (str):   Predicted class label.
        confidence  (float): Confidence of the top prediction (0–1).
        all_probs   (dict):  {label: probability} for all classes.
    """
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    else:
        image = image.convert("RGB")

    model  = load_model(dataset, project_root)
    tensor = preprocess(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(tensor)
        probs   = torch.softmax(outputs, dim=1)[0]

    labels     = CLASS_INFO[dataset]["labels"]
    pred_idx   = probs.argmax().item()
    pred_class = labels[pred_idx]
    confidence = probs[pred_idx].item()
    all_probs  = {label: round(probs[i].item(), 4)
                  for i, label in enumerate(labels)}

    return pred_class, confidence, all_probs


if __name__ == "__main__":
    img_path = sys.argv[1] if len(sys.argv) > 1 else None
    dataset  = sys.argv[2] if len(sys.argv) > 2 else "brain"

    if img_path is None:
        print("Usage: python vit_predict.py <image_path> <brain|breast>")
        sys.exit(1)

    pred_class, confidence, all_probs = predict(img_path, dataset)
    print(f"Prediction : {pred_class}")
    print(f"Confidence : {confidence:.2%}")
    print(f"All probs  : {all_probs}")
