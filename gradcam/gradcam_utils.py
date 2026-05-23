"""
gradcam_generator.py
---------------------
Grad-CAM visualisation for OncoStream.

    brain  → ViT      (hooks into last encoder block, 14x14 patch grid)
    breast → ResNet50 (hooks into layer4, 7x7 conv feature map)

The core Grad-CAM logic is shared — only the hook location and feature
map reshaping differ between the two architectures.

Usage (standalone):
    from gradcam.gradcam_generator import generate_gradcam
    from PIL import Image

    heatmap = generate_gradcam(Image.open("scan.jpg"), dataset="brain")
    heatmap.save("heatmap.png")
"""

import os
import sys
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import cv2

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

CLASS_INFO = {
    "brain":  {"num_classes": 4,
               "labels": ["glioma", "meningioma", "pituitary", "no_tumor"]},
    "breast": {"num_classes": 2,
               "labels": ["benign", "malignant"]},
}

# Which model handles each dataset
MODEL_SELECTION = {
    "brain":  "vit",
    "breast": "resnet50",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#  Core Grad-CAM logic 
def _compute_cam(features: torch.Tensor,
                 grads: torch.Tensor) -> np.ndarray:
    """
    Given feature maps and their gradients (both in CxHxW format),
    compute the normalised Grad-CAM activation map.

    Args:
        features: (C, H, W) feature map tensor.
        grads:    (C, H, W) gradient tensor.

    Returns:
        cam: np.ndarray in [0, 1], shape (H, W).
    """
    weights = grads.mean(dim=(1, 2))                          # (C,)
    cam     = (weights[:, None, None] * features).sum(dim=0)  # (H, W)
    cam     = torch.relu(cam)
    cam     = cam.detach().cpu().numpy()
    if cam.max() > 0:
        cam = cam / cam.max()
    return cam


def _overlay(image: Image.Image,
             cam: np.ndarray,
             alpha: float = 0.45) -> Image.Image:
    """
    Resize cam to 224x224, apply JET colourmap, blend with original.
    """
    cam_resized = cv2.resize(cam, (224, 224))
    heatmap     = cv2.applyColorMap(np.uint8(255 * cam_resized),
                                    cv2.COLORMAP_JET)
    heatmap     = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    img_np      = np.array(image.resize((224, 224)).convert("RGB"))
    overlay     = (alpha * heatmap + (1.0 - alpha) * img_np).astype(np.uint8)
    return Image.fromarray(overlay)


#  ViT Grad-CAM 
def _gradcam_vit(model: torch.nn.Module,
                 tensor: torch.Tensor,
                 class_idx: int = None):
    """
    Hook into the last encoder block (B, N, C).
    Skip the class token, reshape 196 patch tokens → 14x14 spatial grid.
    """
    features_store = {}

    def forward_hook(module, input, output):
        features_store["out"] = output
        output.retain_grad()

    hook = model.encoder.layers[-1].register_forward_hook(forward_hook)

    output = model(tensor)
    if class_idx is None:
        class_idx = output.argmax(dim=1).item()

    model.zero_grad()
    output[0, class_idx].backward()
    hook.remove()

    feat  = features_store["out"]
    # Skip class token → patch tokens (B, 196, C)
    f = feat[0, 1:]            # (196, C)
    g = feat.grad[0, 1:]       # (196, C)

    h = w = int(f.shape[0] ** 0.5)   # 14
    f = f.reshape(h, w, -1).permute(2, 0, 1)  # (C, 14, 14)
    g = g.reshape(h, w, -1).permute(2, 0, 1)  # (C, 14, 14)

    return _compute_cam(f, g), class_idx


#  ResNet50 Grad-CAM
def _gradcam_resnet(model: torch.nn.Module,
                    tensor: torch.Tensor,
                    class_idx: int = None):
    """
    Hook into model.layer4 (B, 2048, 7, 7) — last conv block.
    """
    features_store = {}
    grads_store    = {}

    def forward_hook(module, input, output):
        features_store["out"] = output

    def backward_hook(module, grad_input, grad_output):
        grads_store["out"] = grad_output[0]

    target = model.layer4[-1]
    fhook  = target.register_forward_hook(forward_hook)
    bhook  = target.register_full_backward_hook(backward_hook)

    output = model(tensor)
    if class_idx is None:
        class_idx = output.argmax(dim=1).item()

    model.zero_grad()
    output[0, class_idx].backward()
    fhook.remove()
    bhook.remove()

    f = features_store["out"][0]  # (2048, 7, 7)
    g = grads_store["out"][0]     # (2048, 7, 7)

    return _compute_cam(f, g), class_idx


#  Model loader
def _load_model(model_name: str, dataset: str, project_root: str):
    num_classes = CLASS_INFO[dataset]["num_classes"]

    if model_name == "vit":
        sys.path.insert(0, os.path.join(project_root, "models", "transformer"))
        from vit import get_model

    elif model_name == "resnet50":
        sys.path.insert(0, os.path.join(project_root, "models", "cnn"))
        from resnet50 import get_model

    else:
        raise ValueError(f"Unsupported model for Grad-CAM: '{model_name}'")

    model     = get_model(num_classes=num_classes)
    ckpt_path = os.path.join(project_root, "results", dataset,
                             f"{model_name}_best.pth")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}. "
            f"Train '{model_name}' on '{dataset}' first."
        )

    checkpoint = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    return model

#  Public interface
def generate_gradcam(image,
                     dataset: str,
                     project_root: str = ".",
                     class_idx: int = None,
                     alpha: float = 0.45) -> Image.Image:
    """
    Load the selected model for the dataset, compute Grad-CAM, return overlay.

    Args:
        image:        PIL Image or path string.
        dataset:      'brain' or 'breast'
        project_root: Root folder of the project.
        class_idx:    Class to explain. None = predicted class.
        alpha:        Heatmap opacity (0-1).

    Returns:
        PIL Image — original scan with Grad-CAM heatmap overlaid.
    """
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    else:
        image = image.convert("RGB")

    model_name = MODEL_SELECTION[dataset]
    model      = _load_model(model_name, dataset, project_root)
    tensor     = preprocess(image).unsqueeze(0).to(DEVICE)

    if model_name == "vit":
        cam, _ = _gradcam_vit(model, tensor, class_idx)
    elif model_name == "resnet50":
        cam, _ = _gradcam_resnet(model, tensor, class_idx)

    return _overlay(image, cam, alpha)


if __name__ == "__main__":
    import sys as _sys
    img_path    = _sys.argv[1] if len(_sys.argv) > 1 else None
    dataset     = _sys.argv[2] if len(_sys.argv) > 2 else "brain"
    output_path = _sys.argv[3] if len(_sys.argv) > 3 else "gradcam_output.png"

    if img_path is None:
        print("Usage: python gradcam_generator.py <image_path> "
              "<brain|breast> [output_path]")
        _sys.exit(1)

    result = generate_gradcam(img_path, dataset=dataset)
    result.save(output_path)
    print(f"Grad-CAM saved → {output_path}")