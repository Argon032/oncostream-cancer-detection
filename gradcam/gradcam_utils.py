"""
gradcam_utils.py
-----------------
Attention Rollout visualisation for OncoStream.
Uses ViT's internal attention weights for spatial explainability.
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
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
               "labels": ["glioma", "meningioma", "no_tumor", "pituitary"]},
    "breast": {"num_classes": 2,
               "labels": ["benign", "malignant"]},
}

MODEL_SELECTION = {
    "brain":  "vit",
    "breast": "vit",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _overlay(image: Image.Image, cam: np.ndarray, alpha: float = 0.55) -> Image.Image:
    cam_resized = cv2.resize(cam, (224, 224))
    # Stretch to full range
    cam_min, cam_max = cam_resized.min(), cam_resized.max()
    if cam_max > cam_min:
        cam_resized = (cam_resized - cam_min) / (cam_max - cam_min)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    img_np  = np.array(image.resize((224, 224)).convert("RGB"))
    overlay = (alpha * heatmap + (1.0 - alpha) * img_np).astype(np.uint8)
    return Image.fromarray(overlay)


def _attention_rollout(model: torch.nn.Module, tensor: torch.Tensor) -> np.ndarray:
    """
    Attention Rollout for torchvision ViT-B/16.
    Hooks into each encoder block's self-attention to capture attention maps,
    then rolls them out across all layers.
    """
    attention_maps = []
    hooks = []

    def make_hook(idx):
        def hook_fn(module, input, output):
            # torchvision MultiheadAttention returns (output, attn_weights)
            # but attn_weight_output is not returned by default
            # so we hook the input — Q, K, V are in input[0]
            # Instead we use the module's internal state after forward
            pass
        return hook_fn

    # torchvision ViT stores attention in encoder.layers[i].self_attention
    # We need to monkey-patch to capture attn weights
    original_forwards = []

    for block in model.encoder.layers:
        attn = block.self_attention
        original_forward = attn.forward

        def patched_forward(q, k, v, *args, _orig=original_forward, **kwargs):
            out, weights = _orig(q, k, v, need_weights=True, average_attn_weights=True)
            attention_maps.append(weights.detach())
            return out, weights

        attn.forward = patched_forward
        original_forwards.append((attn, original_forward))

    with torch.no_grad():
        _ = model(tensor)

    # Restore original forwards
    for attn, orig in original_forwards:
        attn.forward = orig

    # Rollout: multiply attention maps across layers
    # Each attention_map: (B, N, N) where N = 197 (1 cls + 196 patches)
    rollout = torch.eye(attention_maps[0].shape[-1]).to(DEVICE)
    for attn_map in attention_maps:
        attn_map = attn_map[0]  # (N, N)
        # Add residual connection
        attn_map = attn_map + torch.eye(attn_map.shape[-1]).to(DEVICE)
        attn_map = attn_map / attn_map.sum(dim=-1, keepdim=True)
        rollout = torch.matmul(attn_map, rollout)

    # cls token row → attention from cls to all patches
    mask = rollout[0, 1:]  # (196,) — skip cls token
    mask = mask.cpu().numpy()
    mask = mask.reshape(14, 14)
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-7)
    return mask


def _load_model(model_name: str, dataset: str, project_root: str):
    num_classes = CLASS_INFO[dataset]["num_classes"]

    if model_name == "vit":
        sys.path.insert(0, os.path.join(project_root, "models", "transformer"))
        from vit import get_model
    else:
        raise ValueError(f"Unsupported model: '{model_name}'")

    model     = get_model(num_classes=num_classes)
    ckpt_path = os.path.join(project_root, "results", dataset,
                             f"{model_name}_best.pth")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}.")

    checkpoint = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    return model


def generate_gradcam(image, dataset: str, project_root: str = ".",
                     class_idx: int = None, alpha: float = 0.55) -> Image.Image:
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    else:
        image = image.convert("RGB")

    model_name = MODEL_SELECTION[dataset]
    model      = _load_model(model_name, dataset, project_root)
    tensor     = preprocess(image).unsqueeze(0).to(DEVICE)
    cam        = _attention_rollout(model, tensor)

    return _overlay(image, cam, alpha)


if __name__ == "__main__":
    img_path    = sys.argv[1] if len(sys.argv) > 1 else None
    dataset     = sys.argv[2] if len(sys.argv) > 2 else "brain"
    output_path = sys.argv[3] if len(sys.argv) > 3 else "gradcam_output.png"

    if img_path is None:
        print("Usage: python gradcam_utils.py <image_path> <brain|breast> [output_path]")
        sys.exit(1)

    result = generate_gradcam(img_path, dataset=dataset)
    result.save(output_path)
    print(f"Saved → {output_path}")