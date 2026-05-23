"""
main.py
-------
FastAPI backend for OncoStream.

Exposes a single endpoint:
    POST /predict
        - Accepts an image file and a dataset type ('brain' or 'breast')
        - Returns predicted class, confidence, per-class probabilities,
          and a Grad-CAM heatmap as a base64-encoded PNG string

HOW TO RUN
Install dependencies:
    pip install fastapi uvicorn python-multipart pillow

Start the server:
    uvicorn api.main:app --reload --port 8000

The frontend POSTs to:
    http://localhost:8000/predict
"""

import io
import os
import base64
import sys

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

# ── Project root ───────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

# ── Model selection per dataset ────────────────────────────────────────────────
# Update these if the selected model changes.
MODEL_SELECTION = {
    "brain":  "vit",       # ViT — best F1 on brain MRI
    "breast": "resnet50",  # ResNet50 — best F1 on breast histopathology
}

app = FastAPI(
    title="OncoStream API",
    description=(
        "Cancer detection API. "
        "Uses ViT for brain MRI classification and ResNet50 for breast "
        "histopathology, with Grad-CAM explainability overlays."
    ),
    version="1.0.0",
)

# Allow all origins during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

VALID_DATASETS = {"brain", "breast"}


@app.post("/predict")
async def predict(
    file:    UploadFile = File(...),
    dataset: str        = Form(...),
):
    """
    Classify a medical image and return Grad-CAM explainability.

    Form fields:
        file    — image file (JPEG, PNG, etc.)
        dataset — 'brain' or 'breast'

    Returns:
        predicted_class   : str   — e.g. "glioma"
        confidence        : float — e.g. 94.21 (as a percentage)
        all_probabilities : dict  — {class_label: probability}
        gradcam_image     : str   — base64-encoded PNG of heatmap overlay
        dataset           : str   — echoed back for frontend use
        model_used        : str   — which model produced the result
    """
    # ── Validate dataset ───────────────────────────────────────────────────────
    if dataset not in VALID_DATASETS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid dataset '{dataset}'. Choose 'brain' or 'breast'."
        )

    # ── Read image ─────────────────────────────────────────────────────────────
    try:
        contents = await file.read()
        image    = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Could not read the uploaded file. Please upload a valid image."
        )

    model_name = MODEL_SELECTION[dataset]

    # ── Prediction ─────────────────────────────────────────────────────────────
    try:
        if model_name == "vit":
            from inference.vit_predict import predict as run_predict
        elif model_name == "resnet50":
            from inference.resnet_predict import predict as run_predict
        elif model_name == "mobilenet":
            from inference.mobilenet_predict import predict as run_predict
        elif model_name == "swin":
            from inference.swin_predict import predict as run_predict
        else:
            raise ValueError(f"No inference script for model '{model_name}'")

        pred_class, confidence, all_probs = run_predict(
            image, dataset=dataset, project_root=PROJECT_ROOT
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500,
                            detail=f"Prediction failed: {str(e)}")

    # ── Grad-CAM ───────────────────────────────────────────────────────────────
    heatmap_b64 = None
    try:
        from gradcam.gradcam_generator import generate_gradcam
        heatmap_image = generate_gradcam(
            image,
            dataset=dataset,
            model_name=model_name,
            project_root=PROJECT_ROOT
        )
        buffer = io.BytesIO()
        heatmap_image.save(buffer, format="PNG")
        heatmap_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    except Exception as e:
        # Grad-CAM failure does not block the prediction result
        print(f"[WARNING] Grad-CAM failed: {e}")

    return JSONResponse({
        "predicted_class":   pred_class,
        "confidence":        round(confidence * 100, 2),
        "all_probabilities": all_probs,
        "gradcam_image":     heatmap_b64,
        "dataset":           dataset,
        "model_used":        model_name,
    })

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

frontend_path = os.path.join(PROJECT_ROOT, "frontend")
app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")