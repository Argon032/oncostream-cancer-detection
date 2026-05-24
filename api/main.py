"""
main.py
-------
FastAPI backend for OncoStream.

Exposes a single endpoint:
    POST /predict
        - Accepts an image file and a dataset type ('brain' or 'breast')
        - Returns predicted class, confidence, per-class probabilities,
          and a Grad-CAM heatmap as a base64-encoded PNG string

Model selection (based on evaluation results):
    brain  → ViT (best macro F1: 0.9560, macro AUC: 0.9846)
    breast → ViT (best macro AUC: 0.9994, macro F1: 0.9883)

HOW TO RUN
----------
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
from fastapi.staticfiles import StaticFiles
from PIL import Image

# ── Project root ───────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

# ── Model selection per dataset ────────────────────────────────────────────────
# To change the deployed model, update the value here.
# The inference script and Grad-CAM will both follow this automatically.
MODEL_SELECTION = {
    "brain":  "vit",
    "breast": "vit",
}

app = FastAPI(
    title="OncoStream API",
    description=(
        "Cancer detection API. "
        "Uses Vision Transformer (ViT) for both brain MRI and breast "
        "histopathology classification, with Grad-CAM explainability overlays."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

VALID_DATASETS = {"brain", "breast"}


@app.get("/")
def root():
    return {
        "message": "OncoStream API is running. POST to /predict to classify an image.",
        "models":  MODEL_SELECTION,
    }


@app.post("/predict")
async def predict(
    file:    UploadFile = File(...),
    dataset: str        = Form(...),
):
    if dataset not in VALID_DATASETS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid dataset '{dataset}'. Choose 'brain' or 'breast'."
        )

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
        from gradcam.gradcam_utils import generate_gradcam
        heatmap_image = generate_gradcam(
            image,
            dataset=dataset,
            project_root=PROJECT_ROOT
        )
        buffer = io.BytesIO()
        heatmap_image.save(buffer, format="PNG")
        heatmap_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"[WARNING] Grad-CAM failed: {e}")

    return JSONResponse({
        "predicted_class":   pred_class,
        "confidence":        round(confidence * 100, 2),
        "all_probabilities": all_probs,
        "gradcam_image":     heatmap_b64,
        "dataset":           dataset,
        "model_used":        model_name,
    })


# ── Serve frontend ─────────────────────────────────────────────────────────────
frontend_path = os.path.join(PROJECT_ROOT, "frontend")
if os.path.isdir(frontend_path):
    app.mount("/", StaticFiles(directory=frontend_path, html=True),
              name="frontend")