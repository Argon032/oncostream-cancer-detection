---
title: OncoStream
emoji: 🔬
colorFrom: red
colorTo: gray
sdk: docker
pinned: false
---

# OncoStream - Cancer Detection System

A deep learning system for cancer classification from medical images. Classifies brain MRI scans into four tumour categories and breast histopathology slides as benign or malignant. Four models (ResNet50, MobileNetV2, ViT, Swin Transformer) were trained and benchmarked; **Vision Transformer (ViT) is deployed for both datasets** based on evaluation results.

## Project Structure

```
oncostream-cancer-detection/
├── datasets/                   # Downloaded datasets go here (not committed)
│   ├── brain/                  # train/ val/ test/ subfolders
│   └── breast/                 # train/ val/ test/ subfolders
├── preprocessing/
│   ├── dataset_organize.py     # Flattens raw BreaKHis folder structure
│   ├── dataset_split.py        # Splits datasets into train/val/test
│   ├── dataloader.py           # Shared PyTorch DataLoader (used by training scripts)
│   └── visualize_augmentation.py # For random samples of augmentation examples
├── models/
│   ├── cnn/
│   │   ├── resnet50.py
│   │   └── mobilenet.py
│   └── transformer/
│       ├── vit.py
│       └── swin.py
├── training/
│   ├── train_cnn.py            # Trains ResNet50 or MobileNetV2
│   └── train_transformer.py    # Trains ViT or Swin
├── inference/
│   ├── vit_predict.py          # Used by the live API
│   ├── swin_predict.py
│   ├── resnet_predict.py
│   └── mobilenet_predict.py
├── evaluation/
│   ├── metrics.py              # Precision, recall, F1, AUC-ROC per model
│   ├── confusion_matrix.py     # Confusion matrix plots
│   └── compare_models.py       # Cross-model comparison plots and summary
├── gradcam/
│   └── gradcam_utils.py        # Attention Rollout (ViT) - used by live API
├── api/
│   └── main.py                 # FastAPI backend - POST /predict
├── frontend/                   # Web UI
├── results/
│   ├── brain/                  # Checkpoints + CSVs go here after training
│   ├── breast/
│   └── plots/                  # All generated figures saved here
└── requirements.txt
```

## Installation

```bash
git clone https://github.com/Argon032/oncostream-cancer-detection.git
cd oncostream-cancer-detection
pip install -r requirements.txt
```

## Dataset Setup

Datasets are not included in the repo. Download and set up manually:

**Brain MRI**
1. Download from [Kaggle - Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
2. Extract into `datasets/brain/` - the Kaggle download comes pre-split into `Training/` and `Testing/` folders
3. Run `dataset_split.py` to create the `val/` split:
```bash
python preprocessing/dataset_split.py
```

**Breast Histopathology**
1. Download from [Kaggle - BreaKHis](https://www.kaggle.com/datasets/ambarish/breakhis)
2. Extract into `datasets/breast/` - the download has a nested folder structure
3. Run `dataset_organize.py` first to flatten it, then `dataset_split.py`:
```bash
python preprocessing/dataset_organize.py
python preprocessing/dataset_split.py
```

Expected structure after setup:
```
datasets/brain/train/   glioma/  meningioma/  no_tumor/  pituitary/
datasets/brain/val/     ...
datasets/brain/test/    ...
datasets/breast/train/  benign/  malignant/
datasets/breast/val/    ...
datasets/breast/test/   ...
```

## Training

Training is designed to run on **Google Colab or Kaggle** (GPU required). A pre-configured Colab notebook with all training cells is available here:

**[OncoStream Training Notebook](https://colab.research.google.com/drive/1OXB5E8IeuW6QzWvQSMqc_V73fhQUmIDi?usp=sharing)**

The notebook runs cells sequentially - no additional configuration needed setting the `project_root` path and kaggle API key.

**To train manually**, edit the `CONFIG` block at the top of the relevant script and run:

```bash
# CNN models (ResNet50 or MobileNetV2)
python training/train_cnn.py

# Transformer models (ViT or Swin)
python training/train_transformer.py
```

Key CONFIG options:
```python
"model_name":  "vit"       # vit / swin / resnet50 / mobilenet
"dataset":     "brain"     # brain / breast
"project_root": "."        # update to your project root path on Colab
```

Checkpoints are saved automatically to `results/{dataset}/{model}_best.pth`.

## Using Pre-trained Checkpoints

Trained model checkpoints (`.pth` files) are available on Kaggle. 
**Important Note** After downloading:
1. Place brain checkpoints in `results/brain/`
2. Place breast checkpoints in `results/breast/`

Expected filenames:
```
results/brain/vit_best.pth
results/brain/swin_best.pth
results/brain/resnet50_best.pth
results/brain/mobilenet_best.pth
results/breast/vit_best.pth
results/breast/swin_best.pth
results/breast/resnet50_best.pth
results/breast/mobilenet_best.pth
```

This step is required before running evaluation or the API.

## Evaluation

Run each script independently. Edit the `CONFIG` block to select model and dataset.

```bash
python evaluation/metrics.py          # Precision, recall, F1, AUC-ROC + plots
python evaluation/confusion_matrix.py # Confusion matrix plot
python evaluation/compare_models.py   # Cross-model comparison (run after all models)
```

All plots are saved to `results/plots/`.

## Running the API

```bash
pip install fastapi uvicorn python-multipart
uvicorn api.main:app --reload --port 8000
```

**Endpoint:** `POST http://localhost:8000/predict`

| Field | Type | Description |
|---|---|---|
| `file` | image | JPEG or PNG scan |
| `dataset` | string | `brain` or `breast` |

**Response:**
```json
{
  "predicted_class":   "glioma",
  "confidence":        94.21,
  "all_probabilities": {"glioma": 0.9421, "meningioma": 0.031, ...},
  "gradcam_image":     "<base64 PNG>",
  "dataset":           "brain",
  "model_used":        "vit"
}
```

The deployed model is ViT for both datasets. To change this, update `MODEL_SELECTION` in `api/main.py`.

## Results

**Brain MRI** (4 classes: Glioma, Meningioma, No Tumor, Pituitary)

| Model | Val Accuracy | Macro F1 | Macro AUC |
|---|---|---|---|
| **ViT** ! | 0.9905 | **0.9560** | 0.9846 |
| ResNet50 | 0.9857 | 0.9515 | 0.9928 |
| Swin | 0.9929 | 0.9463 | 0.9918 |
| MobileNetV2 | 0.9750 | 0.9471 | 0.9909 |

**Breast Histopathology** (2 classes: Benign, Malignant)

| Model | Val Accuracy | Macro F1 | Macro AUC |
|---|---|---|---|
| **ViT** ! | 0.9850 | 0.9883 | **0.9994** |
| Swin | 0.9750 | **0.9917** | 0.9992 |
| ResNet50 | 0.9833 | 0.9867 | 0.9989 |
| MobileNetV2 | 0.9817 | 0.9800 | 0.9980 |

! Deployed model

Working Interface - [OncoStream](https://huggingface.co/spaces/Argon032/oncostream)

## Team

| Name | Responsibility |
|---|---|
| Dr. Amrutanshu Panigrahi | Supervisor |
| Arunima (Lead) | Swin Transformer · Training pipeline · Attention Rollout |
| Bedangshi | MobileNetV2 · Breast dataset pipeline · Preprocessing |
| Symantak | Frontend · API integration · Dataset pipeline |
| Samikshya | Vision Transformer (ViT) · Brain dataset training execution |
| Radhakanta | ResNet50 · Evaluation scripts · Literature survey |