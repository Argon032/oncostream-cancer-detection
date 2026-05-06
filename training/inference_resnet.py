import torch
import os
import sys
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Fix import paths
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from preprocessing.dataloader import get_dataloaders
from models.cnn.resnet50 import get_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset path
dataset_path = "breast_dataset"

# Load data
_, _, test_loader = get_dataloaders(
    dataset_path,
    batch_size=32,
    num_workers=0
)

# Load model
model = get_model(num_classes=2)
model.load_state_dict(torch.load("resnet_breast.pth", map_location=device))
model.to(device)
model.eval()

predictions = []
true_labels = []

# Inference
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        predictions.extend(preds.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

# Save predictions
df = pd.DataFrame({
    'true_label': true_labels,
    'prediction': predictions
})
df.to_csv('resnet_predictions.csv', index=False)

# Metrics
print("\nClassification Report:")
print(classification_report(true_labels, predictions, target_names=['benign', 'malignant']))

print("\nConfusion Matrix:")
cm = confusion_matrix(true_labels, predictions)
print(cm)

# Plot confusion matrix
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=['benign', 'malignant'],
            yticklabels=['benign', 'malignant'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("ResNet50 Confusion Matrix")
plt.savefig("resnet_confusion_matrix.png")
plt.show()

print("\nInference complete! CSV + metrics + confusion matrix saved.")