import torch
import os
import sys
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from preprocessing.dataloader import get_dataloaders
from models.cnn.mobilenet import get_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 📁 dataset path
dataset_path = "breast_dataset"

# Load data
_, _, test_loader = get_dataloaders(
    dataset_path,
    batch_size=32,
    num_workers=0   # Windows fix
)

# Load model
model = get_model(num_classes=2)
model.load_state_dict(torch.load("mobilenet_breast.pth", map_location=device))
model.to(device)
model.eval()

predictions = []
true_labels = []

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
df.to_csv('mobilenet_predictions.csv', index=False)

# Metrics
print("\nClassification Report:")
print(classification_report(true_labels, predictions, target_names=['benign', 'malignant']))

print("\nConfusion Matrix:")
cm = confusion_matrix(true_labels, predictions)
print(cm)

# Optional: Save confusion matrix image
import matplotlib.pyplot as plt
import seaborn as sns

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("MobileNetV2 Confusion Matrix")
plt.savefig("mobilenet_confusion_matrix.png")
plt.show()

print("\n Inference complete! CSV + metrics + confusion matrix saved.")