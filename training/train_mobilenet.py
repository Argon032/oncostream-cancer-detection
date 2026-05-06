import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys

# Fix import path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models.cnn.mobilenet import get_model
from preprocessing.dataloader import get_dataloaders

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 📁 SET YOUR DATASET PATH HERE
dataset_path = "breast_dataset"   # ⚠️ change if needed

# Load data
train_loader, val_loader, test_loader = get_dataloaders(
    dataset_path,
    batch_size=32,
    num_workers=0   # ✅ IMPORTANT for Windows
)

# Load model
model = get_model(num_classes=2).to(device)

# Loss & optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 1   # 🔥 keep 1 for testing first

best_acc = 0.0

for epoch in range(epochs):
    model.train()
    train_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    print(f"Epoch {epoch+1}, Training Loss: {train_loss:.4f}")

    # Validation
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    print(f"Validation Accuracy: {acc:.2f}%")

    # Save best model
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "mobilenet_breast.pth")
        print("Best model saved!")

# Save final model
torch.save(model.state_dict(), "mobilenet_breast.pth")

print("Training complete & model saved")