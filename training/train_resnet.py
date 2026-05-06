import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models.cnn.resnet50 import get_model
from preprocessing.dataloader import get_dataloaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset path
dataset_path = "breast_dataset"

train_loader, val_loader, test_loader = get_dataloaders(
    dataset_path,
    batch_size=16,   #  ResNet heavier than MobileNet
    num_workers=0
)

model = get_model(num_classes=2).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

epochs = 1   #  keep 1 for testing first

best_acc = 0.0

for epoch in range(epochs):
    model.train()
    train_loss = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        if batch_idx % 10 == 0:
            print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")

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

    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "resnet_breast.pth")
        print("Best model saved!")

torch.save(model.state_dict(), "resnet_breast.pth")
print("Training complete!")