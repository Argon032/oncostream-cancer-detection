from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Data augmentation for training
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2),
    transforms.ToTensor()
])

# Validation and test transforms
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load datasets
train_dataset = datasets.ImageFolder("breast_dataset/train", transform=train_transform)
val_dataset = datasets.ImageFolder("breast_dataset/val", transform=test_transform)
test_dataset = datasets.ImageFolder("breast_dataset/test", transform=test_transform)

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)

print("Dataset loaded successfully!")