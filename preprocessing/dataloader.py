from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloaders(dataset_name, batch_size=32):
    base_path = f"datasets/{dataset_name}"

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2),
        transforms.ToTensor()
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    train_dataset = datasets.ImageFolder(f"{base_path}/train", transform=train_transform)
    val_dataset = datasets.ImageFolder(f"{base_path}/val", transform=test_transform)
    test_dataset = datasets.ImageFolder(f"{base_path}/test", transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    print(f"\nLoaded {dataset_name} dataset")
    print("Classes:", train_dataset.classes)

    return train_loader, val_loader, test_loader