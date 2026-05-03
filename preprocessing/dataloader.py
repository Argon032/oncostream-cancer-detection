from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ImageNet normalization stats — required for all models
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def get_dataloaders(dataset_path, batch_size=32, num_workers=2):
    """
    Build and return train, val, and test DataLoaders.

    Args:
        dataset_path: Full path to the dataset folder (e.g.
                      "/content/drive/MyDrive/.../datasets/brain").
                      Must contain train/, val/, test/ subfolders.
        batch_size:   Images per batch (default 32 for CNN, use 16 for transformers).
        num_workers:  Parallel data loading workers (default 2, use 0 on Windows).

    Returns:
        train_loader, val_loader, test_loader
    """
    train_dataset = datasets.ImageFolder(f"{dataset_path}/train",
                                         transform=train_transform)
    val_dataset   = datasets.ImageFolder(f"{dataset_path}/val",
                                         transform=val_test_transform)
    test_dataset  = datasets.ImageFolder(f"{dataset_path}/test",
                                         transform=val_test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers,
                              pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=True)

    print(f"\nLoaded dataset from: {dataset_path}")
    print(f"Train: {len(train_dataset)} images | "
          f"Val: {len(val_dataset)} images | "
          f"Test: {len(test_dataset)} images")
    print("Classes:", train_dataset.classes)

    return train_loader, val_loader, test_loader