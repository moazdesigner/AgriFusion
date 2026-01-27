import torchvision.transforms as T

# Image Hyperparameters
IMG_SIZE = 224
BATCH_SIZE = 16
MEAN = [0.485, 0.456, 0.406]  # ImageNet standards
STD = [0.229, 0.224, 0.225]   # ImageNet standards

def get_train_transforms():
    """
    Returns transformations for the training set (with augmentation).
    """
    return T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=15),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])

def get_val_transforms():
    """
    Returns transformations for the validation/test set (no augmentation).
    """
    return T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])