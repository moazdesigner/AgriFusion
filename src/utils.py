import torch
import numpy as np
import os

def check_device():
    """Returns the available device (MPS for Mac, CUDA for Nvidia, or CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def count_parameters(model):
    """Counts trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def verify_image_paths(df, col_name='image_path'):
    """Checks how many images in the dataframe actually exist on disk."""
    missing = 0
    total = len(df)
    for path in df[col_name]:
        if not os.path.exists(path):
            missing += 1
    print(f"Verified {total} images. Missing: {missing}")
    return missing
def save_checkpoint(model, optimizer, filename="best_model.pth"):
    """Saves the model and optimizer state."""
    state = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(state, filename)
    print(f"Model saved to {filename}")

def load_checkpoint(model, filename="best_model.pth", device="cpu"):
    """Loads model weights."""
    checkpoint = torch.load(filename, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model
