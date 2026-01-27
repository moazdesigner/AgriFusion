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