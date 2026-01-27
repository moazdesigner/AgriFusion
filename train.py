import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader

from src.dataset import SoilHealthDataset
from src.models import AgriFusionModel
from src.engine import train_fn, eval_fn
from src.config import get_train_transforms, get_val_transforms, BATCH_SIZE
from src.utils import check_device, save_checkpoint

def main():
    # 1. SETUP
    EPOCHS = 5
    LR = 3e-4
    DATA_FILE = 'final_soil_data.csv'
    
    device = check_device()
    print(f"Using device: {device}")

    # 2. DATA PREP
    df = pd.read_csv(DATA_FILE)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['deficiency_label'])
    
    train_df.to_csv('train_temp.csv', index=False)
    val_df.to_csv('val_temp.csv', index=False)
    
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    train_ds = SoilHealthDataset('train_temp.csv', tokenizer, transform=get_train_transforms())
    val_ds = SoilHealthDataset('val_temp.csv', tokenizer, transform=get_val_transforms())
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 3. MODEL INITIALIZATION
    model = AgriFusionModel(num_classes=4)
    model.to(device)
    
    # 4. OPTIMIZER & SCHEDULER
    optimizer = AdamW(model.parameters(), lr=LR)
    num_train_steps = int(len(train_df) / BATCH_SIZE * EPOCHS)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=0, 
        num_training_steps=num_train_steps
    )
    
    # 5. TRAINING LOOP
    best_accuracy = 0
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        
        train_loss = train_fn(train_loader, model, optimizer, device, scheduler)
        val_loss, val_acc = eval_fn(val_loader, model, device)
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # Save Best Model
        if val_acc > best_accuracy:
            print(f"Accuracy improved from {best_accuracy:.4f} to {val_acc:.4f}. Saving model...")
            best_accuracy = val_acc
            save_checkpoint(model, optimizer, filename="best_agri_model.pth")

if __name__ == "__main__":
    main()