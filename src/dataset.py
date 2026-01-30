import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import os

class SoilHealthDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_len=128, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            tokenizer (BertTokenizer): Transformer tokenizer.
            max_len (int): Maximum sequence length for text.
            transform (callable, optional): Optional transform to be applied on a sample (for images).
        """
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.transform = transform
        
        # Mapping labels to integers
        self.label_map = {
            'Healthy': 0,
            'N_Deficient': 1,
            'P_Deficient': 2,
            'K_Deficient': 3
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Text Processing
        text = str(self.data.iloc[idx]['text_description'])
        label_str = self.data.iloc[idx]['deficiency_label']
        label = self.label_map.get(label_str, 0)
        
        # Tokenize text (Updated method)
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        # Image Processing
        img_path = self.data.iloc[idx]['image_path']
        
        # Handle cases where image might be missing or path is wrong
        try:
            image = Image.open(img_path).convert("RGB")
        except (OSError, FileNotFoundError):
            # Fallback for missing images (create a black image)
            image = Image.new('RGB', (224, 224), color='black')
            
        if self.transform:
            image = self.transform(image)

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'pixel_values': image,  # This is the new image tensor
            'label': torch.tensor(label, dtype=torch.long)
        }
