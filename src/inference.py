import torch
from transformers import DistilBertTokenizer
from PIL import Image
import torchvision.transforms as T

from src.models import AgriFusionModel
from src.config import get_val_transforms
from src.utils import check_device, load_checkpoint

class SoilHealthPredictor:
    def __init__(self, model_path="best_agri_model.pth"):
        self.device = check_device()
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.transforms = get_val_transforms()
        
        # Load Model
        self.model = AgriFusionModel(num_classes=4)
        try:
            self.model = load_checkpoint(self.model, filename=model_path, device=self.device)
            print("Model loaded successfully.")
        except FileNotFoundError:
            print("Warning: Model file not found. Initializing with random weights (for testing only).")
            
        self.model.to(self.device)
        self.model.eval()
        
        self.idx_to_label = {
            0: 'Healthy',
            1: 'N_Deficient',
            2: 'P_Deficient',
            3: 'K_Deficient'
        }

    def predict(self, text, image_path):
        # Preprocess Text
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Preprocess Image
        try:
            image = Image.open(image_path).convert("RGB")
            image = self.transforms(image).unsqueeze(0).to(self.device)
        except Exception as e:
            return {"error": f"Image processing failed: {str(e)}"}

        # Inference
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask, image)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            confidence, prediction = torch.max(probs, dim=1)
            
        return {
            "label": self.idx_to_label[prediction.item()],
            "confidence": confidence.item()
        }
