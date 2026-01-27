import torch.nn as nn
from transformers import DistilBertModel
import torchvision.models as models
import torch

class SoilTextEncoder(nn.Module):
    def __init__(self, pretrained_model_name='distilbert-base-uncased', freeze_bert=True):
        super(SoilTextEncoder, self).__init__()
        self.bert = DistilBertModel.from_pretrained(pretrained_model_name)
        
        # Freezing BERT layers to speed up training
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
                
    def forward(self, input_ids, attention_mask):
        # DistilBert outputs: (last_hidden_state, )
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Using the representation of the [CLS] token (first token)
        hidden_state = output.last_hidden_state
        cls_token_emb = hidden_state[:, 0, :]
        
        return cls_token_emb
class SoilImageEncoder(nn.Module):
    def __init__(self, output_dim=512, freeze_resnet=True):
        super(SoilImageEncoder, self).__init__()
        # Load pre-trained ResNet18
        resnet = models.resnet18(pretrained=True)
        
        # Removing the last classification layer (fc)
        # Only want the features (embeddings)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        
        # Freeze ResNet weights
        if freeze_resnet:
            for param in self.resnet.parameters():
                param.requires_grad = False
                
    def forward(self, images):
        # Input shape: (Batch, 3, 224, 224)
        features = self.resnet(images)
        
        # Output shape is (Batch, 512, 1, 1), so we flatten it
        return features.view(features.size(0), -1)    
class AgriFusionModel(nn.Module):
    def __init__(self, num_classes=4, freeze_backbones=True):
        super(AgriFusionModel, self).__init__()
        # Initialize the two branches
        self.text_encoder = SoilTextEncoder(freeze_bert=freeze_backbones)
        self.image_encoder = SoilImageEncoder(freeze_resnet=freeze_backbones)
        
        # Define the fusion layer
        # Text (768 from DistilBERT) + Image (512 from ResNet18) = 1280
        self.fusion_dim = 768 + 512
        
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, input_ids, attention_mask, images):
        # Get features from both branches
        text_features = self.text_encoder(input_ids, attention_mask)
        image_features = self.image_encoder(images)
        
        # Concatenate features
        combined_features = torch.cat((text_features, image_features), dim=1)
        
        # Pass through classifier
        output = self.classifier(combined_features)
        return output    
