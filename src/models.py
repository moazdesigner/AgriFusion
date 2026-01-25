import torch.nn as nn
from transformers import DistilBertModel

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