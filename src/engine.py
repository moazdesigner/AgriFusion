import torch
from tqdm import tqdm

def train_fn(data_loader, model, optimizer, device, scheduler=None):
    model.train()
    final_loss = 0
    
    # Progress bar
    tk0 = tqdm(data_loader, total=len(data_loader), desc="Training")
    
    for data in tk0:
        # Move inputs to device
        input_ids = data['input_ids'].to(device, dtype=torch.long)
        attention_mask = data['attention_mask'].to(device, dtype=torch.long)
        pixel_values = data['pixel_values'].to(device, dtype=torch.float)
        targets = data['label'].to(device, dtype=torch.long)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(input_ids, attention_mask, pixel_values)
        
        # Calculate Loss (Cross Entropy)
        loss = torch.nn.CrossEntropyLoss()(outputs, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()
            
        final_loss += loss.item()
        
        # Update progress bar
        tk0.set_postfix(loss=loss.item())
        
    return final_loss / len(data_loader)

def eval_fn(data_loader, model, device):
    model.eval()
    final_loss = 0
    correct_predictions = 0
    total_samples = 0
    
    with torch.no_grad():
        tk0 = tqdm(data_loader, total=len(data_loader), desc="Validating")
        for data in tk0:
            input_ids = data['input_ids'].to(device, dtype=torch.long)
            attention_mask = data['attention_mask'].to(device, dtype=torch.long)
            pixel_values = data['pixel_values'].to(device, dtype=torch.float)
            targets = data['label'].to(device, dtype=torch.long)
            
            outputs = model(input_ids, attention_mask, pixel_values)
            loss = torch.nn.CrossEntropyLoss()(outputs, targets)
            final_loss += loss.item()
            
            # Calculate accuracy
            _, preds = torch.max(outputs, dim=1)
            correct_predictions += torch.sum(preds == targets)
            total_samples += targets.size(0)
            
    accuracy = correct_predictions.double() / total_samples
    return final_loss / len(data_loader), accuracy.item()