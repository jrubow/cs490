import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import os
from typing import List, Dict, Union
from transformers import BertTokenizer, BertModel, BertConfig
from tqdm import tqdm

# --- Utility: Data Loader ---
def get_data(path: str) -> List[Dict[str, Union[str, int]]]:
    data = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

# --- Dataset Class ---
class SarcasmDataset(Dataset):
    def __init__(self, data: List[Dict], tokenizer: BertTokenizer, max_length: int = 128):
        """
        Args:
            data: List of dictionaries (from get_data)
            tokenizer: Instance of BertTokenizer
            max_length: Maximum sequence length for truncation/padding
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        Returns a dictionary containing:
            - input_ids: (max_length,) Tensor of token IDs 
            - attention_mask: (max_length,) Tensor for attention masking
            - label: (1,) Tensor containing the label
        """
        item = self.data[index]
        encoding = self.tokenizer(item['headline'], padding='max_length', truncation=True, max_length=self.max_length, return_attention_mask=True, return_tensors='pt')
        return {'input_ids': encoding['input_ids'].squeeze(0).long(),'attention_mask': encoding['attention_mask'].squeeze(0).long(),'label': torch.tensor(int(item['is_sarcastic']), dtype=torch.long),}

# --- Model Class ---
class SarcasmBERT(nn.Module):
    def __init__(self):
        super(SarcasmBERT, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(p=0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        """
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
        Returns:
            logits: (batch_size, 2)
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        clsr = outputs.last_hidden_state[:, 0, :]
        return self.classifier(self.dropout(clsr))

# --- Training Loop ---
def train_loop(
    model: nn.Module, 
    dataloader: DataLoader, 
    device: torch.device, 
    lr: float, 
    epochs: int,
    **kwargs
) -> List[float]:
    
    # We use AdamW as the optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    model.to(device)
    model.train()
    
    loss_history = []
    
    for epoch in range(epochs):
        epoch_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].squeeze(-1).to(device)

            model.train()
            optimizer.zero_grad()
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

        # Feel free to pass additional arguments (like validation dataloader) 
        # through kwargs and perform validation here.
        
    return loss_history

# --- Main Execution ---
if __name__ == "__main__":
    # Check for GPU/MPS
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Data
    try:
        train_data = get_data('train.jsonl')
        valid_data = get_data('valid.jsonl')
    except NotImplementedError:
        print("Error: Implement get_data first.")
        exit(1)
    except FileNotFoundError:
        print("Error: Data files not found.")
        exit(1)

    # 2. Tokenizer & Dataset
    print("Initializing Tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    try:
        train_dataset = SarcasmDataset(train_data, tokenizer)
        valid_dataset = SarcasmDataset(valid_data, tokenizer)
        
        # Create DataLoaders
        batch_size = 8  # Change batch size if needed

        # DO NOT CHANGE THE FOLLOWING LINES
        if int(os.environ.get("GS_TESTING_BATCH_SIZE", "0")) > 0:
            batch_size = int(os.environ["GS_TESTING_BATCH_SIZE"])
        # END OF DO NOT CHANGE

        seed_env = os.environ.get("SEED", "")
        seed = int(seed_env) if seed_env != "" else None
        if seed is not None:
            import random as _random
            _random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        if seed is not None:
            gen = torch.Generator()
            gen.manual_seed(seed)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=gen)
            valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, generator=gen)
        else:
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    except Exception as e:
        print(f"Error initializing datasets: {e}")
        exit(1)

    # 3. Model
    try:
        model = SarcasmBERT()
    except NotImplementedError:
        print("Error: Implement SarcasmBERT first.")
        exit(1)

    # 4. Training
    # DO NOT CHANGE THE FOLLOWING LINES
    is_testing = os.environ.get("GS_TESTING", "0") == "1"
    checkpoint_path = "checkpoint.pt"
    if is_testing:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.to(device)
    else:
        print("Starting Training...")
        # END OF DO NOT CHANGE
        try:
            lr = 3e-5
            epochs = 3
            train_loop(model, train_loader, device, lr, epochs)
        except NotImplementedError:
            print("Error: Implement train_loop.")
            exit(1)

        # DO NOT MODIFY THE FOLLOWING LINES
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Model checkpoint saved to {checkpoint_path}")
        # END OF DO NOT CHANGE

    """
        YOUR ADDITIONAL CODE BELOW (DO NOT DELETE THIS COMMENT)
    """