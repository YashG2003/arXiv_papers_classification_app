import torch
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    BertForSequenceClassification,
    get_linear_schedule_with_warmup
)
from config import Config
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

class ArXivDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
        
    def __getitem__(self, idx):
        item = {
            key: val[idx] for key, val in self.encodings.items()
        }
        item['labels'] = torch.tensor(self.labels[idx])
        return item
        
    def __len__(self):
        return len(self.labels)

class BERTTrainer:
    def __init__(self, num_labels):
        self.device = torch.device("cpu")
        self.model = BertForSequenceClassification.from_pretrained(
            Config.BERT_MODEL_NAME,
            num_labels=num_labels
        ).to(self.device)
        
        # Freeze only embeddings (not layers) for better speed/performance balance
        for param in self.model.bert.embeddings.parameters():
            param.requires_grad = False
        
    def create_data_loader(self, encodings, labels, batch_size):
        dataset = ArXivDataset(encodings, labels)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,  # Use CPU parallel loading
            pin_memory=False  # Disable for pure CPU
        )

    def train(self, train_loader, val_loader, epochs=Config.EPOCHS):
        optimizer = AdamW(self.model.parameters(), lr=Config.LEARNING_RATE)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=len(train_loader) * epochs
        )

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0

            loop = tqdm(train_loader, leave=True)
            for batch in loop:
                optimizer.zero_grad()
                inputs = {
                    k: v.to(self.device) for k, v in batch.items()
                }
                outputs = self.model(**inputs)
                loss = outputs.loss
                total_loss += loss.item()
                loss.backward()
                optimizer.step()
                scheduler.step()

                loop.set_description(f"Epoch [{epoch+1}/{epochs}]")
                loop.set_postfix(loss=loss.item())

            avg_loss = total_loss / len(train_loader)
            val_acc, val_f1 = self.evaluate(val_loader)

            print(f"\nEpoch {epoch+1}/{epochs}")
            print(f"Train Loss: {avg_loss:.4f}")
            print(f"Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")

            
    def evaluate(self, data_loader):
        self.model.eval()
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for batch in data_loader:
                inputs = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**inputs)
                preds = torch.argmax(outputs.logits, dim=1)
                predictions.extend(preds.cpu().tolist())
                true_labels.extend(inputs['labels'].cpu().tolist())
        
        # Debug prints
        print("Sample predictions:", predictions[:10])
        print("Sample true labels:", true_labels[:10])
        
        return (
            accuracy_score(true_labels, predictions),
            f1_score(true_labels, predictions, average='weighted')
        )