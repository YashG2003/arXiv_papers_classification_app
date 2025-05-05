import torch
import os
import shutil
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
import mlflow
from mlflow.pyfunc import PythonModel
from data_processing.preprocessing import TextPreprocessor

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

class BertTextClassifier(PythonModel):
    """Custom MLflow PythonModel that includes text preprocessing. It has custom predict function."""
    
    def load_context(self, context):
        """Load the PyTorch model and initialize preprocessor"""
        self.device = torch.device("cpu")
        
        # Load the model architecture
        self.model = BertForSequenceClassification.from_pretrained(Config.BERT_MODEL_NAME, num_labels=len(Config.CATEGORY_MAP))
        
        # Load the trained weights from the artifact
        self.model.load_state_dict(torch.load(context.artifacts["bert_model"], map_location=self.device))   
        self.model.eval()
        
        # Initialize the preprocessor
        self.preprocessor = TextPreprocessor()

    def predict(self, context, model_input):
        """Make predictions with preprocessing included
        
        Args:
            context: MLflow context
            model_input: DataFrame with 'text' column
            
        Returns:
            numpy array of predicted class indices
        """
        # Extract text from DataFrame
        texts = model_input["text"].tolist()
        
        # Tokenize using the same preprocessor used in training
        encodings = self.preprocessor.tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=Config.MAX_LENGTH,
            return_tensors="pt"
        )
        
        # Move to device and predict
        inputs = {k: v.to(self.device) for k, v in encodings.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()
        
        return preds

class BERTTrainer:
    def __init__(self, num_labels):
        self.device = torch.device("cpu")
        self.model = None
        self.num_labels = num_labels
    '''
    def load_model(self, model_path=None):
        """
        Load model from path or initialize a new model
        """
        if model_path:
            # Load from MLflow or local path
            self.model = mlflow.pyfunc.load_model(model_path)
            print(f"Loaded model from {model_path}")
        else:
            # Initialize a new model
            self.model = BertForSequenceClassification.from_pretrained(
                Config.BERT_MODEL_NAME,
                num_labels=self.num_labels
            )
            # Freeze only embeddings for better speed/performance balance
            for param in self.model.bert.embeddings.parameters():
                param.requires_grad = False
        
        return self.model
    '''
    
    def load_model(self, model_path=None):
        import mlflow
        if model_path:
            self.model = mlflow.pyfunc.load_model(model_path)
            print(f"Loaded PyFunc model from {model_path}")
        else:
            self.model = BertForSequenceClassification.from_pretrained(
                Config.BERT_MODEL_NAME,
                num_labels=self.num_labels
            )
            # Optionally freeze embeddings
            for param in self.model.bert.embeddings.parameters():
                param.requires_grad = False
            self.model = self.model.to(self.device)
            
        return self.model

    
    def create_data_loader(self, encodings, labels, batch_size):
        dataset = ArXivDataset(encodings, labels)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,  # Use CPU parallel loading
            pin_memory=False  # Disable for pure CPU
        )

    def train(self, train_loader, val_loader, epochs=Config.EPOCHS, experiment_name="arxiv_classification_7"):
        """
        Train the model with MLflow tracking
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call load_model() first.")
        
        # Start MLflow run
        with mlflow.start_run(run_name=experiment_name) as run:
            # Log parameters
            mlflow.log_params({
                "model_name": Config.BERT_MODEL_NAME,
                "batch_size": Config.BATCH_SIZE,
                "epochs": epochs,
                "learning_rate": Config.LEARNING_RATE,
                "max_length": Config.MAX_LENGTH
            })
            
            optimizer = AdamW(self.model.parameters(), lr=Config.LEARNING_RATE)
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=0,
                num_training_steps=len(train_loader) * epochs
            )

            # Track metrics across epochs
            metrics = {}

            for epoch in range(epochs):
                self.model.train()
                total_loss = 0
                all_preds = []
                all_labels = []
                loop = tqdm(train_loader, leave=True)
                
                for batch in loop:
                    optimizer.zero_grad()
                    
                    inputs = {
                        k: v.to(self.device) for k, v in batch.items()
                    }
                    
                    outputs = self.model(**inputs)
                    loss = outputs.loss
                    
                    # Collect predictions for training accuracy
                    preds = torch.argmax(outputs.logits, dim=1)
                    all_preds.extend(preds.cpu().tolist())
                    all_labels.extend(inputs['labels'].cpu().tolist())
                    
                    total_loss += loss.item()
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    
                    loop.set_description(f"Epoch [{epoch+1}/{epochs}]")
                    loop.set_postfix(loss=loss.item())
                
                # Calculate training metrics
                avg_loss = total_loss / len(train_loader)
                train_acc = accuracy_score(all_labels, all_preds)
                train_f1 = f1_score(all_labels, all_preds, average='weighted')
                
                # Calculate validation metrics
                val_acc, val_f1, val_loss = self.evaluate(val_loader, return_loss=True)
                
                print(f"\nEpoch {epoch+1}/{epochs}")
                print(f"Train Loss: {avg_loss:.4f}, Train Acc: {train_acc:.4f}")
                print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
                
                # Log metrics to MLflow
                mlflow.log_metrics({
                    "train_loss": avg_loss,
                    "train_accuracy": train_acc,
                    "train_f1": train_f1,
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                    "val_f1": val_f1
                }, step=epoch)
                
                # Store the metrics for the final epoch
                if epoch == epochs - 1:
                    metrics = {
                        "train_loss": avg_loss,
                        "train_accuracy": train_acc,
                        "train_f1": train_f1,
                        "val_loss": val_loss,
                        "val_accuracy": val_acc,
                        "val_f1": val_f1
                    }
            
            # Save the PyTorch model to a file before logging the PyFunc model
            torch.save(self.model.state_dict(), "bert_model.pt")

            mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=BertTextClassifier(),
                artifacts={"bert_model": "bert_model.pt"},
                code_path=["data_processing/preprocessing.py", "model/bert_trainer.py"],
                conda_env=None
            )
        
            # Return the run ID and metrics
            return run.info.run_id, metrics

    def evaluate(self, data_loader, return_loss=False):
        """
        Evaluate the model on validation or test data
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call load_model() first.")
            
        predictions = []
        true_labels = []
        total_loss = 0
        
        with torch.no_grad():
            for batch in data_loader:
                inputs = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**inputs)
                
                if return_loss:
                    total_loss += outputs.loss.item()
                    
                preds = torch.argmax(outputs.logits, dim=1)
                predictions.extend(preds.cpu().tolist())
                true_labels.extend(inputs['labels'].cpu().tolist())
        
        accuracy = accuracy_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions, average='weighted')
        
        if return_loss:
            avg_loss = total_loss / len(data_loader)
            return accuracy, f1, avg_loss
        
        return accuracy, f1
