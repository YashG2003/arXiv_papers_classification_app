from data_processing.versioning import DataVersioner
from data_processing.preprocessing import TextPreprocessor
from model.bert_trainer import BERTTrainer
from config import Config
import pandas as pd
import subprocess
import os
import shutil
from pathlib import Path

def manage_dvc_data(version: int, action: str = "checkout"):
    """Manage version data through DVC operations"""
    version_path = Config.PROCESSED_DATA_DIR / f"v{version}"
    
    if action == "checkout":
        # Checkout only specific version from DVC cache
        subprocess.run(["dvc", "checkout", str(version_path)], check=True)
    elif action == "clean":
        if version_path.exists():
            # Remove directory but keep DVC tracking
            shutil.rmtree(version_path)
            # Remove DVC file if exists
            dvc_file = version_path.with_suffix('.dvc')
            if dvc_file.exists():
                dvc_file.unlink()

def main():
    # Check DVC initialization
    if not Path(".dvc").exists():
        raise RuntimeError("DVC not initialized. Run 'dvc init' first")

    # Run full pipeline
    if not Config.PROCESSED_DATA_DIR.exists():
        print("Running DVC pipeline...")
        subprocess.run(["dvc", "repro"], check=True)
        
        # Cleanup processed data after pipeline
        shutil.rmtree(Config.PROCESSED_DATA_DIR)
        print("Cleaned up processed data after pipeline")

    # Training loop
    preprocessor = TextPreprocessor()
    
    for version in [1]:
        version_dir = Config.PROCESSED_DATA_DIR / f"v{version}"
        
        # Check and restore from DVC
        if not version_dir.exists():
            print(f"Restoring version {version} from DVC cache")
            manage_dvc_data(version, "checkout")
            
        try:
            # Load data
            train = pd.read_csv(version_dir / "train.csv")
            val = pd.read_csv(version_dir / "val.csv")
            
            # Preprocess and train
            train = preprocessor.preprocess_for_bert(train)
            val = preprocessor.preprocess_for_bert(val)
        
            # Tokenize
            train_encodings = preprocessor.tokenize_data(train)
            val_encodings = preprocessor.tokenize_data(val)
            
            # Create loaders (use label_idx directly)
            trainer = BERTTrainer(num_labels=len(Config.CATEGORY_MAP))  # Fixed to 10 classes
            
            train_loader = trainer.create_data_loader(
                train_encodings,
                train['label_idx'].tolist(),  # Use pre-mapped indices
                Config.BATCH_SIZE
            )
            
            val_loader = trainer.create_data_loader(
                val_encodings,
                val['label_idx'].tolist(),
                Config.BATCH_SIZE
            )
            
            # Train
            print(f"Training Version {version}")
            trainer.train(train_loader, val_loader, epochs=Config.EPOCHS)  # Increased epochs
            
        finally:
            # Cleanup after training
            manage_dvc_data(version, "clean")
            print(f"Cleaned up version {version} data")

if __name__ == "__main__":
    main()