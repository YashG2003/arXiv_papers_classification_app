import argparse
import mlflow
import pandas as pd
import os
from pathlib import Path
import json
import itertools

from config import Config
from data_processing.preprocessing import TextPreprocessor
from model.bert_trainer import BERTTrainer

def parse_args():
    parser = argparse.ArgumentParser(description='Train BERT model on arXiv data')
    parser.add_argument('--data_version', type=str, default='v1', help='Data version to use')
    parser.add_argument('--run_experiments', action='store_true', help='Run multiple experiments with different hyperparameters')
    return parser.parse_args()

def setup_mlflow():
    # Set up MLflow tracking
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("arxiv_classification_7")

def run_experiments(data_version):
    """Run multiple experiments with different hyperparameter combinations"""
    # Hyperparameter grid
    param_grid = {
        "max_length": [192],
        "batch_size": [32], #[32, 64],
        "learning_rate": [4e-5] #[3e-5, 4e-5]
    }
    
    # Generate all combinations
    param_combinations = list(itertools.product(
        param_grid["max_length"],
        param_grid["batch_size"],
        param_grid["learning_rate"]
    ))
    
    best_val_acc = 0
    best_run_id = None
    best_params = None
    
    print(f"Running {len(param_combinations)} experiments with different hyperparameters")
    
    for max_length, batch_size, lr in param_combinations:
        # Update config for this run
        Config.MAX_LENGTH = max_length
        Config.BATCH_SIZE = batch_size
        Config.LEARNING_RATE = lr
        
        print(f"\nExperiment with: MAX_LENGTH={max_length}, BATCH_SIZE={batch_size}, LEARNING_RATE={lr}")
        
        # Train with these parameters
        run_id, metrics = train_model(
            data_version, 
            experiment_name=f"length_{max_length}_batch_{batch_size}_lr_{lr}"
        )
        
        # Track best model
        if metrics["val_accuracy"] > best_val_acc:
            best_val_acc = metrics["val_accuracy"]
            best_run_id = run_id
            best_params = {
                "max_length": max_length,
                "batch_size": batch_size,
                "learning_rate": lr
            }
    
    print(f"\nBest validation accuracy: {best_val_acc:.4f}")
    print(f"Best parameters: {best_params}")
    print(f"Best run ID: {best_run_id}")
    
    # Save best model info
    metrics_dir = Path("models/metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path("models/initial_model")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    with open(metrics_dir / "best_model_metrics.json", "w") as f:
        json.dump({
            "best_val_accuracy": best_val_acc,
            "best_parameters": best_params,
            "best_run_id": best_run_id,
            "model_uri": f"runs:/{best_run_id}/model"
        }, f)
    
    # Save reference to the best MLflow model
    with open(model_dir / "model_info.json", "w") as f:
        json.dump({
            "run_id": best_run_id,
            "model_uri": f"runs:/{best_run_id}/model"
        }, f)
    
    return best_run_id, best_params

def train_model(data_version, experiment_name=None):
    """Train a single model with current config settings"""
    # Create directories
    metrics_dir = Path("models/metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path("models/initial_model")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    data_dir = Config.PROCESSED_DATA_DIR / data_version
    train_df = pd.read_csv(data_dir / "train.csv")
    val_df = pd.read_csv(data_dir / "val.csv")
    test_df = pd.read_csv(data_dir / "test.csv")  # Also load test data
    
    # Preprocess data
    preprocessor = TextPreprocessor()
    train_df = preprocessor.preprocess_for_bert(train_df)
    val_df = preprocessor.preprocess_for_bert(val_df)
    test_df = preprocessor.preprocess_for_bert(test_df)
    
    # Tokenize
    train_encodings = preprocessor.tokenize_data(train_df)
    val_encodings = preprocessor.tokenize_data(val_df)
    test_encodings = preprocessor.tokenize_data(test_df)
    
    # Train
    trainer = BERTTrainer(num_labels=len(Config.CATEGORY_MAP))
    trainer.load_model()  # Initialize a new model
    
    # Create data loaders
    train_loader = trainer.create_data_loader(
        train_encodings,
        train_df['label_idx'].tolist(),
        Config.BATCH_SIZE
    )
    
    val_loader = trainer.create_data_loader(
        val_encodings,
        val_df['label_idx'].tolist(),
        Config.BATCH_SIZE
    )
    
    test_loader = trainer.create_data_loader(
        test_encodings,
        test_df['label_idx'].tolist(),
        Config.BATCH_SIZE
    )
    
    # Train and get run ID and metrics
    run_id, metrics = trainer.train(
        train_loader, 
        val_loader, 
        epochs=Config.EPOCHS,
        experiment_name=experiment_name or "arxiv_classification_7"
    )
    
    # Evaluate on test set after training
    test_acc, test_f1, test_loss = trainer.evaluate(test_loader, return_loss=True)
    print(f"Test Accuracy: {test_acc:.4f}, Test F1: {test_f1:.4f}")
    
    # Log test metrics to the same MLflow run
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics({
            "test_accuracy": test_acc,
            "test_f1": test_f1,
            "test_loss": test_loss
        })
        # Update metrics dictionary with test metrics
        metrics.update({
            "test_accuracy": test_acc,
            "test_f1": test_f1,
            "test_loss": test_loss
        })
    
    # Save metrics for DVC
    run_metrics = {
        "data_version": data_version,
        "run_id": run_id,
        "model": Config.BERT_MODEL_NAME,
        "epochs": Config.EPOCHS,
        "max_length": Config.MAX_LENGTH,
        "batch_size": Config.BATCH_SIZE,
        "learning_rate": Config.LEARNING_RATE,
        **metrics  # Include all training metrics
    }
    
    with open(metrics_dir / "train_metrics.json", "w") as f:
        json.dump(run_metrics, f)
    
    # If this is a single run, save model reference
    if not experiment_name:
        with open(model_dir / "model_info.json", "w") as f:
            json.dump({
                "run_id": run_id,
                "model_uri": f"runs:/{run_id}/model"
            }, f)
    
    print(f"Training completed. Run ID: {run_id}")
    return run_id, metrics

if __name__ == "__main__":
    Config.setup()
    args = parse_args()
    setup_mlflow()
    
    if args.run_experiments:
        run_experiments(args.data_version)
    else:
        train_model(args.data_version)
