import argparse
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
import os
import json
from pathlib import Path
import shutil

from config import Config
from data_processing.preprocessing import TextPreprocessor
from model.bert_trainer import BERTTrainer

# Path to feedback CSV file
FEEDBACK_CSV_PATH = Path("data/feedback/user_corrections.csv")
FEEDBACK_THRESHOLD = 2

def parse_args():
    parser = argparse.ArgumentParser(description='Fine-tune BERT model on arXiv data')
    parser.add_argument('--data_version', type=str, default='v2', help='Data version to use for fine-tuning')
    parser.add_argument('--check_feedback', action='store_true', help='Check if user feedback threshold is reached')
    return parser.parse_args()

def setup_mlflow():
    # Set up MLflow tracking using server
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("arxiv_classification_7")

def get_best_model_from_mlflow():
    client = MlflowClient()
    experiment = client.get_experiment_by_name("arxiv_classification_7")
    if not experiment:
        raise ValueError("Experiment 'arxiv_classification_7' not found. Run training first.")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="",
        run_view_type=mlflow.entities.ViewType.ACTIVE_ONLY,
        max_results=500
    )
    if not runs:
        raise ValueError("No runs found in experiment 'arxiv_classification_7'")

    best_run = None
    best_test_acc = -1
    for run in runs:
        metrics = run.data.metrics
        if 'test_accuracy' in metrics and metrics['test_accuracy'] > best_test_acc:
            best_test_acc = metrics['test_accuracy']
            best_run = run

    if not best_run:
        raise ValueError("No runs with test_accuracy found")

    pyfunc_uri = f"runs:/{best_run.info.run_id}/model"
    print(f"Using best model from MLflow run: {best_run.info.run_id}")
    print(f"Best test accuracy: {best_test_acc:.4f}")
    return pyfunc_uri, best_run.info.run_id


def check_feedback_threshold():
    """Check if user feedback has reached the threshold for retraining"""
    if not FEEDBACK_CSV_PATH.exists():
        return False
    
    feedback_df = pd.read_csv(FEEDBACK_CSV_PATH)
    feedback_count = len(feedback_df)
    
    print(f"Current feedback count: {feedback_count}/{FEEDBACK_THRESHOLD}")
    return feedback_count >= FEEDBACK_THRESHOLD

def fine_tune_with_feedback():
    # Create metrics directory
    metrics_dir = Path("models/metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Load feedback data
    feedback_df = pd.read_csv(FEEDBACK_CSV_PATH)
    print(f"Loaded {len(feedback_df)} feedback entries for retraining")

    # Load v1 train data
    v1_train_df = pd.read_csv(Config.PROCESSED_DATA_DIR / "v1" / "train.csv")

    # Combine feedback and v1 train data
    train_df = pd.concat([v1_train_df, feedback_df], ignore_index=True)

    # Get validation and test data from v1
    val_df = pd.read_csv(Config.PROCESSED_DATA_DIR / "v1" / "val.csv")
    test_df = pd.read_csv(Config.PROCESSED_DATA_DIR / "v1" / "test.csv")

    # Preprocess data
    preprocessor = TextPreprocessor()
    train_df = preprocessor.preprocess_for_bert(train_df)
    val_df = preprocessor.preprocess_for_bert(val_df)
    test_df = preprocessor.preprocess_for_bert(test_df)

    # Tokenize
    train_encodings = preprocessor.tokenize_data(train_df)
    val_encodings = preprocessor.tokenize_data(val_df)
    test_encodings = preprocessor.tokenize_data(test_df)

    # Initialize a new model for training from scratch
    trainer = BERTTrainer(num_labels=len(Config.CATEGORY_MAP))
    trainer.load_model()  # This will initialize a new model

    # Create data loaders
    train_loader = trainer.create_data_loader(
        train_encodings, train_df['label_idx'].tolist(), Config.BATCH_SIZE)
    val_loader = trainer.create_data_loader(
        val_encodings, val_df['label_idx'].tolist(), Config.BATCH_SIZE)
    test_loader = trainer.create_data_loader(
        test_encodings, test_df['label_idx'].tolist(), Config.BATCH_SIZE)

    # Train and log model
    run_id, metrics = trainer.train(
        train_loader, val_loader, epochs=Config.EPOCHS, experiment_name="feedback_fine_tuned_model"
    )

    # Evaluate on test set after training
    test_acc, test_f1, test_loss = trainer.evaluate(test_loader, return_loss=True)
    print(f"Test Accuracy: {test_acc:.4f}, Test F1: {test_f1:.4f}")

    # Log test metrics to the same MLflow run
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics({
            "test_accuracy": test_acc,
            "test_f1": test_f1,
            "test_loss": test_loss,
            "feedback_entries": len(feedback_df)
        })

    # Save metrics for DVC
    run_metrics = {
        "data_source": "user_feedback",
        "feedback_entries": len(feedback_df),
        "run_id": run_id,
        "model": Config.BERT_MODEL_NAME,
        "epochs": Config.EPOCHS,
        "test_accuracy": test_acc,
        "test_f1": test_f1,
        **metrics
    }

    with open(metrics_dir / "feedback_fine_tune_metrics.json", "w") as f:
        json.dump(run_metrics, f)

    # Archive the feedback file after successful training
    archive_dir = Path("data/feedback/archive")
    archive_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(FEEDBACK_CSV_PATH, archive_dir / f"user_corrections_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv")
    # Clear the feedback file (but keep the header)
    feedback_df.iloc[0:0].to_csv(FEEDBACK_CSV_PATH, index=False)

    print(f"Feedback fine-tuning completed. Run ID: {run_id}")
    return run_id, metrics


def fine_tune_model(data_version):
    import shutil
    # Create metrics directory
    metrics_dir = Path("models/metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Load v1 train data and user feedback corrections
    v1_train_path = Config.PROCESSED_DATA_DIR / "v1" / "train.csv"
    feedback_path = Path("data/feedback/user_corrections.csv")
    train_dfs = []
    if v1_train_path.exists():
        train_dfs.append(pd.read_csv(v1_train_path))
    if feedback_path.exists() and feedback_path.stat().st_size > 0:
        feedback_df = pd.read_csv(feedback_path)
        if not feedback_df.empty:
            train_dfs.append(feedback_df)
    if not train_dfs:
        raise ValueError("No training data found for fine-tuning.")
    train_df = pd.concat(train_dfs, ignore_index=True)

    # Load validation and test data from v1
    val_df = pd.read_csv(Config.PROCESSED_DATA_DIR / "v1" / "val.csv")
    test_df = pd.read_csv(Config.PROCESSED_DATA_DIR / "v1" / "test.csv")

    # Preprocess data
    preprocessor = TextPreprocessor()
    train_df = preprocessor.preprocess_for_bert(train_df)
    val_df = preprocessor.preprocess_for_bert(val_df)
    test_df = preprocessor.preprocess_for_bert(test_df)

    # Tokenize
    train_encodings = preprocessor.tokenize_data(train_df)
    val_encodings = preprocessor.tokenize_data(val_df)
    test_encodings = preprocessor.tokenize_data(test_df)

    # Initialize a new model for training from scratch
    trainer = BERTTrainer(num_labels=len(Config.CATEGORY_MAP))
    trainer.load_model()  # No model_path: train from scratch

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

    # Train and log model
    run_id, metrics = trainer.train(
        train_loader,
        val_loader,
        epochs=Config.EPOCHS,
        experiment_name="fine_tuned_model"
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
        **metrics
    }

    with open(metrics_dir / "fine_tune_metrics.json", "w") as f:
        json.dump(run_metrics, f)

    print(f"Fine-tuning completed. Run ID: {run_id}")
    return run_id, metrics


if __name__ == "__main__":
    Config.setup()
    args = parse_args()
    setup_mlflow()
    
    # Check if we need to fine-tune based on user feedback
    if args.check_feedback and check_feedback_threshold():
        print("Feedback threshold reached. Starting fine-tuning with user feedback...")
        fine_tune_with_feedback()
    else:
        # Regular fine-tuning
        fine_tune_model(args.data_version)
