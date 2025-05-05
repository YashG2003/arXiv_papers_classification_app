# app/backend/tasks.py
import pandas as pd
import subprocess
import time
from pathlib import Path
import mlflow
from mlflow.tracking import MlflowClient
import os
import json
from contextlib import contextmanager
from filelock import FileLock

# Paths and thresholds
FEEDBACK_THRESHOLD = 2
FEEDBACK_CSV_PATH = Path("data/feedback/user_corrections.csv")
FEEDBACK_LOCK_PATH = FEEDBACK_CSV_PATH.with_suffix('.lock')
MODEL_INFO_PATH = Path("models/serving/current_model_info.json")
MODEL_LOCK_PATH = MODEL_INFO_PATH.with_suffix('.lock')

# Create directories if they don't exist
Path("models/serving").mkdir(parents=True, exist_ok=True)
Path("data/feedback").mkdir(parents=True, exist_ok=True)

def check_feedback():
    """Check if feedback threshold is reached and trigger retraining if needed"""
    if not FEEDBACK_CSV_PATH.exists():
        print(f"Feedback file not found at {FEEDBACK_CSV_PATH}")
        return False

    try:
        with FileLock(str(FEEDBACK_LOCK_PATH)):
            feedback_df = pd.read_csv(FEEDBACK_CSV_PATH)
            feedback_count = len(feedback_df)
            print(f"[TASK] Current feedback count: {feedback_count}/{FEEDBACK_THRESHOLD}")

            if feedback_count >= FEEDBACK_THRESHOLD:
                print("[TASK] Feedback threshold reached. Triggering retraining...")
                subprocess.run(["dvc", "repro", "check_feedback"], check=True)
                return True
        return False
    except Exception as e:
        print(f"[TASK] Error checking feedback: {str(e)}")
        return False

def get_best_model_info():
    """Get info about the best model in MLflow"""
    try:
        # Set up MLflow client
        mlflow.set_tracking_uri("http://localhost:5000")
        client = MlflowClient()
        
        # Get experiment by name - check all relevant experiments
        experiments = []
        for exp_name in ["arxiv_classification_7"]:
            exp = client.get_experiment_by_name(exp_name)
            if exp:
                experiments.append(exp.experiment_id)
        
        if not experiments:
            print("[TASK] No experiments found in MLflow")
            return None
        
        # Get all runs with test_accuracy metric
        all_runs = []
        for exp_id in experiments:
            runs = client.search_runs(
                experiment_ids=[exp_id],
                filter_string="metrics.test_accuracy > 0",
                order_by=["metrics.test_accuracy DESC"]
            )
            all_runs.extend(runs)
        
        if not all_runs:
            print("[TASK] No runs with test_accuracy found")
            return None
        
        # Sort all runs by test_accuracy
        best_runs = sorted(all_runs, key=lambda r: r.data.metrics.get("test_accuracy", 0), reverse=True)
        best_run = best_runs[0]
        
        return {
            "run_id": best_run.info.run_id,
            "model_uri": f"runs:/{best_run.info.run_id}/model",
            "test_accuracy": best_run.data.metrics.get("test_accuracy", 0),
            "timestamp": best_run.info.start_time
        }
    except Exception as e:
        print(f"[TASK] Error getting best model info: {str(e)}")
        return None

def monitor_mlflow_models():
    """Monitor MLflow for new models and update serving model if better one found"""
    try:
        print("[TASK] Checking for better models in MLflow...")
        best_model_info = get_best_model_info()
        
        if not best_model_info:
            print("[TASK] Could not get best model info")
            return False
        
        with FileLock(str(MODEL_LOCK_PATH)):
            # Check if we already have a serving model
            if not MODEL_INFO_PATH.exists():
                # No current model, use the best one
                print(f"[TASK] No current model found. Using best model: {best_model_info['run_id']}")
                with open(MODEL_INFO_PATH, 'w') as f:
                    json.dump(best_model_info, f)
                return True
            
            # Compare with current model
            with open(MODEL_INFO_PATH, 'r') as f:
                current_model = json.load(f)
            
            current_accuracy = current_model.get("test_accuracy", 0)
            new_accuracy = best_model_info.get("test_accuracy", 0)
            
            # If new model is better by at least 0.1% or it's a newer version of the same accuracy
            if (new_accuracy > current_accuracy + 0.001 or 
                (abs(new_accuracy - current_accuracy) < 0.001 and 
                 best_model_info["timestamp"] > current_model.get("timestamp", 0))):
                
                print(f"[TASK] Better model found! Updating from {current_accuracy:.4f} to {new_accuracy:.4f}")
                with open(MODEL_INFO_PATH, 'w') as f:
                    json.dump(best_model_info, f)
                return True
            
            print(f"[TASK] Current model ({current_accuracy:.4f}) is still the best")
            return False
    except Exception as e:
        print(f"[TASK] Error monitoring models: {str(e)}")
        return False
