# app/backend/mlflow_utils.py
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
import json
from pathlib import Path

'''
def get_best_model():
    """Get the best model from MLflow based on the current model info file"""
    
    mlflow.set_tracking_uri("http://localhost:5000")
    model_info_path = Path("models/serving/current_model_info.json")
    
    if model_info_path.exists():
        # Load model info from file
        with open(model_info_path, 'r') as f:
            model_info = json.load(f)
        
        # Use the model URI from the info file
        model_uri = model_info["model_uri"]
        print(f"Loading model from: {model_uri}")
        print(f"Model test accuracy: {model_info.get('test_accuracy', 'N/A')}")
        
        return mlflow.pyfunc.load_model(model_uri)
    else:
        # Fall back to finding the best model directly
        client = MlflowClient()
        
        # Get experiment by name
        experiment = client.get_experiment_by_name("arxiv_classification_5")
        if not experiment:
            raise ValueError("Experiment 'arxiv_classification_5' not found. Make sure MLflow server is running.")
        
        # Get all runs for this experiment
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="",
            order_by=["metrics.test_accuracy DESC"]
        )
        
        if not runs:
            raise ValueError("No runs found in MLflow")
        
        # Get the best run
        best_run = runs[0]
        
        # Load the model
        model_uri = f"runs:/{best_run.info.run_id}/pyfunc-model"
        print(f"Loading best model from run: {best_run.info.run_id}")
        print(f"Test accuracy: {best_run.data.metrics.get('test_accuracy', 'N/A')}")
        
        # Save this info for future reference
        with open(model_info_path, 'w') as f:
            json.dump({
                "run_id": best_run.info.run_id,
                "model_uri": model_uri,
                "test_accuracy": best_run.data.metrics.get("test_accuracy", 
                                                          best_run.data.metrics.get("test_accuracy", 0)),
                "timestamp": best_run.info.start_time
            }, f)
        
        return mlflow.pyfunc.load_model(model_uri)
'''

def get_best_model():
    mlflow.set_tracking_uri("http://localhost:5000")
    client = MlflowClient()
    experiment = client.get_experiment_by_name("arxiv_classification_7")
    if not experiment:
        raise ValueError("Experiment 'arxiv_classification_7' not found. Make sure MLflow server is running.")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="",
        order_by=["metrics.test_accuracy DESC"]
    )
    if not runs:
        raise ValueError("No runs found in MLflow")

    best_run = runs[0]
    pyfunc_uri = f"runs:/{best_run.info.run_id}/model"
    print(f"Loading best PyFunc model from run: {best_run.info.run_id}")
    print(f"Test accuracy: {best_run.data.metrics.get('test_accuracy', 'N/A')}")
    return mlflow.pyfunc.load_model(pyfunc_uri)


def get_category_mapping():
    """Get the mapping from category IDs to human-readable names"""
    # This is the mapping used in your application
    category_map = {
        'astro-ph': 'Astrophysics',
        'cond-mat': 'Condensed Matter Physics',
        'cs': 'Computer Science',
        'eess': 'Electrical Engineering and Systems Science',
        'hep-ph': 'High Energy Physics - Phenomenology',
        'hep-th': 'High Energy Physics - Theory',
        'math': 'Mathematics',
        'physics': 'Physics (General)',
        'quant-ph': 'Quantum Physics',
        'stat': 'Statistics'
    }
    
    # Create mapping from index to human-readable name
    label_map = {idx: name for idx, name in enumerate(category_map.values())}
    
    # Create reverse mapping (code to index)
    code_to_idx = {code: idx for idx, code in enumerate(category_map.keys())}
    
    return label_map, code_to_idx
