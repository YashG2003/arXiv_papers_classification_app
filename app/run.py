# app/run.py
import subprocess
import threading
import time
import os
import sys

def run_backend():
    """Run the FastAPI backend server"""
    print("Starting FastAPI backend server...")
    subprocess.run([
        "uvicorn", 
        "app.backend.main:app", 
        "--host", "0.0.0.0", 
        "--port", "8000"
    ])

def run_frontend():
    """Run the Streamlit frontend server"""
    print("Starting Streamlit frontend server...")
    subprocess.run([
        "streamlit", 
        "run", 
        "app/frontend/app.py",
        "--server.port", "8501"
    ])

def run_mlflow_server():
    """Ensure MLflow server is running"""
    print("Checking MLflow server...")
    try:
        # Try connecting to MLflow server
        import requests
        requests.get("http://localhost:5000/")
        print("MLflow server is already running.")
    except:
        print("Starting MLflow server...")
        subprocess.run([
            "mlflow", 
            "server",
            "--host", "0.0.0.0",
            "--port", "5000"
        ])

if __name__ == "__main__":
    # Start MLflow first with longer delay
    mlflow_thread = threading.Thread(target=run_mlflow_server)
    mlflow_thread.start()
    time.sleep(5)  # Increased delay for MLflow
    
    # Then start backend and frontend
    backend_thread = threading.Thread(target=run_backend)
    frontend_thread = threading.Thread(target=run_frontend)
    
    backend_thread.start()
    time.sleep(2)
    
    frontend_thread.start()
    
    # Keep threads alive
    mlflow_thread.join()
    backend_thread.join()
    frontend_thread.join()
