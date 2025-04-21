# arXiv Classifier MLOps Implementation Roadmap

## Phase 1: Project Setup & Data Pipeline

1. **Project Structure Setup**  

```
arxiv-mlops/
├── data/         # DVC-tracked raw/processed data
├── models/       # MLflow-tracked models
├── frontend/     # Web UI (Streamlit/Flask)
├── backend/      # FastAPI + Model serving
├── pipelines/    # Airflow/DVC data pipelines
├── monitoring/   # Prometheus + Grafana configs
└── docker/       # Docker compose setup
```

2. **Data Versioning**  
   - Initialize DVC: `dvc init`  
   - Create data pipeline stages:  
     ```yaml
     # dvc.yaml
     stages:
       preprocess:
         cmd: python preprocessing.py
         deps: [data/raw]
         outs: [data/processed]
     ```

3. **Data Engineering Pipeline**
   * Create Spark/Airflow pipeline for:
     * Data ingestion from arXiv API/CSV
     * Basic text cleaning
     * Train/val/test splits
   * **Alternative**: Simple DVC pipeline if using static dataset

## Phase 2: Model Development

1. **Experiment Tracking**
```python
import mlflow
mlflow.autolog()

with mlflow.start_run():
    # Your training code
    mlflow.log_param("model_name", "bert-tiny")
    mlflow.log_metric("val_acc", 0.82)
    mlflow.pytorch.log_model(model, "model")
```

2. **Model Serving Setup**
   * Create FastAPI endpoints:
```python
@app.post("/predict")
async def predict(text: str):
    return {"category": model(text)}
```

## Phase 3: Monitoring Infrastructure

1. **Prometheus Instrumentation**
   * Add metrics to FastAPI:
```python
from prometheus_client import Counter, generate_latest
API_REQUESTS = Counter('api_requests', 'Total API requests')

@app.get("/metrics")
async def metrics():
    return Response(generate_latest())
```

2. **Grafana Dashboard**
   * Create dashboard JSON monitoring:
     * API request rates
     * CPU/Memory usage
     * Model inference latency
     * Error rates

## Phase 4: Dockerization & Deployment

1. **Containerization**
   * `Dockerfile` for:
     * Frontend (Node/React)
     * Backend (FastAPI + Model)
     * Monitoring (Prometheus+Grafana)
   * `docker-compose.yml`:
```yaml
services:
  backend:
    build: ./backend
  frontend:
    build: ./frontend
  prometheus:
    image: prom/prometheus
  grafana:
    image: grafana/grafana
```

## Phase 5: CI/CD & Testing

1. **Git+DVC Workflow**
   * Data versioning:
```bash
dvc add data/raw
git add .gitignore data/raw.dvc
```

   * CI pipeline with DVC DAG:
```yaml
# .github/workflows/ci.yml
jobs:
  train:
    runs-on: ubuntu-latest
    steps:
      - dvc pull
      - python train.py
```

## Phase 6: Documentation & Finalization

1. **Required Docs**
   * Architecture diagram (Mermaid/Diagrams.net)
   * API spec (OpenAPI/Swagger)
   * Test plan with 20+ test cases
   * User manual (1-page PDF)
   * Deployment guide

## Phase 7: Model Retraining

1. **Feedback Loop**
   * Implement endpoint for user feedback:
```python
@app.post("/feedback")
async def feedback(prediction: str, correct_label: str):
    # Log to DB → Trigger retraining
```

   * Airflow pipeline for periodic retraining