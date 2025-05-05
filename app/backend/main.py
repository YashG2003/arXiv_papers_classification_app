# app/backend/main.py
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import traceback
import time
import io
from pathlib import Path
from contextlib import asynccontextmanager
from apscheduler.schedulers.background import BackgroundScheduler
import atexit
from filelock import FileLock
import json
import mlflow
import time
import threading

from app.backend.mlflow_utils import get_best_model, get_category_mapping
from app.backend.pdf_extractor import extract_from_pdf
from app.backend.tasks import check_feedback, monitor_mlflow_models

from prometheus_client import Counter, Histogram, make_asgi_app, Gauge

# Define data models
class PaperInput(BaseModel):
    title: str
    abstract: str

class PredictionResponse(BaseModel):
    category: str
    category_id: int
    confidence: float
    processing_time: float
    title: str
    abstract: str

class SamplePaper(BaseModel):
    title: str
    abstract: str
    true_category: str
    predicted_category: str
    correct: bool

class SampleResponse(BaseModel):
    samples_by_category: Dict[str, List[SamplePaper]]
    total_count: int
    accuracy: float
    
class FeedbackInput(BaseModel):
    title: str
    abstract: str
    predicted_category: str
    correct_category: str

# Initialize scheduler
scheduler = BackgroundScheduler()
        

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup code (before yield)
    global model
    try:
        model = get_best_model()
        print("Model loaded successfully!")
        
        # Start the schedulers
        print("Starting scheduled tasks...")
        
        # Schedule the feedback check task (every 2 minutes)
        scheduler.add_job(
            check_feedback,
            'interval',
            minutes=2,
            id='feedback_check_job',
            replace_existing=True
        )
        
        # Schedule the model monitoring task (every 2 minutes)
        scheduler.add_job(
            monitor_mlflow_models,
            'interval',
            minutes=2,
            id='model_monitor_job',
            replace_existing=True
        )
        
        # Start the scheduler
        scheduler.start()
        
        # Register a function to shut down the scheduler when the app exits
        atexit.register(lambda: scheduler.shutdown())
        
    except Exception as e:
        print(f"Error during startup: {str(e)}")
        print(traceback.format_exc())
    
    yield  # This is where the application runs
    
    # Shutdown code (after yield)
    print("Shutting down and cleaning up resources")
    
    # Shutdown the scheduler
    scheduler.shutdown()
    
# Initialize metrics (add before route definitions)
CLASSIFY_PDF_COUNTER = Counter(
    'classify_pdf_requests_total',
    'Total classify PDF requests',
    ['status']
)
CLASSIFY_PDF_LATENCY = Histogram(
    'classify_pdf_latency_seconds',
    'Classification latency',
    buckets=[0.1, 0.5, 1, 2, 5]
)

FEEDBACK_COUNTER = Counter(
    'feedback_requests_total',
    'Total feedback submissions',
    ['status']
)

FEEDBACK_LATENCY = Histogram(
    'feedback_latency_seconds',
    'Feedback latency',
    buckets=[0.1, 0.5, 1, 2, 5]
)

SAMPLES_COUNTER = Counter(
    'samples_requests_total',
    'Total sample requests',
    ['status']
)

SAMPLES_LATENCY = Histogram(
    'samples_latency_seconds',
    'samples latency',
    buckets=[0.1, 0.5, 1, 2, 5]
)

UNIQUE_USERS = Counter(
    "unique_users_total",
    "Unique users by IP",
    ["ip"]
)

# Initialize FastAPI app
app = FastAPI(title="arXiv Paper Classifier API", lifespan=lifespan)

# Add metrics endpoint (uncomment and modify)
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

@app.middleware("http")
async def track_unique_users(request: Request, call_next):
    ip = request.client.host
    UNIQUE_USERS.labels(ip=ip).inc()
    response = await call_next(request)
    return response

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = None
label_map, code_to_idx = get_category_mapping()

@app.get("/")
def root():
    return {"message": "arXiv Paper Classifier API"}

'''
@app.post("/classify_pdf", response_model=PredictionResponse)
async def classify_pdf(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = time.time()
    
    try:
        # Read the uploaded PDF file
        contents = await file.read()
        pdf_bytes = io.BytesIO(contents)
        
        # Extract title and abstract
        title, abstract = extract_from_pdf(pdf_bytes)
        
        if not title or not abstract:
            raise HTTPException(status_code=400, detail="Could not extract title and abstract from PDF")
        
        # Prepare input for prediction
        input_text = f"{title} {abstract}"
        
        # Make prediction using existing logic
        prediction = model.predict(pd.DataFrame({"text": [input_text]}))
        
        # Process prediction result
        if isinstance(prediction, np.ndarray):
            if len(prediction.shape) == 2:  # Logits output
                predicted_class = np.argmax(prediction[0])
                confidence = float(np.max(prediction[0]))
            else:  # Class index output
                predicted_class = int(prediction[0])
                confidence = 1.0
        else:
            predicted_class = int(prediction)
            confidence = 1.0
        
        processing_time = time.time() - start_time
        
        return PredictionResponse(
            category=label_map[predicted_class],
            category_id=predicted_class,
            confidence=confidence,
            processing_time=processing_time,
            title=title,  
            abstract=abstract  
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF processing error: {str(e)}")
'''
@app.post("/classify_pdf", response_model=PredictionResponse)
async def classify_pdf(file: UploadFile = File(...)):
    start_time = time.time()
    try:
        if model is None:
            CLASSIFY_PDF_COUNTER.labels(status='503').inc()
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Existing processing logic
        contents = await file.read()
        pdf_bytes = io.BytesIO(contents)
        title, abstract = extract_from_pdf(pdf_bytes)
        
        if not title or not abstract:
            CLASSIFY_PDF_COUNTER.labels(status='400').inc()
            raise HTTPException(status_code=400, detail="Extraction failed")
        
        # Prediction logic
        input_text = f"{title} {abstract}"
        prediction = model.predict(pd.DataFrame({"text": [input_text]}))
        
        # Process prediction
        if isinstance(prediction, np.ndarray):
            if len(prediction.shape) == 2:
                predicted_class = np.argmax(prediction[0])
                confidence = float(np.max(prediction[0]))
            else:
                predicted_class = int(prediction[0])
                confidence = 1.0
        else:
            predicted_class = int(prediction)
            confidence = 1.0

        processing_time = time.time() - start_time
        CLASSIFY_PDF_LATENCY.observe(processing_time)
        CLASSIFY_PDF_COUNTER.labels(status='200').inc()

        return PredictionResponse(
            category=label_map.get(predicted_class, "Unknown"),
            category_id=predicted_class,
            confidence=confidence,
            processing_time=processing_time,
            title=str(title),
            abstract=str(abstract)
        )

    except Exception as e:
        CLASSIFY_PDF_COUNTER.labels(status='500').inc()
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
    
# Add a new endpoint to collect feedback
@app.post("/submit_feedback")
async def submit_feedback(feedback: FeedbackInput):

    start_time = time.time()
    
    try:
        feedback_dir = Path("data/feedback")
        feedback_dir.mkdir(parents=True, exist_ok=True)
        feedback_csv_path = feedback_dir / "user_corrections.csv"
        lock_path = feedback_csv_path.with_suffix('.lock')

        feedback_data = {
            "title": [feedback.title],
            "abstract": [feedback.abstract],
            "predicted_category": [feedback.predicted_category],
            "label": [feedback.correct_category],
            "timestamp": [pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")]
        }
        feedback_df = pd.DataFrame(feedback_data)

        with FileLock(str(lock_path)):
            if not feedback_csv_path.exists():
                feedback_df.to_csv(feedback_csv_path, index=False)
            else:
                feedback_df.to_csv(feedback_csv_path, mode='a', header=False, index=False)
            # Get the current count of feedback entries
            full_df = pd.read_csv(feedback_csv_path)
            feedback_count = len(full_df)
            
        FEEDBACK_COUNTER.labels(status='200').inc()
        processing_time = time.time() - start_time
        FEEDBACK_LATENCY.observe(processing_time)
        
        return {
            "status": "success",
            "message": "Feedback recorded successfully",
            "feedback_count": feedback_count,
            "threshold": 3
        }
    except Exception as e:
        FEEDBACK_COUNTER.labels(status='500').inc()
        raise HTTPException(status_code=500, detail=f"Error recording feedback: {str(e)}")
    

@app.get("/samples", response_model=SampleResponse)
async def get_samples():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = time.time()
    
    try:
        # Load a subset of test data (v1)
        test_data = pd.read_csv("data/processed/v1/test.csv")
        
        # Map label codes to indices if needed
        if "label_idx" not in test_data.columns:
            test_data["label_idx"] = test_data["label"].map(code_to_idx)
        
        # Get 3 samples from each category
        samples_per_category = 3
        selected_samples = []
        
        for category_id in range(10):
            category_samples = test_data[test_data["label_idx"] == category_id]
            # select first 3 samples per category
            if len(category_samples) >= samples_per_category:
                selected_samples.append(category_samples.head(samples_per_category))
            elif len(category_samples) > 0:
                selected_samples.append(category_samples)
        
        combined_samples = pd.concat(selected_samples).reset_index(drop=True)
        
        # Make predictions for these samples
        results = []
        correct_predictions = 0
        
        for _, row in combined_samples.iterrows():
            # Prepare input
            input_text = f"{row['title']} {row['abstract']}"
            
            # Get prediction
            prediction = model.predict(pd.DataFrame({"text": [input_text]}))
            
            # Process prediction
            if isinstance(prediction, np.ndarray):
                if len(prediction.shape) == 2:
                    predicted_class = np.argmax(prediction[0])
                else:
                    predicted_class = int(prediction[0])
            else:
                predicted_class = int(prediction)
            
            # Get category names
            true_category = label_map[int(row["label_idx"])]
            predicted_category = label_map[predicted_class]
            
            # Check if prediction is correct
            is_correct = predicted_class == int(row["label_idx"])
            if is_correct:
                correct_predictions += 1
            
            # Create sample object
            sample = SamplePaper(
                title=row["title"],
                abstract=row["abstract"],
                true_category=true_category,
                predicted_category=predicted_category,
                correct=is_correct
            )
            
            results.append(sample)
        
        # Organize samples by predicted category
        samples_by_category = {}
        for sample in results:
            if sample.true_category not in samples_by_category:
                samples_by_category[sample.true_category] = []
            samples_by_category[sample.true_category].append(sample)
        
        # Calculate accuracy
        accuracy = correct_predictions / len(results) if results else 0
        
        SAMPLES_COUNTER.labels(status='200').inc()
        processing_time = time.time() - start_time
        SAMPLES_LATENCY.observe(processing_time)
        
        return SampleResponse(
            samples_by_category=samples_by_category,
            total_count=len(results),
            accuracy=accuracy
        )
    except Exception as e:
        SAMPLES_COUNTER.labels(status='500').inc()
        raise HTTPException(status_code=500, detail=f"Error getting samples: {str(e)}")
    
# Add a new endpoint to manually trigger model reload
@app.post("/reload-model")
async def reload_model():
    global model
    try:
        model = get_best_model()
        return {"message": "Model reloaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reloading model: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
