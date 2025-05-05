import os
from pathlib import Path

class Config:
    
    # ARXIV-10 Categories
    CATEGORY_MAP = {
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

    
    # Data paths
    RAW_DATA_PATH = Path("data/arxiv100.csv")
    PROCESSED_DATA_DIR = Path("data/processed")
    
    # Versioning
    VERSION_SPLITS = [0.3333, 0.3333, 0.3334]  # Split ratios for 3 versions
    
    # Preprocessing
    MAX_LENGTH = 192  # For BERT
    STOPWORDS = None  # Will be initialized later
    
    # Training
    BERT_MODEL_NAME = "prajjwal1/bert-tiny"
    BATCH_SIZE = 16
    EPOCHS = 3
    LEARNING_RATE = 3e-5
    
    @classmethod
    def setup(cls):
        cls.PROCESSED_DATA_DIR.mkdir(exist_ok=True)
        import nltk
        nltk.download('stopwords')
        from nltk.corpus import stopwords
        cls.STOPWORDS = set(stopwords.words('english'))