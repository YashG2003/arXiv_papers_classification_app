import string
import pandas as pd
from transformers import BertTokenizer
from config import Config

class TextPreprocessor:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained(Config.BERT_MODEL_NAME)
        # Predefined label mapping from Config
        self.label_map = {k: idx for idx, k in enumerate(Config.CATEGORY_MAP.keys())}
        self.reverse_map = {v: k for k, v in self.label_map.items()}
        
    def clean_text(self, text):
        """Minimal cleaning - BERT handles most punctuation"""
        return str(text).lower().strip()  # Remove casing only
        
    def preprocess_for_bert(self, df):
        df = df.copy()
        # Combine title + abstract with BERT's [SEP] token
        df['text'] = df['title'] + " " + self.tokenizer.sep_token + " " + df['abstract']
        df['text'] = df['text'].apply(self.clean_text)
        
        # Map labels using predefined categories (handle unknown as -1)
        df['label_idx'] = df['label'].map(self.label_map).fillna(-1).astype(int)
        
        # Verify all labels are valid
        if (df['label_idx'] == -1).any():
            invalid = df[df['label_idx'] == -1]['label'].unique()
            raise ValueError(f"Invalid labels detected: {invalid}")
            
        return df
        
    def tokenize_data(self, df):
        """Convert text to BERT input format"""
        return self.tokenizer(
            df['text'].tolist(),
            padding='max_length',  # Use fixed-length padding
            max_length=Config.MAX_LENGTH,  # Now 128
            truncation=True,
            return_tensors="pt"
        )