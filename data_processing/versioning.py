import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import numpy as np
import json
import time
import os

class DataVersioner:
    def __init__(self, input_path="data/raw/arxiv100.csv", 
                 output_dir="data/processed"):
        self.df = None
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def _create_versions(self):
        """Create 3 versions with class-balanced splits"""
        self.df = pd.read_csv(self.input_path)
        versions = []
        
        # Shuffle once at beginning
        self.df = self.df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        for label in self.df['label'].unique():
            class_data = self.df[self.df['label'] == label]
            splits = np.split(
                class_data,
                [int(0.3333*len(class_data)), 
                 int(0.6666*len(class_data))]
            )
            versions.append(splits)
            
        return [
            pd.concat([version[i] for version in versions]).reset_index(drop=True)
            for i in range(3)
        ]
    
    def _save_version(self, version_num, df):
        """Save version data with train/val/test splits"""
        version_dir = self.output_dir / f"v{version_num}"
        version_dir.mkdir(exist_ok=True)
        
        train, val, test = self._split_data(df)
        train.to_csv(version_dir / "train.csv", index=False)
        val.to_csv(version_dir / "val.csv", index=False)
        test.to_csv(version_dir / "test.csv", index=False)
        
        return len(train), len(val), len(test)
    
    def _split_data(self, df):
        """70-15-15 stratified split"""
        train, temp = train_test_split(df, test_size=0.3, stratify=df['label'], random_state=42)
        val, test = train_test_split(temp, test_size=0.5, stratify=temp['label'], random_state=42)
        return train, val, test
    
    def run_pipeline(self):
        """Full versioning pipeline with metrics"""
        start_time = time.time()
        versions = self._create_versions()
        
        metrics = {}
        for i, version_df in enumerate(versions, 1):
            counts = self._save_version(i, version_df)
            metrics[f"v{i}"] = {
                "total_samples": len(version_df),
                "train": counts[0],
                "val": counts[1],
                "test": counts[2]
            }
        
        metrics["processing_time"] = time.time() - start_time
        
        metrics_dir = "data/metrics"
        os.makedirs(metrics_dir, exist_ok=True)
        
        with open(Path(metrics_dir)/ "processing_metrics.json", "w") as f:
            json.dump(metrics, f)
        
        return metrics