import json
import pandas as pd
from pathlib import Path
import os

def validate_data(processed_dir="data/processed"):
    report = {}
    
    for version in Path(processed_dir).glob("v*"):
        train = pd.read_csv(version/"train.csv")
        val = pd.read_csv(version/"val.csv")
        test = pd.read_csv(version/"test.csv")
        
        # Convert int64 to native Python int
        version_report = {
            "class_balance": {
                "train": {k: int(v) for k, v in train['label'].value_counts().items()},
                "val": {k: int(v) for k, v in val['label'].value_counts().items()},
                "test": {k: int(v) for k, v in test['label'].value_counts().items()}
            },
            "split_ratios": {
                "train": float(len(train)/(len(train)+len(val)+len(test))),
                "val": float(len(val)/(len(train)+len(val)+len(test))),
                "test": float(len(test)/(len(train)+len(val)+len(test)))
            }
        }
        report[str(version.name)] = version_report
        
    metrics_dir = "data/metrics"
    os.makedirs(metrics_dir, exist_ok=True)
    
    with open(Path(metrics_dir)/"validation_report.json", "w") as f:
        json.dump(report, f, indent=2)  # Added indent for readability
    
    return report

if __name__ == "__main__":
    validate_data()