import argparse
import subprocess
from pathlib import Path
from config import Config

def run_stage(stage_name=None):
    """Run a specific DVC stage or all stages if none specified"""
    if stage_name:
        print(f"Running DVC stage: {stage_name}")
        subprocess.run(["dvc", "repro", stage_name], check=True)
    else:
        print("Running full DVC pipeline")
        subprocess.run(["dvc", "repro"], check=True)

def ensure_data_available(version=None):
    """Ensure data is available in the workspace"""
    if not Config.PROCESSED_DATA_DIR.exists() or (version and not (Config.PROCESSED_DATA_DIR / f"v{version}").exists()):
        print("Data not found in workspace, checking out from DVC cache")
        if version:
            path = str(Config.PROCESSED_DATA_DIR / f"v{version}")
            subprocess.run(["dvc", "checkout", path], check=True)
        else:
            subprocess.run(["dvc", "checkout", str(Config.PROCESSED_DATA_DIR)], check=True)
        
        if Config.PROCESSED_DATA_DIR.exists():
            print("Successfully restored data from DVC cache")
        else:
            print("Warning: Could not restore data. You may need to run the preprocess stage first.")

def main():
    parser = argparse.ArgumentParser(description="Run arXiv classification pipeline")
    parser.add_argument("--stage", type=str, choices=["preprocess", "train", "fine_tune"], 
                        help="Specific pipeline stage to run (default: run all stages)")
    args = parser.parse_args()
    
    # Check DVC initialization
    if not Path(".dvc").exists():
        raise RuntimeError("DVC not initialized. Run 'dvc init' first")
    
    # If running specific stage
    if args.stage:
        if args.stage == "preprocess":
            run_stage("preprocess")
        elif args.stage == "train":
            ensure_data_available(version=1)
            run_stage("train")
        elif args.stage == "fine_tune":
            ensure_data_available(version=2)
            run_stage("fine_tune")
    else:
        # Run all stages in sequence
        run_stage()

if __name__ == "__main__":
    main()
