from data_processing.versioning import DataVersioner

if __name__ == "__main__":
    pipeline = DataVersioner()
    metrics = pipeline.run_pipeline()
    print(f"Pipeline metrics: {metrics}")