import argparse
import subprocess
import os

# Define paths
DATA_DIR = "data"
MODEL_DIR = "model_training"

def run_script(directory, script_name, description):
    """Run a Python script from the specified directory with error handling."""
    script_path = os.path.join(directory, script_name)
    print(f"\n[INFO] Running: {description} ({script_path})")
    
    try:
        subprocess.run(["python", script_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] {description} failed: {e}")

def main():
    parser = argparse.ArgumentParser(description="Financial NLP Pipeline")
    
    parser.add_argument("--download", action="store_true", help="Download and preprocess data (edgar_s&p_get, process).")
    parser.add_argument("--augment", action="store_true", help="Run data augmentation pipeline.")
    parser.add_argument("--split", action="store_true", help="Split dataset into train/val/test.")
    parser.add_argument("--train", action="store_true", help="Train summarization model.")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate trained model.")

    args = parser.parse_args()

    if args.download:
        run_script(DATA_DIR, "edgar_s&p_get.py", "Downloading data")
        run_script(DATA_DIR, "process.py", "Processing and cleaning data")

    if args.augment:
        run_script(DATA_DIR, "augmentation_pipeline.py", "Data augmentation")

    if args.split:
        run_script(DATA_DIR, "data_split.py", "Splitting dataset into train/val/test")

    if args.train:
        run_script(MODEL_DIR, "t5_training.py", "Training summarization model")

    if args.evaluate:
        run_script(MODEL_DIR, "evaluate_model.py", "Evaluating trained model")

    if not any(vars(args).values()):
        parser.print_help()

if __name__ == "__main__":
    main()
