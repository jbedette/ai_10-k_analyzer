import os
import json
import random
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

# Define paths
DATA_DIR = "data"
CLEANED_DIR = os.path.join(DATA_DIR, "cleaned_10k_reports")
AUGMENTED_DIR = os.path.join(DATA_DIR, "augmented_reports")
OUTPUT_DIR = "split_data"

# Ensure output directories exist
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(OUTPUT_DIR, split), exist_ok=True)

def load_reports(directory):
    """Load JSON reports from a directory."""
    reports = []
    for file in Path(directory).rglob("*.json"):  # Assuming JSON format
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)
            data["filename"] = str(file)  # Store filename for tracking
            reports.append(data)
    return reports

# Load data
cleaned_reports = load_reports(CLEANED_DIR)
augmented_reports = load_reports(AUGMENTED_DIR)

# Combine and shuffle
all_reports = cleaned_reports + augmented_reports
random.shuffle(all_reports)

# Split data (80% train, 10% val, 10% test)
train, temp = train_test_split(all_reports, test_size=0.2, random_state=42)
val, test = train_test_split(temp, test_size=0.5, random_state=42)

# Function to save files into appropriate directories
def save_split(data, split_name):
    split_dir = os.path.join(OUTPUT_DIR, split_name)
    for report in data:
        filename = os.path.basename(report["filename"])
        dest_path = os.path.join(split_dir, filename)
        shutil.copy(report["filename"], dest_path)

# Save data
save_split(train, "train")
save_split(val, "val")
save_split(test, "test")

print("Data split completed. Check the 'split_data' folder.")
