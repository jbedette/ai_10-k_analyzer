import os
import re
import spacy
import nltk
from tqdm import tqdm
from nltk.tokenize import sent_tokenize
nltk.download('punkt')

# Load NLP Model
nlp = spacy.load("en_core_web_sm")

# Define Parent Input and Output Folder Paths
PARENT_INPUT_FOLDER = "sec-edgar-filings"
PARENT_OUTPUT_FOLDER = "cleaned_10k_reports"

# Ensure output folder exists
os.makedirs(PARENT_OUTPUT_FOLDER, exist_ok=True)

# Function to clean text
def clean_text(text):
    """Cleans 10-K text by removing unwanted characters and sections."""
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\n+', '\n', text)  # Normalize line breaks
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
    text = re.sub(r'\d+', '', text)  # Remove numbers
    return text.strip()

# Function to chunk document by section headers
def chunk_text(text):
    """Splits text into sections based on common 10-K headers."""
    headers = [
        "business", "risk factors", "management’s discussion and analysis",
        "financial statements", "legal proceedings", "quantitative and qualitative disclosures"
    ]
    
    sections = {}
    current_section = "introduction"
    sections[current_section] = []

    for line in text.split("\n"):
        line = line.strip()
        if any(h in line.lower() for h in headers):
            current_section = line.lower()
            sections[current_section] = []
        sections[current_section].append(line)

    return {k: " ".join(v) for k, v in sections.items()}

# Step 1: Remove old processed files before starting
if os.path.exists(PARENT_OUTPUT_FOLDER):
    print("🗑️ Deleting old processed files...")
    for root, dirs, files in os.walk(PARENT_OUTPUT_FOLDER, topdown=False):
        for file in files:
            os.remove(os.path.join(root, file))
        for dir in dirs:
            os.rmdir(os.path.join(root, dir))

# Walk through all subdirectories in input folder
for root, _, files in os.walk(PARENT_INPUT_FOLDER):
    for filename in tqdm(files):
        if filename.endswith(".txt"):
            input_path = os.path.join(root, filename)

            # Construct relative path for output
            relative_path = os.path.relpath(root, PARENT_INPUT_FOLDER)
            output_dir = os.path.join(PARENT_OUTPUT_FOLDER, relative_path)
            os.makedirs(output_dir, exist_ok=True)

            output_path = os.path.join(output_dir, filename)

            with open(input_path, "r", encoding="utf-8") as f:
                text = f.read()

            # Step 2: Clean the text
            text = clean_text(text)

            # Step 3: Split into sections
            sections = chunk_text(text)

            # Step 4: Tokenize sentences
            for section, content in sections.items():
                sections[section] = " ".join(sent_tokenize(content))

            # Step 5: Save to output file
            with open(output_path, "w", encoding="utf-8") as f:
                for section, content in sections.items():
                    f.write(f"\n\n### {section.upper()} ###\n{content}\n")

print(f"✅ Processing complete! Cleaned files saved in {PARENT_OUTPUT_FOLDER}")
