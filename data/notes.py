import os
import random
import logging
import nltk
import datetime
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from transformers import pipeline
from transformers.utils.logging import set_verbosity_error
from deep_translator import GoogleTranslator, exceptions
from nltk.corpus import wordnet
import warnings

warnings.filterwarnings("ignore", message=".*max_length.*")
set_verbosity_error()
logging.getLogger("transformers").setLevel(logging.ERROR)

# Setup logging
logging.basicConfig(
    filename='./data/augmentation_pipeline.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Define maximum token length per chunk
MAX_TOKENS = 512

# Parallel Processing Configuration
MAX_WORKERS = os.cpu_count() or 4

def ensure_nltk_resources():
    """Ensures required NLTK resources are available before downloading."""
    resources = ['punkt', 'stopwords', 'wordnet', 'omw-1.4']
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}')
        except LookupError:
            nltk.download(resource)

try:
    wordnet.ensure_loaded()
except AttributeError:
    import importlib
    importlib.reload(nltk.corpus.wordnet)

def load_models(use_gpu):
    """Loads models based on whether CPU or GPU should be used."""
    device = 0 if use_gpu else -1  # GPU is 0, CPU is -1
    logging.info("[1/5] Loading models on GPU" if use_gpu else "[1/5] Loading models on CPU")
    
    paraphraser = pipeline("text2text-generation", model="Vamsi/T5_Paraphrase_Paws", device=device)
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=device)
    
    return paraphraser, summarizer

def split_into_chunks(text, max_tokens=MAX_TOKENS):
    """Splits text into chunks of max_tokens size."""
    words = text.split()
    return [" ".join(words[i:i+max_tokens]) for i in range(0, len(words), max_tokens)]

def augment_text(text):
    """Applies multiple augmentation techniques to a text chunk."""
    return [
        paraphraser(text, max_length=150, num_return_sequences=1, do_sample=True)[0]["generated_text"],
        " ".join(np.random.permutation(nltk.sent_tokenize(text))),
        summarizer(text, max_length=250, min_length=100, do_sample=False)[0]["summary_text"],
        summarizer(text, max_length=50, min_length=20, do_sample=False)[0]["summary_text"],
        replace_with_synonyms(text)
    ]

def replace_with_synonyms(text):
    """Replaces words with synonyms using WordNet."""
    words = text.split()
    return " ".join([random.choice(get_synonyms(word)) if get_synonyms(word) else word for word in words])

def get_synonyms(word):
    """Fetch synonyms from WordNet."""
    return [lemma.name().replace("_", " ") for syn in wordnet.synsets(word) for lemma in syn.lemmas()]

def augment_summary(text, use_threading):
    """Applies augmentation in chunks, choosing between threading and processing."""
    logging.info("🔄 Augmenting text...")
    chunks = split_into_chunks(text)
    executor_class = ThreadPoolExecutor if use_threading else ProcessPoolExecutor
    
    with executor_class(max_workers=MAX_WORKERS) as executor:
        results = list(executor.map(augment_text, chunks))

    logging.info("✅ Augmentation complete.")
    return [item for sublist in results for item in sublist]

def import_text_files(input_folder):
    """Reads random .txt files from a folder and combines them into a DataFrame."""
    logging.info("[2/5] Importing text files...")
    txt_files = [os.path.join(root, file) for root, _, files in os.walk(input_folder) for file in files if file.endswith(".txt")]
    sampled_files = random.sample(txt_files, min(1, len(txt_files)))
    
    data = [{"report_text": open(file, "r", encoding="utf-8").read(), "summary_text": open(file, "r", encoding="utf-8").read()} for file in sampled_files]
    
    logging.info(f"✅ Imported {len(data)} text files.")
    return pd.DataFrame(data)

def export_dataset(df, output_folder):
    """Exports the augmented dataset into a CSV file."""
    timestamp_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_folder, f"augmented_train_{timestamp_id}.csv")
    df.to_csv(output_file, index=False)
    logging.info(f"✅ Dataset saved to {output_file}")

def augment_dataset(input_folder, output_folder, use_threading):
    """Processes the dataset and applies augmentation."""
    df = import_text_files(input_folder)
    augmented_data = []
    
    logging.info("[3/5] Applying augmentation...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        augmented_versions = augment_summary(row["summary_text"], use_threading)
        augmented_data.extend([{"report_text": row["report_text"], "summary_text": aug_text} for aug_text in augmented_versions])

    export_dataset(pd.DataFrame(augmented_data), output_folder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Augmentation Pipeline")
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU if available")
    parser.add_argument("--use_threading", action="store_true", help="Use ThreadPool instead of ProcessPool")

    args = parser.parse_args()
    
    ensure_nltk_resources()
    paraphraser, summarizer = load_models(args.use_gpu)

    if paraphraser and summarizer:
        logging.info("✅ Models loaded. 🚀 Starting data augmentation pipeline...")
        output_folder = "./data/augmented_reports"
        os.makedirs(output_folder, exist_ok=True)
        input_folder = "./data/cleaned_10k_reports"

        for i in range(10):
            print(f"Working on batch #{i+1} of 10")
            augment_dataset(input_folder, output_folder, args.use_threading)
    else:
        logging.error("❌ Model loading failed.")
