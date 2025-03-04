import os
import random
import logging
import nltk
import datetime
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from transformers import pipeline
from transformers.utils.logging import set_verbosity_error
from deep_translator import GoogleTranslator, exceptions
from nltk.corpus import wordnet
# from functools import lru_cache
import warnings


warnings.filterwarnings("ignore",message=".*max_length.*")
set_verbosity_error()
logging.getLogger("tranformers").setLevel(logging.ERROR)

# Setup logging
logging.basicConfig(
    filename='./data/augmentation_pipeline.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def ensure_nltk_resources():
    """Ensures required NLTK resources are available before downloading."""
    resources = ['punkt', 'stopwords', 'wordnet', 'omw-1.4']
    
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}')  # Check if the resource is available
        except LookupError:
            nltk.download(resource)  # Download only if missing



# # Download required resources
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('omw-1.4')

try:
    wordnet.ensure_loaded()
except AttributeError:
    import importlib
    importlib.reload(nltk.corpus.wordnet)

# Load models
logging.info("[1/5] Loading models...")
# fastest with cpu in mind
paraphraser = pipeline("text2text-generation", model="t5-small", device=0)  # Enable GPU if availabl
summarizer = pipeline("summarization", model="t5-small", device=0) 

# faster
# paraphraser = pipeline("text2text-generation", model="Vamsi/T5_Paraphrase_Paws", device=0)  # GPU-accelerated
# summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=0)  # GPU-accelerated

# slowest
# logging.info("[1/5] Loading models...")
# paraphraser = pipeline("text2text-generation", model="ramsrigouthamg/t5_paraphraser")
# summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
# logging.info("✅ Models loaded.")

# Define maximum token length per chunk
MAX_TOKENS = 512

# Parallel Processing Configuration
MAX_WORKERS = os.cpu_count() or 4

def split_into_chunks(text, max_tokens=MAX_TOKENS):
    """Splits text into chunks of max_tokens size."""
    logging.info("split_into_chunks")
    words = text.split()
    return [" ".join(words[i:i+max_tokens]) for i in range(0, len(words), max_tokens)]

def paraphrase_text(text):
    """Generates a paraphrased version of the given summary."""
    logging.info("paraphrase_text")
    truncated_text = text[:MAX_TOKENS]
    return paraphraser(truncated_text, max_length=min(len(truncated_text.split()) - 1, 150), num_return_sequences=1, do_sample=True)[0]["generated_text"]


def shuffle_sentences(text):
    """Shuffles sentences efficiently using NumPy."""
    sentences = nltk.sent_tokenize(text)
    np.random.shuffle(sentences)
    return " ".join(sentences)


# def shuffle_sentences(text):
#     """Shuffles sentences in the given text to create a new variation."""
#     logging.info("shuffle_sentences")
#     sentences = text.split(". ")
#     random.shuffle(sentences)
#     return ". ".join(sentences)

def back_translate(text, lang="fr"):
    """Translates text to another language and back to English."""
    logging.info("translate_text")
    try:
        truncated_text = text[:MAX_TOKENS]
        translated = GoogleTranslator(source="en", target=lang).translate(truncated_text)
        return GoogleTranslator(source=lang, target="en").translate(translated)
    except exceptions.TranslationNotFound:
        logging.warning("⚠️ Back translation failed: No translation found. Skipping this augmentation.")
        return text

def expand_text(text):
    """Generates a longer variation of the summary."""
    logging.info("expand_text")
    truncated_text = text[:MAX_TOKENS]
    return summarizer(truncated_text, max_length=min(len(truncated_text.split()) * 2 - 1, 250), min_length=min(len(truncated_text.split()) - 1, 100), do_sample=False)[0]["summary_text"]

def compress_text(text):
    """Generates a shorter version of the summary."""
    logging.info("compress_text")
    truncated_text = text[:MAX_TOKENS]
    return summarizer(truncated_text, max_length=max(len(truncated_text.split()) // 2 - 1, 50), min_length=max(len(truncated_text.split()) // 3 - 1, 20), do_sample=False)[0]["summary_text"]

# Manual cache dictionary
synonym_cache = {}

def get_synonyms(word):
    """Fetch synonyms from WordNet with a manual dictionary cache."""
    if word in synonym_cache:
        return synonym_cache[word]  # Return cached value

    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().replace("_", " "))  # Replace underscores with spaces

    # Cache the result to avoid redundant lookups
    synonym_cache[word] = list(synonyms) if synonyms else [word]
    return synonym_cache[word]

def replace_with_synonyms(text):
    """Replaces words with synonyms using manual caching."""
    logging.info("replace_with_synonyms")
    words = text.split()
    
    for i, word in enumerate(words):
        synonyms = get_synonyms(word)  # Use manual caching instead of calling wordnet multiple times
        if len(synonyms) > 1:
            words[i] = random.choice(synonyms)  # Random synonym selection

    return " ".join(words)

# @lru_cache(maxsize=10000)  # Cache results to speed up repeated lookups
# def replace_with_synonyms(word):
#     """Fetch synonyms from WordNet with caching."""
#     synonyms = set()
#     for syn in wordnet.synsets(word):
#         for lemma in syn.lemmas():
#             synonyms.add(lemma.name())
#     return list(synonyms) if synonyms else [word]


# def replace_with_synonyms(text):
#     """Replaces key financial words with synonyms to introduce variation."""
#     logging.info("replace_with_synonyms")
#     words = text.split()
#     for i, word in enumerate(words):
#         synonyms = wordnet.synsets(word)
#         if synonyms:
#             synonym = random.choice(synonyms).lemmas()[0].name()
#             if(synonym != word):
#                 words[i] = synonym
#     return " ".join(words)

# def replace_with_synonyms(text):
#     """Replaces key financial words with synonyms to introduce variation."""
#     logging.info("replace_with_synonyms")
#     words = text.split()
#     for i, word in enumerate(words):
#         synonyms = wordnet.synsets(word)
#         if synonyms:
#             synonym = random.choice(synonyms).lemmas()[0].name()
#             if(synonym != word):
#                 words[i] = synonym
#     return " ".join(words)

def augment_text(text):
    """Applies multiple augmentation techniques to a text chunk."""
    return [
        paraphrase_text(text),
        shuffle_sentences(text),
        # back_translate(text),
        expand_text(text),
        compress_text(text),
        replace_with_synonyms(text)
    ]

def augment_summary(text):
    """Applies augmentation in chunks to avoid exceeding model limits."""
    logging.info("🔄 Augmenting text...")
    chunks = split_into_chunks(text)
    augmented_versions = []
    tasks = [text]
    
    # gpu
    # with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    #     results = list(executor.map(augment_text, chunks))

    # cpu
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(executor.map(augment_text, tasks))
    
    for result in results:
        augmented_versions.extend(result)
    
    logging.info("✅ Augmentation complete.")
    return augmented_versions

def import_text_files(input_folder):
    """Reads 10 random .txt files from a folder, prints their names, and combines them into a DataFrame."""
    logging.info("[2/5] Importing 1 random text file...")
    data = []
    txt_files = []
    
    # Walk through the folder and collect .txt file paths
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".txt"):
                txt_files.append(os.path.join(root, file))
    
    # Select 10 random files (if there are less than 10, select all of them)
    sampled_files = random.sample(txt_files, k=min(1, len(txt_files)))
    
    # Print out the names of the selected files
    print("Selected files:")
    for file_path in sampled_files:
        print(os.path.basename(file_path))
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
            data.append({"report_text": text, "summary_text": text})
    
    logging.info(f"✅ Imported {len(data)} text files.")
    return pd.DataFrame(data)

def all_import_text_files(input_folder):
    """Reads all .txt files from a folder and combines them into a DataFrame."""
    logging.info("[2/5] Importing text files...")
    data = []
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".txt"):
                with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                    text = f.read()
                    data.append({"report_text": text, "summary_text": text})
    logging.info(f"✅ Imported {len(data)} text files.")
    return pd.DataFrame(data)

def export_dataset(df, output_folder):
    """Exports the augmented dataset into the specified folder with a timestamp."""
    logging.info("[5/5] Exporting dataset...")
    timestamp_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_folder, f"augmented_train_{timestamp_id}.csv")
    df.to_csv(output_file, index=False)
    logging.info(f"✅ Dataset saved to {output_file}")

def augment_dataset(input_folder, output_folder):
    """Applies augmentation to all text files in the input folder and saves the results."""
    df = import_text_files(input_folder)
    augmented_data = []
    logging.info("[3/5] Applying augmentation...")
    
    for i, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df))):
        text = row["summary_text"]
        augmented_versions = augment_summary(text)
        
        for aug_text in augmented_versions:
            augmented_data.append({"report_text": row["report_text"], "summary_text": aug_text})
    
    augmented_df = pd.DataFrame(augmented_data)
    logging.info(f"✅ Augmented {len(augmented_df)} samples.")
    export_dataset(augmented_df, output_folder)

if __name__ == "__main__":
    # paths = ["A","B","C","D","F","G","H","I","J","K"]
    ensure_nltk_resources()
    logging.info("🚀 Starting data augmentation pipeline...")
    output_folder = "./data/augmented_reports"
    os.makedirs(output_folder, exist_ok=True)

    timestamp_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    print(timestamp_id)
    input_folder = "./data/cleaned_10k_reports"
    for i in range(10):
        print(f"working on #{i+1} of 10")
        augment_dataset(input_folder, output_folder)


