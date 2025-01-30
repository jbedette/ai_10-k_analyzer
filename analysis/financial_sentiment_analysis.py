import os
import datetime
import torch
import spacy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from transformers import pipeline
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

nltk.download('punkt')
nltk.download('stopwords')

timestamp_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Load NLP Model
nlp = spacy.load("en_core_web_sm")

# Load Financial Sentiment Model (FinBERT)
sentiment_analyzer = pipeline("sentiment-analysis", model="yiyanghkust/finbert-tone")

# Define Parent Folder for Processed 10-K Reports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # Directory where this script is located
DATA_FOLDER = os.path.join(SCRIPT_DIR, "../data/cleaned_10k_reports")  # Relative path to reports

# Find all full-submission.txt files in subdirectories
files = []
for root, _, filenames in os.walk(DATA_FOLDER):
    for filename in filenames:
        if filename == "full-submission.txt":
            files.append(os.path.join(root, filename))

# Define output directory
output_dir = "../data/fin_sentiment_analysis"
os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exist

# Define filename with timestamp
output_filename = f"fin_sentiment_analysis_{timestamp_id}.csv"
output_path = os.path.join(output_dir, output_filename)

# List all processed files
files = [os.path.join(root, f) for root, _, filenames in os.walk(DATA_FOLDER) for f in filenames if f.endswith(".txt")]

print(f"Total reports: {len(files)}")

# Risk-related keywords
risk_keywords = [
    "bankruptcy", "litigation", "debt", "insolvency", "lawsuit", "credit risk",
    "compliance", "fraud", "regulatory", "lawsuit", "penalty", "risk exposure"
]

# Store Results
sentiment_data = []

for file in tqdm(files):
    with open(file, "r", encoding="utf-8") as f:
        text = f.read()
        
        # Run FinBERT Sentiment Analysis
        sentiment_result = sentiment_analyzer(text[:512])  # Limit to 512 tokens
        sentiment_label = sentiment_result[0]["label"]
        sentiment_score = sentiment_result[0]["score"]

        # Extract risk-related words
        words = word_tokenize(text.lower())
        risk_counts = Counter(word for word in words if word in risk_keywords)

        # Store results
        sentiment_data.append({
            "filename": file,
            "sentiment_label": sentiment_label,
            "sentiment_score": sentiment_score,
            "risk_terms": risk_counts
        })

# Convert to DataFrame
df = pd.DataFrame(sentiment_data)

# Save to CSV
df.to_csv(output_path, index=False)

# 📊 Display Sentiment Distribution
sns.histplot(df["sentiment_label"], bins=3, kde=False)
plt.xlabel("Sentiment Category")
plt.ylabel("Number of Reports")
plt.title("Sentiment Analysis of 10-K Reports")
plt.show()

# 📊 Generate Risk Word Cloud
from wordcloud import WordCloud

all_risk_words = " ".join([" ".join(d.keys()) for d in df["risk_terms"]])
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(all_risk_words)

plt.figure(figsize=(10,5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Financial Risk Keywords in 10-K Reports")
plt.show()

# 📊 Display Data
import ace_tools as tools
tools.display_dataframe_to_user(name="Financial Sentiment Analysis", dataframe=df)
