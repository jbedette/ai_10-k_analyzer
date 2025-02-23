#! /bin/bash
python -m venv venv
. .venv/bin/activate
python -m pip install --upgrade pip
pip install transformers torch  sec-edgar-downloader nltk spacy pandas tqdm matplotlib wordcloud seaborn sentencepiece datasets rouge-score accelerate
python -m spacy download en_core_web_sm deep-translator tiktoken protobuf blobfile
python -c "import nltk; nltk.download('punkt')"
python -c "import nltk; nltk.download('punkt_tab')"