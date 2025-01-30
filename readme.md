# AI/ML Financial analysis
## summary
As a learning tool I am analyzing 10-k's from the S&P 500

## goals
1. Extract some meaningful conclusions about company trajectories 
2. Highlight companies in a strong position for growth
3. Make reccommendations of companies that are better buys than others

## progress
1. get data
`complete` good enough for current scope, got a dataset from sec-edgar
2. process data
`in-progress` processing data from html format, need to check correctness
3. Determine tools
    finbert
4. Early Data Analysis
    skipping for now
5. Summarize

6. Sentiment & Financial Analysis
7. Build ML pipeline
8. Automate analysis
nltk
    punkt
spacy
pandas
pqdm
## Prepare python env
1. set up virtual environment

`python -m venv venv`

2. turn on venv

`venv\Scripts\activate`

3. update pip

`python -m pip install --upgrade pip`

4. install necessary packages
```
pip install transformers torch  sec-edgar-downloader nltk spacy pandas tqdm matplotlib wordcloud seaborn
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt')"
python -c "import nltk; nltk.download('punkt_tab')"

```

## prepare file structure
Data locations

`mkdir sec-edgar-filings`

`mkdir cleaned_10k_reports`

Make scratchwork/notes section, always nice to have

`mkdir scratch`

Don't upload that big ol' meatball of data, our scratchwork, or venv

`touch .gitignore`

`echo "sec-edgar-filings/" >> .gitignore`

`echo "cleaned_10k_reports/" >> .gitignore`

`echo "scratch/" >> .gitignore`

`echo "venv/" >> .gitignore`

## Get data
Get the data from sec edgar, pulls all the 10-k's from api, will take a long time and a lot of memory

`python ./edgar_s&p_get.py`

I have an outdated list hardcoded into the file, right now there are a few outdated ticker symbols and probably a few missing companies, but this is okay for now. 
todo: before beginning data retrieval from edgar, make an api call for a current s&p 500 ticker symbol set or CIK set

## process data
currently this destroys the initally retrieved data, I need more hd space

`python process.py`

##



