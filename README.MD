##
**Sentiment Analysis of Google Reviews**
##

A clean, reproducible notebook project for classifying sentiment in Google reviews using both a traditional ML pipeline (TF–IDF + Multinomial Naive Bayes) and an LLM-assisted baseline (Gemini). This README documents the dataset, methodology, environment setup, and how to run/evaluate/infer with the project at a professional standard.


Project Overview
Google Reviews dataset (reviews.csv) via gdown (or lets you provide your own).
Performs quick EDA on score distribution.
Maps review score → sentiment (negative, neutral, positive).
Builds a reproducible ML pipeline: text preprocessing → TF–IDF vectorization → MultinomialNB classifier.
Offers interactive inference for ad-hoc sentiment checks.
Includes an optional LLM baseline using Google Gemini for comparison.

TF–IDF + MultinomialNB is a strong baseline for short-text classification, fast to train, interpretable, and easy to deploy. The LLM baseline demonstrates zero-shot/ICL-style sentiment classification on the same inputs.

Model Details

Preprocessing: lowercase → punctuation removal → English stopwords → lemmatization/stemming
Vectorizer: TfidfVectorizer (consider ngram_range=(1,2), min_df, max_df)
Classifier: MultinomialNB (fast & robust for sparse text)

LLM Baseline (Optional)

Model: gemini-pro via google-generativeai
Prompt: outputs normalized JSON: {"sentiment":"positive|neutral|negative"}
Note: Pass API key via environment variable and avoid sending PII.

Extending the Project

Better features: character n-grams, domain stopwords.
Hyperparameter search: GridSearchCV / RandomizedSearchCV.
Robust evaluation: stratified CV, calibration.
Error analysis: inspect FP/FN; word clouds per class.
Modern embeddings: sentence-transformers + linear classifier.
Ship it: FastAPI inference service + basic CI tests.

Limitations & Ethics

The score→sentiment mapping is heuristic.
Reviews can include sarcasm, code-switching, or multilingual text.
Handle PII responsibly
