# Sentiment Analysis of Google Reviews

A clean, reproducible notebook project for **classifying sentiment in Google reviews** using both a **traditional ML pipeline (TF–IDF + Multinomial Naive Bayes)** and an **LLM-assisted baseline** (Gemini). This README documents the dataset, methodology, environment setup, and how to run/evaluate/infer with the project at a professional standard.

---

## 🔍 Project Overview

This repository contains a single Jupyter notebook that:
- Downloads a Google Reviews dataset (`reviews.csv`) via `gdown` (or lets you provide your own).
- Performs quick **EDA** on `score` distribution.
- Maps review `score` → **sentiment** (`negative`, `neutral`, `positive`).
- Builds a reproducible **ML pipeline**: text preprocessing → TF–IDF vectorization → **MultinomialNB** classifier.
- Offers **interactive inference** for ad-hoc sentiment checks.
- Includes an **optional LLM baseline** using Google **Gemini** for comparison.

> **Why this design?** TF–IDF + MultinomialNB is a strong baseline for short-text classification, fast to train, interpretable, and easy to deploy. The LLM baseline demonstrates how prompting can perform zero-shot/ICL-style sentiment classification on the same inputs.

---

## 📦 Tech Stack

- **Python**: 3.10+ recommended
- **Core**: `pandas`, `scikit-learn`, `nltk`, `matplotlib`
- **Data I/O**: `gdown` (to fetch dataset from Google Drive)
- **LLM (Optional)**: `google-generativeai` (Gemini)

---

## 🗂️ Data

- Expected CSV file name: `reviews.csv`
- Required columns:
  - `content` (str): raw review text
  - `score` (int): star rating, typically 1—5

Two ingestion paths:
1. **From Google Drive** (default in the notebook): `gdown.download()` with a file id/URL.
2. **Local file**: place `reviews.csv` next to the notebook and comment out `gdown` lines.

> The notebook converts `score` → `sentiment` with:
> - 1, 2 → `negative`
> - 3 → `neutral`
> - 4, 5 → `positive`

---

## 🧰 Environment Setup

```bash
# 1) Create & activate a virtual environment (any tool is fine)
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# 2) Install dependencies
pip install -U pip
pip install pandas scikit-learn nltk matplotlib gdown google-generativeai
```

> If using Jupyter locally: `pip install jupyter ipykernel && python -m ipykernel install --user --name reviews-env`

**NLTK data** (the notebook downloads as needed):
```python
import nltk
nltk.download("stopwords")
nltk.download("wordnet")
```

---

## 📓 How to Run

1. Open the notebook: `Sentiment Analysis of Google Reviews.ipynb`
2. (Optional) Update the `gdown` file id if you’re pulling the dataset from your own Drive link.
3. Run all cells in order. Key steps you’ll see:
   - **EDA**: plot ratings distribution, derive label distribution.
   - **Preprocessing**: lowercasing, punctuation removal, stopword removal, lemmatization/stemming.
   - **Vectorization**: `TfidfVectorizer`.
   - **Model**: `MultinomialNB` (sklearn Pipeline).
   - **Split**: train/test with `train_test_split(random_state=42)`.
   - **Inference**: type a review; get a predicted sentiment.
   - **LLM baseline (optional)**: configure `google-generativeai` API key and run the Gemini prompt.

---

## 🧪 Evaluation

> **Add these cells if you haven’t already** (recommended):
- **Accuracy, precision, recall, F1** (macro/weighted) via `classification_report`.
- **Confusion matrix** to see misclassifications by class.
- **Baseline comparison**:
  - Compare MultinomialNB vs. Gemini on a held-out labeled subset to get an apples-to-apples view.

**Example (sketch):**
```python
from sklearn.metrics import classification_report, confusion_matrix

y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred, digits=3))

import matplotlib.pyplot as plt
import seaborn as sns  # If you prefer matplotlib only, replace with pure plt
cm = confusion_matrix(y_test, y_pred, labels=["negative","neutral","positive"])
# ...plot cm...
```

---

## 🧠 Model Details

- **Preprocessing**: lowercase → remove punctuation → remove English stopwords → lemmatize/stem tokens.
- **Vectorizer**: `TfidfVectorizer` (default params are a good baseline; consider `ngram_range=(1,2)`, `min_df`, `max_df`, etc.).
- **Classifier**: `MultinomialNB` (fast, robust for sparse text features).

**Why these choices?**
- Works well on short texts with limited compute.
- Easy to interpret and maintain.
- Great baseline to iterate on with `LogisticRegression`, `LinearSVC`, or `ComplementNB`.

---

## 🤖 LLM Baseline (Optional)

- **Model**: `gemini-pro` via `google-generativeai`
- **Prompt**: Asks the model to output a normalized JSON string: `{"sentiment":"positive|neutral|negative"}`
- **Use cases**: Zero-shot classification, qualitative checks, edge cases LLMs often capture better (sarcasm, idioms).

> Keep in mind: Ensure **compliance** with API terms and avoid sending PII if the dataset contains sensitive content.

---

## 🧩 Reproducibility

- Set seeds where possible: `numpy`, `random`, and sklearn splits (already uses `random_state=42`).
- Log package versions:
```python
import sys, sklearn, pandas, nltk, matplotlib
print(sys.version)
print("sklearn:", sklearn.__version__)
print("pandas:", pandas.__version__)
print("nltk:", nltk.__version__)
print("matplotlib:", matplotlib.__version__)
```

- Consider exporting the trained pipeline with `joblib` for reuse:
```python
import joblib
joblib.dump(pipeline, "sentiment_nb.joblib")
# later: pipeline = joblib.load("sentiment_nb.joblib")
```

---

## 🚀 Inference

- **Interactive** (in-notebook): input a review and print the predicted sentiment.
- **Batch** (suggested): create a helper that takes a CSV and adds a `predicted_sentiment` column for downstream analytics.

**Example (sketch):**
```python
def predict_batch(csv_path, text_col="content"):
    df = pd.read_csv(csv_path)
    df["predicted_sentiment"] = pipeline.predict(df[text_col].apply(preprocess_text))
    df.to_csv("predictions.csv", index=False)
    return df
```

---

## 📈 Extending the Project

- Try stronger classical models: `LinearSVC`, `LogisticRegression`, `ComplementNB`.
- Add hyperparameter search: `GridSearchCV` or `RandomizedSearchCV`.
- Improve text features: character n-grams, `ngram_range`, `min_df`, `max_df`, domain stopwords.
- Add robust evaluation: stratified splits, cross-validation, calibration plots.
- Include error analysis: inspect false positives/negatives; word clouds for each class.
- Try modern embeddings: `sentence-transformers` + linear classifier.
- Productionize the pipeline with a REST API (FastAPI) and CI for tests.

---

## ⚠️ Limitations & Ethics

- Star ratings can be noisy; **score→sentiment** mapping is heuristic.
- Reviews may contain bias, sarcasm, or multilingual text not fully handled by this baseline.
- Avoid exposing PII; follow data governance and platform policies when handling real reviews.

---

## 📜 License

Add your preferred license (e.g., MIT) here.

---

## 🙌 Acknowledgements

- scikit-learn, NLTK, Matplotlib
- Google Gemini (optional baseline)
- Dataset: Google Reviews (replace with your citation/source if appropriate)
