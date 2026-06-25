# Email Spam Classifier

A modular NLP pipeline that classifies SMS/email messages as spam or ham using Logistic Regression and Multinomial Naive Bayes — with side-by-side model evaluation, saved confusion matrices, precision-recall curves, and a live Flask API for real-time inference.

---

## What this project does

- Trains **two classifiers** (Logistic Regression + Multinomial Naive Bayes) on the UCI SMS Spam Collection dataset
- Compares both models across Accuracy, Precision, Recall, and F1 — auto-selects the winner by F1 score
- Saves **confusion matrices** and **precision-recall curves** as output plots
- Serialises the best model and exposes it via a **Flask REST API**
- Auto-downloads the dataset on first run — no manual setup needed

---

## Project structure

```
email-spam-classifier/
│
├── data/
│   └── SMSSpamCollection      # Auto-downloaded on first run
│
├── models/
│   ├── lr_model.pkl           # Trained Logistic Regression
│   ├── nb_model.pkl           # Trained Naive Bayes
│   ├── best_model.pkl         # Best model by F1 score (auto-selected)
│   ├── best_model_info.txt    # Records which model won
│   ├── vectorizer.pkl         # Fitted TF-IDF vectorizer
│   └── test_data.pkl          # Serialised test split for evaluation
│
├── reports/
│   ├── confusion_matrices.png      # Side-by-side confusion matrices
│   └── precision_recall_curves.png # PR curves with Average Precision scores
│
├── data.py          # Dataset download, loading, cleaning, train-test split
├── model.py         # Training pipeline for both classifiers
├── evaluate.py      # Metrics, plots, best-model selection
├── vectorizer.py    # TF-IDF vectorizer: fit, save, load
├── utils.py         # Text cleaning (lowercase, remove special chars)
├── app.py           # Flask prediction API
└── requirements.txt
```

---

## Quickstart

```bash
# Clone the repo
git clone https://github.com/Kushagra-Kapoor-04/email-spam-classifier.git
cd email-spam-classifier

# Install dependencies
pip install -r requirements.txt

# Step 1 — Train both models
python model.py

# Step 2 — Evaluate and generate plots
python evaluate.py

# Step 3 — Start the prediction API
python app.py
```

The dataset downloads automatically on first run from the UCI ML Repository.

---

## API usage

Once the server is running at `http://localhost:5000`:

```bash
curl -X POST http://localhost:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "Congratulations! You have won a £1000 prize. Call now to claim."}'
```

**Response:**
```json
{
  "prediction": "spam",
  "confidence": 0.98,
  "model_used": "Logistic Regression"
}
```

- `prediction` — `"spam"` or `"ham"`
- `confidence` — probability score from `predict_proba` (where available)
- `model_used` — whichever model had the higher F1 on this run

---

## Text preprocessing pipeline

```
Raw text
  → Lowercase
  → Remove punctuation and special characters  (re.sub)
  → Normalise whitespace
  → TF-IDF Vectorisation  (max_features=5000, unigrams)
  → Model input
```

---

## Dataset

**UCI SMS Spam Collection** — 5,574 labelled messages.

| Class | Count | Share |
|---|---|---|
| Ham (legitimate) | 4,827 | 86.6% |
| Spam | 747 | 13.4% |

The ~87/13 class imbalance means raw accuracy is misleading — a model that predicts "ham" for everything would score 86.6%. F1 on the spam class is the metric that actually matters here.

---

## Results

| Metric | Logistic Regression | Naive Bayes |
|---|---|---|
| Accuracy | ~98% | ~97% |
| Precision (spam) | ~96% | ~100% |
| Recall (spam) | ~94% | ~88% |
| F1 (spam) | ~95% | ~93% |

**Winner: Logistic Regression** (selected automatically by F1 score).

> Naive Bayes achieves 100% precision — it never falsely flags a legitimate message as spam — but misses more actual spam (lower recall). Logistic Regression is the better general-purpose filter; Naive Bayes would be preferable in contexts where false positives are catastrophic (e.g. a transactional email system).

Run `python evaluate.py` to regenerate your own numbers — results are deterministic with `random_state=42`.

---

## Output plots

Generated in `reports/` after running `evaluate.py`:

**Confusion matrices** (`confusion_matrices.png`) — side-by-side heatmaps for both models showing true positives, false positives, true negatives, false negatives.

**Precision-Recall curves** (`precision_recall_curves.png`) — full PR curve for both classifiers with Average Precision (AP) scores, showing the precision/recall tradeoff at every decision threshold.

---

## Key concepts demonstrated

- **TF-IDF vectorisation** — term frequency weighted by inverse document frequency; why it outperforms raw counts for text classification
- **Class imbalance** — why accuracy is the wrong metric; F1 and PR curves are the right lens
- **Model comparison** — same task, two algorithms, different precision/recall tradeoffs — and understanding *when each is the right choice*
- **Model serialisation** — saving trained models and vectorizers with `pickle`; loading them at API startup
- **REST API inference** — Flask endpoint with JSON input/output, graceful error handling, and confidence scores

---

## Tech stack

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=flat&logo=flask&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat&logo=plotly&logoColor=white)

---

## What's next

- [ ] Add NLTK stopword removal and bigrams `(1,2)` to `utils.py` — would meaningfully improve recall
- [ ] Fine-tune a BERT model on the same dataset and compare against TF-IDF baseline
- [ ] Containerise with Docker for reproducible deployment
- [ ] Migrate API from Flask to FastAPI for automatic Swagger docs and async support
- [ ] Add confidence threshold flag — low-confidence predictions returned as `"uncertain"` for human review

---

## Author

**Kushagra Kapoor**
[GitHub](https://github.com/Kushagra-Kapoor-04) · [LinkedIn](https://linkedin.com/in/kushagra-kapoor-04)