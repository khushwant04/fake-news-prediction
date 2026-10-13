# Fake News Prediction

Binary fake/real news classifier (TF-IDF + Logistic Regression) trained in a notebook, with a small
Flask API that serves the pickled model.

## Dataset

`Data/train.csv` (~98 MB, 20,800 rows) with columns `id, title, author, text, label`
(`1` = fake, `0` = real). This matches the Kaggle "Fake News" competition training set.

## Approach (`fakenews.ipynb`)

1. Fill missing values, build `content = author + " " + title` (the article body is not used).
2. Keep letters only, lowercase, drop NLTK English stopwords, Porter-stem.
3. `TfidfVectorizer` -> `LogisticRegression`, 80/20 stratified split (`random_state=42`).
4. Pickle the model to `model.pkl`.

Results printed in the notebook:

| Split | Accuracy |
|-------|----------|
| Train | 0.9872 |
| Test  | 0.9752 |

## Project structure

```
fakenews.ipynb       # preprocessing, training, evaluation
app.py               # Flask API
model.pkl            # trained LogisticRegression
vectorizer.pkl       # fitted TfidfVectorizer used by app.py
templates/home.html  # placeholder home page
Procfile             # gunicorn entrypoint (Heroku-style)
Data/train.csv
```

## Running the API

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python app.py                  # dev server on http://127.0.0.1:5000
# or: gunicorn app:app
```

```bash
curl -X POST http://127.0.0.1:5000/predict_api \
  -H 'Content-Type: application/json' \
  -d '{"data": {"title": "Some news headline"}}'
# -> {"prediction": 0}   (0 = real, 1 = fake)
```

## Known limitations

- `app.py` vectorizes the raw title, while the model was trained on stemmed, stopword-filtered
  `author + title` text, so API predictions do not go through the same preprocessing as training.
- The notebook does not save `vectorizer.pkl`; the committed file was produced separately.
- The pickles are tied to the scikit-learn version they were created with; loading them with a
  different version may warn or fail. Re-run the notebook to regenerate if needed.

## License

Apache License 2.0 - see [LICENSE](LICENSE).
