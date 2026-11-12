# Email Spam Detection

A small machine-learning project that classifies SMS/email-style messages as **spam** or **ham** using text preprocessing, a bag-of-words representation, and scikit-learn classifiers.

## Project files

- `EMAIL_SPAM_DETECTION.ipynb` — notebook workflow with exploration, training, evaluation, and sample predictions.
- `email_spam_detection.py` — runnable command-line training script.
- `email_spam_detection_report.tex` — project report source.
- `requirements.txt` — Python dependencies.

## Dataset

Download the SMS Spam Collection dataset from [Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) or use the provided [Google Drive link](https://drive.google.com/file/d/19Yz5vBQ3T174xeSOOqVzw3rWWm39stsf/view?usp=share_link). Place the CSV in the project directory as `spam.csv`. The script and notebook expect the dataset's `v1` (label) and `v2` (message) columns; extra columns are ignored.

## Run locally

```bash
python -m pip install -r requirements.txt
python email_spam_detection.py
```

The first run downloads the NLTK English stopwords corpus, so it needs internet access. The script prints classification reports and saves the fitted Multinomial Naive Bayes model and vectorizer to `spam_model.pkl`.

To use a different dataset path:

```bash
python email_spam_detection.py --data path/to/spam.csv
```

Open `EMAIL_SPAM_DETECTION.ipynb` in Jupyter or Google Colab and run the cells in order. In either environment, make `spam.csv` available in the notebook's working directory.

## Workflow

1. Load and inspect labeled messages.
2. Normalize text, remove nonalphabetic characters and English stopwords, and stem tokens with Porter stemming.
3. Split the examples into training and test sets, keeping the class distribution stratified.
4. Fit a training-only `CountVectorizer` and compare Random Forest, Decision Tree, and Multinomial Naive Bayes classifiers.
5. Report accuracy, confusion matrices, and per-class precision/recall/F1; save the Multinomial Naive Bayes model and vectorizer together.

The data is class-imbalanced, so accuracy alone should not be used to judge spam recall or false positives. Results can vary with dataset version and library versions.

## Important

Only load `spam_model.pkl` files that you created or otherwise trust. Python pickle files can execute code when loaded.
