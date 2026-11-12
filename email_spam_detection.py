#!/usr/bin/env python3
"""Train and evaluate email/SMS spam classifiers on spam.csv."""

from __future__ import annotations

import argparse
import pickle
import re
from pathlib import Path

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier


def load_stopwords() -> set[str]:
    """Return NLTK English stopwords, downloading the corpus if needed."""
    try:
        return set(stopwords.words("english"))
    except LookupError:
        if not nltk.download("stopwords", quiet=True):
            raise RuntimeError(
                "Could not download NLTK stopwords. Check your internet connection "
                "or install the NLTK stopwords corpus manually."
            )
        return set(stopwords.words("english"))


def preprocess_message(message: object, stemmer: PorterStemmer, stop_words: set[str]) -> str:
    """Normalize a message and return its stemmed, stopword-free text."""
    text = "" if pd.isna(message) else str(message)
    words = re.sub(r"[^a-zA-Z]", " ", text).lower().split()
    return " ".join(stemmer.stem(word) for word in words if word not in stop_words)


def load_dataset(csv_path: Path) -> tuple[pd.Series, pd.Series]:
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {csv_path}\n"
            "Download spam.csv and place it in the project directory, or pass --data PATH."
        )

    data = pd.read_csv(csv_path, encoding="ISO-8859-1")
    missing = {"v1", "v2"} - set(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(sorted(missing))}")

    labels = data["v1"].astype("string").str.strip().str.lower()
    if labels.isna().any() or not labels.isin(["ham", "spam"]).all():
        raise ValueError("Expected every v1 label to be either 'ham' or 'spam'.")

    return labels, data["v2"].fillna("").astype(str)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("spam.csv"), help="Path to spam.csv")
    parser.add_argument(
        "--model-out", type=Path, default=Path("spam_model.pkl"),
        help="Output path for the trained MultinomialNB model and vectorizer",
    )
    args = parser.parse_args()

    labels, messages = load_dataset(args.data)
    stop_words = load_stopwords()
    stemmer = PorterStemmer()

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        messages,
        labels,
        test_size=0.20,
        random_state=42,
        stratify=labels,
    )

    X_train_clean = [preprocess_message(text, stemmer, stop_words) for text in X_train_raw]
    X_test_clean = [preprocess_message(text, stemmer, stop_words) for text in X_test_raw]

    # Fit the vocabulary on training data only to prevent test-set leakage.
    vectorizer = CountVectorizer(max_features=4000)
    X_train = vectorizer.fit_transform(X_train_clean)
    X_test = vectorizer.transform(X_test_clean)
    y_train_binary = y_train.map({"ham": 0, "spam": 1})
    y_test_binary = y_test.map({"ham": 0, "spam": 1})

    models = {
        "Random Forest Classifier": RandomForestClassifier(random_state=42),
        "Decision Tree Classifier": DecisionTreeClassifier(random_state=42),
        "Multinomial Naive Bayes": MultinomialNB(),
    }

    for name, model in models.items():
        model.fit(X_train, y_train_binary)
        predictions = model.predict(X_test)
        print(f"\n{name}")
        print("Confusion matrix (rows=true, columns=predicted; 0=ham, 1=spam):")
        print(confusion_matrix(y_test_binary, predictions, labels=[0, 1]))
        print(f"Accuracy: {accuracy_score(y_test_binary, predictions):.4f}")
        print(classification_report(
            y_test_binary,
            predictions,
            labels=[0, 1],
            target_names=["ham", "spam"],
            zero_division=0,
        ))

    best_model = models["Multinomial Naive Bayes"]
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    with args.model_out.open("wb") as model_file:
        pickle.dump({"model": best_model, "vectorizer": vectorizer}, model_file)
    print(f"Saved Multinomial Naive Bayes model and vectorizer to {args.model_out}")


if __name__ == "__main__":
    main()
