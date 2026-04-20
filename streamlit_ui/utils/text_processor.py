"""Text preprocessing utilities for the Customer Complaint Classifier UI."""

import re

import numpy as np

# ---------------------------------------------------------------------------
# Basic text cleaning
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Clean raw complaint text.

    Steps
    -----
    1. Convert to lowercase.
    2. Remove URLs.
    3. Replace slashes and hyphens with spaces.
    4. Remove non-alphanumeric characters (keep spaces).
    5. Collapse multiple whitespace to a single space.

    Parameters
    ----------
    text:
        Raw complaint string from the user.

    Returns
    -------
    str
        Cleaned text string.
    """
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = text.replace("/", " ").replace("-", " ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_token_count(text: str) -> int:
    """Return the approximate token (word) count for cleaned text."""
    return len(clean_text(text).split())


# ---------------------------------------------------------------------------
# Preprocessing info for display
# ---------------------------------------------------------------------------

def get_preprocessing_info(raw_text: str) -> dict:
    """Return a dict with preprocessing details for display in the UI.

    Parameters
    ----------
    raw_text:
        Original user input text.

    Returns
    -------
    dict with keys: ``original``, ``cleaned``, ``token_count``, ``char_count``
    """
    cleaned = clean_text(raw_text)
    return {
        "original": raw_text,
        "cleaned": cleaned,
        "token_count": get_token_count(raw_text),
        "char_count": len(raw_text),
    }


# ---------------------------------------------------------------------------
# Model-specific prediction helpers
# ---------------------------------------------------------------------------

def predict_baseline(model, text: str) -> tuple[str, np.ndarray]:
    """Predict using the Baseline (DummyClassifier) model.

    DummyClassifier ignores X; pass a minimal dummy array.

    Returns
    -------
    tuple
        ``(predicted_class, probability_array)``
    """
    X_dummy = np.zeros((1, 1))
    prediction = model.predict(X_dummy)[0]
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_dummy)[0]
    else:
        # Build a uniform distribution as fallback
        proba = np.ones(len(model.classes_)) / len(model.classes_)
    return prediction, proba


def predict_svm_pipeline(model, text: str) -> tuple[str, np.ndarray]:
    """Predict using the SVM Pipeline (TfidfVectorizer + LinearSVC).

    LinearSVC does not expose ``predict_proba``; convert ``decision_function``
    scores to pseudo-probabilities via softmax.

    Returns
    -------
    tuple
        ``(predicted_class, probability_array)``
    """
    prediction = model.predict([text])[0]
    scores = model.decision_function([text])[0]
    proba = _softmax(scores)
    return prediction, proba


def predict_with_external_vectorizer(model, vectorizer, text: str) -> tuple[str, np.ndarray]:
    """Predict using a standalone model + externally prepared TF-IDF vectorizer."""
    processed_text = clean_text(text)
    features = vectorizer.transform([processed_text])
    prediction = model.predict(features)[0]

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(features)[0]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(features)[0]
        proba = _softmax(np.asarray(scores))
    else:
        proba = np.ones(len(model.classes_)) / len(model.classes_)

    return prediction, proba


def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over a 1-D array."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
