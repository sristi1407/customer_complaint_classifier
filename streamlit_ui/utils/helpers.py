"""Utility helpers for the Customer Complaint Classifier Streamlit UI."""

import numpy as np
from collections import Counter

from utils.config import (
    CONFIDENCE_THRESHOLD_HIGH,
    CONFIDENCE_THRESHOLD_MEDIUM,
    COLORS,
    MODELS,
)


# ---------------------------------------------------------------------------
# Confidence helpers
# ---------------------------------------------------------------------------

def confidence_label(confidence: float) -> str:
    """Return a human-readable confidence label.

    Parameters
    ----------
    confidence:
        Float in [0, 1].

    Returns
    -------
    str: ``"HIGH"``, ``"MEDIUM"``, or ``"LOW"``
    """
    if confidence >= CONFIDENCE_THRESHOLD_HIGH:
        return "HIGH"
    if confidence >= CONFIDENCE_THRESHOLD_MEDIUM:
        return "MEDIUM"
    return "LOW"


def confidence_color(confidence: float) -> str:
    """Return a hex color string based on confidence level."""
    if confidence >= CONFIDENCE_THRESHOLD_HIGH:
        return COLORS["success"]
    if confidence >= CONFIDENCE_THRESHOLD_MEDIUM:
        return COLORS["warning"]
    return COLORS["danger"]


def confidence_emoji(confidence: float) -> str:
    """Return an emoji indicator for a confidence level."""
    if confidence >= CONFIDENCE_THRESHOLD_HIGH:
        return "🟢"
    if confidence >= CONFIDENCE_THRESHOLD_MEDIUM:
        return "🟡"
    return "🔴"


# ---------------------------------------------------------------------------
# Consensus helpers
# ---------------------------------------------------------------------------

def compute_consensus(predictions: dict[str, str | None]) -> dict:
    """Compute consensus from a dict of model_key → predicted_class.

    Parameters
    ----------
    predictions:
        Dict mapping model keys to predicted class strings.
        Values may be ``None`` if the model could not predict.

    Returns
    -------
    dict with keys:
        ``consensus_class`` (str | None),
        ``agreement_count`` (int),
        ``valid_models`` (int),
        ``total_models`` (int),
        ``confidence_label`` (str)
    """
    total = len(predictions)
    valid = {k: v for k, v in predictions.items() if v is not None}
    if not valid:
        return {
            "consensus_class": None,
            "agreement_count": 0,
            "valid_models": 0,
            "total_models": total,
            "confidence_label": "LOW",
        }

    counter = Counter(valid.values())
    consensus_class, count = counter.most_common(1)[0]
    n_valid = len(valid)
    ratio = count / n_valid

    return {
        "consensus_class": consensus_class,
        "agreement_count": count,
        "valid_models": n_valid,
        "total_models": total,
        "confidence_label": confidence_label(ratio),
    }


# ---------------------------------------------------------------------------
# Prediction formatting
# ---------------------------------------------------------------------------

def sort_predictions(classes: list[str], proba: np.ndarray) -> list[tuple[str, float]]:
    """Return a list of (class, probability) sorted by probability descending.

    Parameters
    ----------
    classes:
        List of class label strings.
    proba:
        Probability array matching ``classes`` in length.

    Returns
    -------
    list of (str, float) tuples
    """
    pairs = sorted(zip(classes, proba), key=lambda x: x[1], reverse=True)
    return pairs


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def validate_input(text: str, min_len: int = 20, max_len: int = 5000) -> tuple[bool, str]:
    """Validate user complaint input.

    Parameters
    ----------
    text:
        Raw input string.
    min_len, max_len:
        Character length constraints.

    Returns
    -------
    tuple of (is_valid: bool, message: str)
    """
    text = text.strip()
    if not text:
        return False, "⚠️ Please enter a complaint before classifying."
    if len(text) < min_len:
        return False, f"⚠️ Complaint is too short ({len(text)} chars). Minimum is {min_len} characters."
    if len(text) > max_len:
        return False, f"⚠️ Complaint is too long ({len(text)} chars). Maximum is {max_len} characters."
    return True, ""
