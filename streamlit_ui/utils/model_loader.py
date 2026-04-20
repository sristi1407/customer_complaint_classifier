"""Model loader with Streamlit caching for the Customer Complaint Classifier."""

import joblib
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer

from utils.config import (
    MODEL_PATHS,
    MODELS,
    MODELS_NEED_VECTORIZER,
    PREPROCESSED_DATA_PATH,
    VECTORIZER_CONFIGS,
)


@st.cache_resource(show_spinner="Loading models…")
def load_all_models() -> dict:
    """Load all trained models from disk and return as a dict.

    Returns
    -------
    dict
        Keys are model identifiers (e.g. ``'svm'``), values are the loaded
        model objects.  If a model fails to load, its value is ``None``.
    """
    loaded: dict = {}
    for key, path in MODEL_PATHS.items():
        try:
            loaded[key] = joblib.load(path)
        except FileNotFoundError:
            st.warning(f"⚠️ Model file not found: `{path}`")
            loaded[key] = None
        except Exception as exc:  # noqa: BLE001
            st.error(f"❌ Failed to load **{MODELS.get(key, key)}**: {exc}")
            loaded[key] = None
    return loaded


def get_model(models: dict, key: str):
    """Return a single model by key, or None if unavailable."""
    return models.get(key)


@st.cache_resource(show_spinner="Preparing text vectorizers…")
def load_required_vectorizers() -> dict:
    """Build and cache TF-IDF vectorizers required by non-pipeline models."""
    loaded: dict = {key: None for key in MODELS_NEED_VECTORIZER}

    try:
        df = pd.read_csv(PREPROCESSED_DATA_PATH)
        text_series = (df["Issue"].fillna("") + " " + df["Sub-issue"].fillna(""))
    except FileNotFoundError:
        st.warning(f"⚠️ Vectorizer training data not found: `{PREPROCESSED_DATA_PATH}`")
        return loaded
    except KeyError as exc:
        st.error(f"❌ Missing expected text columns in vectorizer data: {exc}")
        return loaded
    except Exception as exc:  # noqa: BLE001
        st.error(f"❌ Failed to load vectorizer training data: {exc}")
        return loaded

    for model_key in MODELS_NEED_VECTORIZER:
        cfg = VECTORIZER_CONFIGS.get(model_key, {})
        try:
            vectorizer = TfidfVectorizer(**cfg)
            vectorizer.fit(text_series)
            loaded[model_key] = vectorizer
        except Exception as exc:  # noqa: BLE001
            st.error(f"❌ Failed to prepare vectorizer for **{MODELS.get(model_key, model_key)}**: {exc}")
            loaded[model_key] = None

    return loaded

