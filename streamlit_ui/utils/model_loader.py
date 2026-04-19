"""Model loader with Streamlit caching for the Customer Complaint Classifier."""

import joblib
import streamlit as st

from utils.config import MODEL_PATHS, MODELS


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
