"""Complaint Classification page for the Customer Complaint Classifier UI.

Run with: streamlit run streamlit_ui/main.py
"""

import json
import os
import sys

import numpy as np
import streamlit as st

# Ensure the streamlit_ui directory is on the path so relative imports work
_UI_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _UI_DIR not in sys.path:
    sys.path.insert(0, _UI_DIR)

from utils.config import (
    CATEGORIES,
    INPUT_MAX_LENGTH,
    INPUT_MIN_LENGTH,
    MODELS,
    MODELS_NEED_VECTORIZER,
    ASSETS_DIR,
)
from utils.explanations import get_explanation
from utils.helpers import (
    compute_consensus,
    confidence_color,
    confidence_emoji,
    confidence_label,
    sort_predictions,
    validate_input,
)
from utils.model_loader import load_all_models
from utils.text_processor import (
    get_preprocessing_info,
    predict_baseline,
    predict_svm_pipeline,
)
from utils.visualizations import top5_bar


# ---------------------------------------------------------------------------
# Page configuration (only set when this file is the entry point)
# ---------------------------------------------------------------------------
def _configure_page():
    try:
        st.set_page_config(
            page_title="Complaint Classifier",
            page_icon="🎯",
            layout="wide",
            initial_sidebar_state="expanded",
        )
    except st.errors.StreamlitAPIException:
        pass  # Already configured by main.py


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_samples() -> list[dict]:
    path = os.path.join(ASSETS_DIR, "sample_complaints.json")
    try:
        with open(path, "r") as fh:
            return json.load(fh)
    except Exception:
        return []


def _predict_model(key: str, model, text: str) -> dict:
    """Run prediction for a single model.

    Returns
    -------
    dict with keys:
        ``success`` (bool),
        ``prediction`` (str | None),
        ``confidence`` (float | None),
        ``top5`` (list[(str, float)] | None),
        ``error`` (str | None)
    """
    if model is None:
        return {"success": False, "prediction": None, "confidence": None,
                "top5": None, "error": "Model file not loaded."}

    try:
        if key == "baseline":
            pred, proba = predict_baseline(model, text)
            classes = list(model.classes_)
        elif key == "svm":
            pred, proba = predict_svm_pipeline(model, text)
            classes = list(model.classes_)
        elif key in MODELS_NEED_VECTORIZER:
            return {
                "success": False,
                "prediction": None,
                "confidence": None,
                "top5": None,
                "error": (
                    "⚠️ **Vectorizer not saved** — this model requires its original "
                    "TF-IDF vectorizer to make live predictions.  "
                    "See the **Dashboard** page for its historical performance."
                ),
            }
        else:
            return {
                "success": False, "prediction": None, "confidence": None,
                "top5": None, "error": "Unknown model type."
            }

        top5 = sort_predictions(classes, proba)
        confidence = float(top5[0][1]) if top5 else 0.0

        return {
            "success": True,
            "prediction": pred,
            "confidence": confidence,
            "top5": top5,
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "success": False,
            "prediction": None,
            "confidence": None,
            "top5": None,
            "error": str(exc),
        }


def _render_model_card(key: str, model, results: dict):
    """Render a single model prediction card."""
    display_name = MODELS[key]
    explanation = get_explanation(key)
    icon = explanation.get("icon", "🤖")

    with st.container():
        st.markdown(
            f"""
            <div style="border:1px solid #e0e0e0; border-radius:8px; padding:16px;
                        background:#fafafa; margin-bottom:8px;">
                <h4 style="margin:0 0 8px 0;">{icon} {display_name}</h4>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if not results["success"]:
            st.error(results["error"] or "Prediction failed.")
            # Still show explanation
            with st.expander("📖 How It Works"):
                if explanation:
                    st.markdown(f"**How:** {explanation['how']}")
                    st.markdown(f"**✅ Strength:** {explanation['strength']}")
                    st.markdown(f"**⚠️ Weakness:** {explanation['weakness']}")
            return

        pred = results["prediction"]
        conf = results["confidence"]
        top5 = results["top5"]
        color = confidence_color(conf)
        emoji = confidence_emoji(conf)
        label = confidence_label(conf)

        # Prediction + confidence metric
        col_a, col_b = st.columns([2, 1])
        with col_a:
            st.markdown(f"**🎯 Predicted Category:**")
            st.markdown(
                f"<span style='font-size:1.05rem; font-weight:600; color:{color};'>"
                f"{pred}</span>",
                unsafe_allow_html=True,
            )
        with col_b:
            st.metric(
                label=f"{emoji} Confidence",
                value=f"{conf:.1%}",
                delta=label,
            )

        st.progress(conf)

        # Top-5 predictions
        with st.expander("📊 Top 5 Predictions", expanded=True):
            if top5:
                for rank, (cat, prob) in enumerate(top5[:5], 1):
                    bar_col = confidence_color(prob)
                    st.markdown(
                        f"{rank}. **{cat}** — "
                        f"<span style='color:{bar_col};font-weight:600;'>{prob:.2%}</span>",
                        unsafe_allow_html=True,
                    )
                st.plotly_chart(
                    top5_bar([c for c, _ in top5[:5]], [p for _, p in top5[:5]]),
                    use_container_width=True,
                    config={"displayModeBar": False},
                )

        # Model explanation
        with st.expander("📖 How It Works"):
            if explanation:
                st.markdown(f"**How:** {explanation['how']}")
                st.markdown(f"**✅ Strength:** {explanation['strength']}")
                st.markdown(f"**⚠️ Weakness:** {explanation['weakness']}")

        # Why this prediction
        with st.expander("💡 Why This Prediction"):
            if key == "baseline":
                st.markdown(
                    "The Baseline model always predicts the most frequent category in "
                    "the training data (**Mortgage**) regardless of the complaint text. "
                    "This prediction does not reflect your specific complaint."
                )
            elif key == "svm":
                st.markdown(
                    f"The SVM found that your complaint text most closely matches the "
                    f"**{pred}** category based on TF-IDF features.  "
                    f"Confidence is **{conf:.1%}** ({label})."
                )


def _render_consensus(all_results: dict[str, dict]):
    """Render the model consensus table."""
    predictions_map = {k: v.get("prediction") for k, v in all_results.items()}
    consensus = compute_consensus(predictions_map)

    st.subheader("⚖️ Model Consensus")

    # Build table rows
    consensus_class = consensus["consensus_class"]
    table_rows = []
    for key, display in MODELS.items():
        pred = predictions_map.get(key)
        if pred is None:
            agree_icon = "➖"
        elif pred == consensus_class:
            agree_icon = "✅"
        else:
            agree_icon = "❌"
        table_rows.append(
            {"Model": display, "Prediction": pred or "—", "Agrees": agree_icon}
        )

    import pandas as pd
    df = pd.DataFrame(table_rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    count = consensus["agreement_count"]
    valid = consensus["valid_models"]
    total = consensus["total_models"]
    label = consensus["confidence_label"]

    if consensus_class:
        color = (
            "#2ca02c" if label == "HIGH"
            else "#ff9900" if label == "MEDIUM"
            else "#d62728"
        )
        unavail = total - valid
        unavail_note = f" ({unavail} model(s) unavailable)" if unavail > 0 else ""
        st.markdown(
            f"""
            <div style="border:2px solid {color}; border-radius:8px; padding:12px;
                        background:{color}22; margin-top:8px;">
                <b>CONSENSUS:</b> {consensus_class} &nbsp;|&nbsp;
                <b>{count}/{valid}</b> available models agree{unavail_note} &nbsp;|&nbsp;
                Confidence: <b style="color:{color};">{label}</b>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.info("No consensus could be reached (no successful predictions).")


# ---------------------------------------------------------------------------
# Main page function
# ---------------------------------------------------------------------------

def run():
    """Render the classification page."""
    _configure_page()

    # Header
    st.title("🎯 Customer Complaint Classifier")
    st.markdown(
        "*Multi-class Classification using Machine Learning — "
        "18 Financial Product Categories*"
    )
    st.markdown("---")

    # Load models once (cached)
    models = load_all_models()
    samples = _load_samples()

    # -----------------------------------------------------------------------
    # Sidebar: sample loader
    # -----------------------------------------------------------------------
    with st.sidebar:
        st.header("📋 Sample Complaints")
        sample_titles = [s["title"] for s in samples]
        selected_sample = st.selectbox("Load a sample complaint:", ["(none)"] + sample_titles)

    # -----------------------------------------------------------------------
    # Input section
    # -----------------------------------------------------------------------
    st.subheader("📝 Enter Your Complaint")

    # Session state for input text
    if "complaint_text" not in st.session_state:
        st.session_state.complaint_text = ""

    # Load sample if selected
    if selected_sample != "(none)":
        for s in samples:
            if s["title"] == selected_sample:
                st.session_state.complaint_text = s["text"]
                break

    complaint_text = st.text_area(
        label="Complaint text",
        value=st.session_state.complaint_text,
        placeholder="Describe your complaint here…",
        height=160,
        max_chars=INPUT_MAX_LENGTH,
        label_visibility="collapsed",
        key="complaint_input",
    )

    char_count = len(complaint_text)
    count_color = "red" if char_count < INPUT_MIN_LENGTH else "green"
    st.markdown(
        f"<small style='color:{count_color};'>"
        f"Characters: {char_count} / {INPUT_MAX_LENGTH} "
        f"(min {INPUT_MIN_LENGTH})</small>",
        unsafe_allow_html=True,
    )

    col_classify, col_clear, _ = st.columns([1, 1, 4])
    with col_classify:
        classify_clicked = st.button("🚀 Classify Complaint", type="primary")
    with col_clear:
        if st.button("🔄 Clear"):
            st.session_state.complaint_text = ""
            st.rerun()

    # -----------------------------------------------------------------------
    # Preprocessing preview
    # -----------------------------------------------------------------------
    if complaint_text.strip():
        with st.expander("🔍 View Preprocessing Steps"):
            info = get_preprocessing_info(complaint_text)
            pcol1, pcol2 = st.columns(2)
            pcol1.metric("Characters", info["char_count"])
            pcol2.metric("Token Count", info["token_count"])
            st.markdown("**Cleaned Text:**")
            st.code(info["cleaned"], language=None)

    # -----------------------------------------------------------------------
    # Classification
    # -----------------------------------------------------------------------
    if classify_clicked:
        valid, msg = validate_input(complaint_text)
        if not valid:
            st.warning(msg)
        else:
            st.markdown("---")
            st.subheader("🔮 Prediction Results")

            with st.spinner("Running predictions on all models…"):
                all_results = {key: _predict_model(key, models.get(key), complaint_text)
                               for key in MODELS}

            # Display model cards in a 2-column grid
            model_keys = list(MODELS.keys())
            n = len(model_keys)

            for row_start in range(0, n, 2):
                cols = st.columns(2)
                for col_idx, model_key in enumerate(model_keys[row_start:row_start + 2]):
                    with cols[col_idx]:
                        _render_model_card(model_key, models.get(model_key), all_results[model_key])

            st.markdown("---")
            _render_consensus(all_results)


# ---------------------------------------------------------------------------
# Entry point when file is run directly (multi-page Streamlit navigation)
# ---------------------------------------------------------------------------
run()
