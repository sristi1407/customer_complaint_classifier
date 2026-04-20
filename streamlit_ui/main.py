"""Home / landing page for the Customer Complaint Classifier Streamlit application.

Streamlit automatically discovers pages in the ``pages/`` subdirectory.
Use the sidebar to navigate between:
  - 📝 Classification  →  pages/01_classification.py
  - 📊 Dashboard       →  pages/02_dashboard.py

Run with:
    streamlit run streamlit_ui/main.py
"""

import os
import sys

import streamlit as st

# Ensure the streamlit_ui directory is on the Python path so that pages and
# utils can resolve ``from utils.xxx import ...`` correctly.
_UI_DIR = os.path.dirname(os.path.abspath(__file__))
if _UI_DIR not in sys.path:
    sys.path.insert(0, _UI_DIR)

# ---------------------------------------------------------------------------
# Page configuration — must be the FIRST Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Customer Complaint Classifier",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Home page content
# ---------------------------------------------------------------------------
st.title("🎯 Customer Complaint Classifier")
st.markdown(
    "### Multi-class ML Classification · 18 Financial Product Categories"
)
st.markdown("---")

col1, col2, col3 = st.columns(3)
with col1:
    st.info(
        "#### 📝 Classification\n\n"
        "Enter a customer complaint and get instant predictions from all 6 ML models.\n\n"
        "→ **Use the sidebar to navigate**"
    )
with col2:
    st.success(
        "#### 📊 Dashboard\n\n"
        "Compare model performance across accuracy, precision, recall, and F1-score "
        "with interactive charts."
    )
with col3:
    st.warning(
        "#### 📂 Data\n\n"
        "366,945 real CFPB consumer complaints across 18 financial product categories "
        "from the Consumer Financial Protection Bureau."
    )

st.markdown("---")
st.markdown(
    """
    ### 🤖 Available Models

    | Model | Type | Live Predictions |
    |-------|------|-----------------|
    | 📊 Baseline (Majority Class) | DummyClassifier | ✅ |
    | 🔢 Naive Bayes | MultinomialNB | ✅ |
    | ⚡ Support Vector Machine | LinearSVC Pipeline | ✅ |
    | 📉 Logistic Regression | LogisticRegression Pipeline | ✅ |
    | 🌳 Decision Tree | DecisionTreeClassifier | ✅ |
    | 🌲 Random Forest | RandomForestClassifier | ✅ |

    > **Note:** Naive Bayes, Decision Tree, and Random Forest were originally trained with
    > separate TF-IDF vectorizers. The app now reconstructs those vectorizers from the
    > preprocessed complaint dataset, enabling live predictions for all six models.

    ---
    **How to use:** Use the **sidebar** on the left to navigate between pages.
    """
)
