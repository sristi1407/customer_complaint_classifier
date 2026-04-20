"""Configuration constants for the Customer Complaint Classifier Streamlit UI."""

import os

# ---------------------------------------------------------------------------
# Paths  (relative to project root, where `streamlit run` is invoked)
# ---------------------------------------------------------------------------
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL_PATHS = {
    "baseline": os.path.join(_BASE_DIR, "models", "baseline", "baseline_model.joblib"),
    "naive_bayes": os.path.join(_BASE_DIR, "models", "naivebayes", "naivebayes_model.joblib"),
    "svm": os.path.join(_BASE_DIR, "models", "SVM", "svm_pipeline.joblib"),
    "logistic_regression": os.path.join(
        _BASE_DIR, "models", "LogisticRegression", "logistic_regression_model.joblib"
    ),
    "decision_tree": os.path.join(_BASE_DIR, "models", "DecisionTree", "decision_tree_model.joblib"),
    "random_forest": os.path.join(_BASE_DIR, "models", "RandomForestRegressor", "random_forest_model.joblib"),
}

RESULTS_PATHS = {
    "kunal_csv": os.path.join(_BASE_DIR, "results", "Kunal", "kunal_model_results.csv"),
    "svm_csv": os.path.join(_BASE_DIR, "results", "Dev", "svm_evaluation_metrics.csv"),
    "dt_report": os.path.join(_BASE_DIR, "results", "Sristi", "decision_tree_evaluation_report.txt"),
    "rf_report": os.path.join(_BASE_DIR, "results", "Sristi", "random_forest_evaluation_report.txt"),
    "svm_report": os.path.join(_BASE_DIR, "results", "Dev", "svm_classification_report.txt"),
    "lr_report": os.path.join(_BASE_DIR, "results", "Dev", "logistic_regression_classification_report.txt"),
}

PREPROCESSED_DATA_PATH = os.path.join(
    _BASE_DIR, "data", "preprocess_data", "complaints_cleaned.csv"
)

ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")

# ---------------------------------------------------------------------------
# Model display information
# ---------------------------------------------------------------------------
MODELS = {
    "baseline": "Baseline (Majority Class)",
    "naive_bayes": "Naive Bayes",
    "svm": "Support Vector Machine (SVM)",
    "logistic_regression": "Logistic Regression",
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
}

# Whether a model has its own vectorizer built in (Pipeline) or needs external vectorization
MODEL_HAS_PIPELINE = {
    "baseline": False,   # DummyClassifier — no vectorization needed
    "naive_bayes": False,  # MultinomialNB — vectorizer NOT saved separately
    "svm": True,          # Pipeline(TfidfVectorizer + LinearSVC)
    "logistic_regression": True,  # Pipeline(TfidfVectorizer + LogisticRegression)
    "decision_tree": False,  # DecisionTreeClassifier — vectorizer NOT saved separately
    "random_forest": False,  # RandomForestClassifier — vectorizer NOT saved separately
}

# Models whose vectorizers are unavailable (predictions require the original vectorizer)
MODELS_NEED_VECTORIZER = {"naive_bayes", "decision_tree", "random_forest"}

# Rebuild settings for vectorizers that were not serialized with model artifacts.
VECTORIZER_CONFIGS = {
    "naive_bayes": {
        "max_features": 10000,
        "stop_words": "english",
        "ngram_range": (1, 2),
    },
    "decision_tree": {
        "max_features": 5000,
        "stop_words": "english",
        "ngram_range": (1, 1),
    },
    "random_forest": {
        "max_features": 5000,
        "stop_words": "english",
        "ngram_range": (1, 1),
    },
}

# ---------------------------------------------------------------------------
# 18 Financial Product Categories (from CFPB dataset / baseline model classes)
# ---------------------------------------------------------------------------
CATEGORIES = [
    "Bank account or service",
    "Checking or savings account",
    "Consumer Loan",
    "Credit card",
    "Credit card or prepaid card",
    "Credit reporting",
    "Credit reporting, credit repair services, or other personal consumer reports",
    "Debt collection",
    "Money transfer, virtual currency, or money service",
    "Money transfers",
    "Mortgage",
    "Other financial service",
    "Payday loan",
    "Payday loan, title loan, or personal loan",
    "Prepaid card",
    "Student loan",
    "Vehicle loan or lease",
    "Virtual currency",
]

# ---------------------------------------------------------------------------
# Color scheme
# ---------------------------------------------------------------------------
COLORS = {
    "primary": "#1f77b4",
    "secondary": "#ff7f0e",
    "success": "#2ca02c",
    "warning": "#ff9900",
    "danger": "#d62728",
    "light_bg": "#f0f2f6",
}

# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------
INPUT_MIN_LENGTH = 20
INPUT_MAX_LENGTH = 5000

# ---------------------------------------------------------------------------
# Confidence thresholds
# ---------------------------------------------------------------------------
CONFIDENCE_THRESHOLD_HIGH = 0.75
CONFIDENCE_THRESHOLD_MEDIUM = 0.50

# ---------------------------------------------------------------------------
# Sample complaints (plain text references — full JSON in assets/)
# ---------------------------------------------------------------------------
SAMPLE_COMPLAINTS = [
    "I was charged unauthorized fees on my credit card without my consent. "
    "I never signed up for these services and want an immediate refund.",

    "My mortgage payment was incorrectly processed last month. The bank applied "
    "the payment to the wrong account and is now reporting a late payment on my credit.",

    "I have multiple incorrect entries on my credit report that have been disputed "
    "numerous times but they refuse to remove or correct the inaccurate information.",

    "A debt collector has been calling me multiple times a day regarding a debt "
    "that I do not owe. They are contacting me at inappropriate hours and being harassing.",

    "My student loan servicer is not applying my payments correctly. Despite making "
    "on-time payments, they keep reporting missed payments to the credit bureaus.",

    "I applied for an auto loan and was approved but the dealer added hidden fees "
    "and changed the interest rate without my knowledge after signing the contract.",
]
