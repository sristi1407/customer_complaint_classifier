import sys
from pathlib import Path

repo = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo / "streamlit_ui"))

from utils.model_loader import load_all_models, load_required_vectorizers
from utils.text_processor import (
    predict_baseline,
    predict_svm_pipeline,
    predict_with_external_vectorizer,
)


def main() -> None:
    text = (
        "I was charged unauthorized fees on my credit card, and customer service "
        "did not resolve my dispute after multiple calls."
    )

    models = load_all_models()
    vectorizers = load_required_vectorizers()

    for key in ["baseline", "svm", "logistic_regression", "naive_bayes", "decision_tree", "random_forest"]:
        model = models.get(key)
        if model is None:
            print(f"{key:14s} -> model not loaded")
            continue

        if key == "baseline":
            pred, proba = predict_baseline(model, text)
        elif key in {"svm", "logistic_regression"}:
            pred, proba = predict_svm_pipeline(model, text)
        else:
            vec = vectorizers.get(key)
            if vec is None:
                print(f"{key:14s} -> vectorizer unavailable")
                continue
            pred, proba = predict_with_external_vectorizer(model, vec, text)

        print(f"{key:14s} -> {pred} ({float(max(proba)):.3f})")


if __name__ == "__main__":
    main()

