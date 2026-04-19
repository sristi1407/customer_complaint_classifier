"""Metrics loading utilities for the Customer Complaint Classifier dashboard."""

import re
import pandas as pd
import os

from utils.config import RESULTS_PATHS, MODELS


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_accuracy_from_report(text: str) -> float | None:
    """Extract overall accuracy from a classification-report text file."""
    # Pattern: "accuracy   0.9777..." or "Model Accuracy: 0.9777"
    m = re.search(r"(?:accuracy|Accuracy)[:\s]+([0-9]\.[0-9]+)", text)
    if m:
        return float(m.group(1))
    return None


def _parse_weighted_metrics(text: str) -> dict:
    """Extract weighted precision / recall / f1 from a classification-report string."""
    # The 'weighted avg' line looks like:
    #   "weighted avg   0.98   0.98   0.98   256471"
    m = re.search(
        r"weighted\s+avg\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)",
        text,
        re.IGNORECASE,
    )
    if m:
        return {
            "precision": float(m.group(1)),
            "recall": float(m.group(2)),
            "f1": float(m.group(3)),
        }
    return {}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_all_metrics() -> pd.DataFrame:
    """Load evaluation metrics from all available result files.

    Combines Kunal's CSV (Baseline + Naive Bayes), Dev's CSV (SVM), and
    Sristi's text reports (Decision Tree, Random Forest) into a single
    unified DataFrame with columns:

    ``Model``, ``Accuracy``, ``Precision``, ``Recall``, ``F1-Score``

    Returns
    -------
    pd.DataFrame
    """
    rows = []

    # ------------------------------------------------------------------
    # 1. Kunal's CSV — Baseline and Naive Bayes
    # ------------------------------------------------------------------
    kunal_path = RESULTS_PATHS["kunal_csv"]
    if os.path.exists(kunal_path):
        try:
            kunal_df = pd.read_csv(kunal_path)
            # Expected columns: Model, Accuracy, Precision (Weighted),
            #                   Recall (Weighted), F1-Score (Weighted)
            for _, row in kunal_df.iterrows():
                model_name = str(row.get("Model", ""))
                # Use the tuned Naive Bayes row if both exist
                if "tuned" in model_name.lower():
                    display_name = "Naive Bayes"
                elif "default" in model_name.lower() or "alpha=1" in model_name:
                    # Skip default NB when tuned version exists
                    continue
                elif "baseline" in model_name.lower():
                    display_name = "Baseline (Majority Class)"
                else:
                    display_name = model_name

                rows.append(
                    {
                        "Model": display_name,
                        "Model Key": _name_to_key(display_name),
                        "Accuracy": float(row.get("Accuracy", 0)),
                        "Precision": float(row.get("Precision (Weighted)", 0)),
                        "Recall": float(row.get("Recall (Weighted)", 0)),
                        "F1-Score": float(row.get("F1-Score (Weighted)", 0)),
                    }
                )
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------
    # 2. Dev's CSV + classification report — SVM
    # ------------------------------------------------------------------
    svm_path = RESULTS_PATHS["svm_csv"]
    svm_report_path = RESULTS_PATHS["svm_report"]
    if os.path.exists(svm_path):
        try:
            svm_df = pd.read_csv(svm_path)
            # Try to extract precision/recall from the classification report text
            svm_precision, svm_recall = None, None
            if os.path.exists(svm_report_path):
                with open(svm_report_path, "r") as fh:
                    svm_report_text = fh.read()
                wm = _parse_weighted_metrics(svm_report_text)
                svm_precision = wm.get("precision")
                svm_recall = wm.get("recall")

            for _, row in svm_df.iterrows():
                accuracy = float(row.get("test_accuracy", 0))
                f1 = float(row.get("test_weighted_f1", 0))
                rows.append(
                    {
                        "Model": "Support Vector Machine (SVM)",
                        "Model Key": "svm",
                        "Accuracy": accuracy,
                        # Use values from classification report if available,
                        # otherwise fall back to the closest available metric.
                        "Precision": svm_precision if svm_precision is not None else f1,
                        "Recall": svm_recall if svm_recall is not None else accuracy,
                        "F1-Score": f1,
                    }
                )
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------
    # 3. Sristi's text reports — Decision Tree & Random Forest
    # ------------------------------------------------------------------
    for key, label in [("dt_report", "Decision Tree"), ("rf_report", "Random Forest")]:
        report_path = RESULTS_PATHS[key]
        if os.path.exists(report_path):
            try:
                with open(report_path, "r") as fh:
                    text = fh.read()
                accuracy = _parse_accuracy_from_report(text)
                wm = _parse_weighted_metrics(text)
                if accuracy is not None:
                    rows.append(
                        {
                            "Model": label,
                            "Model Key": _name_to_key(label),
                            "Accuracy": accuracy,
                            "Precision": wm.get("precision", accuracy),
                            "Recall": wm.get("recall", accuracy),
                            "F1-Score": wm.get("f1", accuracy),
                        }
                    )
            except Exception:  # noqa: BLE001
                pass

    if not rows:
        # Return an empty DataFrame with the expected schema
        return pd.DataFrame(
            columns=["Model", "Model Key", "Accuracy", "Precision", "Recall", "F1-Score"]
        )

    df = pd.DataFrame(rows)
    # Deduplicate: keep last occurrence of each model
    df = df.drop_duplicates(subset="Model", keep="last").reset_index(drop=True)
    # Sort by Accuracy descending
    df = df.sort_values("Accuracy", ascending=False).reset_index(drop=True)
    return df


def _name_to_key(name: str) -> str:
    """Map a display name to its model key."""
    name_lower = name.lower()
    if "baseline" in name_lower or "majority" in name_lower:
        return "baseline"
    if "naive" in name_lower:
        return "naive_bayes"
    if "svm" in name_lower or "vector" in name_lower:
        return "svm"
    if "decision" in name_lower:
        return "decision_tree"
    if "forest" in name_lower:
        return "random_forest"
    if "logistic" in name_lower:
        return "logistic_regression"
    return name_lower.replace(" ", "_")
