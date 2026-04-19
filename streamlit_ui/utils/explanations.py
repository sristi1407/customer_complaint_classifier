"""Plain-language model explanations for the Customer Complaint Classifier UI."""

MODEL_EXPLANATIONS = {
    "baseline": {
        "icon": "📊",
        "how": (
            "Predicts the most frequently occurring complaint category in the "
            "training data, regardless of the complaint text."
        ),
        "strength": "Provides a reference point for comparing all other models.",
        "weakness": (
            "Does not analyze complaint text at all; accuracy is often very low "
            "because it always predicts the same class."
        ),
    },
    "naive_bayes": {
        "icon": "🔢",
        "how": (
            "Treats each word independently and calculates the probability of "
            "each category based on word frequency using Bayes' theorem."
        ),
        "strength": "Fast, interpretable, and works very well for text classification.",
        "weakness": (
            "Assumes word independence, which may not hold in practice.  "
            "Requires a pre-fitted TF-IDF vectorizer saved alongside the model."
        ),
    },
    "svm": {
        "icon": "⚡",
        "how": (
            "Finds the optimal hyperplane boundary that best separates different "
            "complaint categories in a high-dimensional TF-IDF feature space."
        ),
        "strength": (
            "Handles complex, high-dimensional patterns well and often achieves "
            "very high accuracy on text classification tasks."
        ),
        "weakness": (
            "Slower prediction time than simpler models; less directly interpretable."
        ),
    },
    "decision_tree": {
        "icon": "🌳",
        "how": (
            "Makes a series of binary yes/no decisions based on word presence "
            "or TF-IDF weight to classify complaints into categories."
        ),
        "strength": "Highly interpretable; each decision path can be traced.",
        "weakness": (
            "Prone to overfitting; may perform worse than ensemble methods on "
            "unseen data.  Requires a pre-fitted TF-IDF vectorizer."
        ),
    },
    "random_forest": {
        "icon": "🌲",
        "how": (
            "Combines many decision trees and uses majority voting among them "
            "to produce the final prediction."
        ),
        "strength": "High accuracy and robustness; reduces overfitting compared with a single tree.",
        "weakness": (
            "Less interpretable than a single tree; slower inference.  "
            "Requires a pre-fitted TF-IDF vectorizer saved alongside the model."
        ),
    },
}


def get_explanation(model_key: str) -> dict:
    """Return the explanation dict for a given model key.

    Parameters
    ----------
    model_key:
        One of ``baseline``, ``naive_bayes``, ``svm``, ``decision_tree``,
        ``random_forest``.

    Returns
    -------
    dict with keys ``icon``, ``how``, ``strength``, ``weakness``
    """
    return MODEL_EXPLANATIONS.get(model_key, {})
