"""
Model loader — loads a trained model artifact from disk.

After completing Part 1 (the notebook), save your model to:
    app/artifacts/model.pkl

This module provides:
    load_model()  — loads and returns the saved model
    predict()     — stub for you to implement
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

_ARTIFACT_PATH = Path(__file__).parent / "artifacts" / "model.pkl"


def load_model() -> Any:
    """
    Load the trained model artifact from app/artifacts/model.pkl.

    Raises:
        FileNotFoundError: if the artifact has not been saved yet.
            Complete Part 1 of the notebook and save your model first.
    """
    if not _ARTIFACT_PATH.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {_ARTIFACT_PATH}.\n"
            "Complete the modelling notebook (notebooks/modelling.ipynb) and "
            "save your trained model to app/artifacts/model.pkl before starting Part 2."
        )
    with _ARTIFACT_PATH.open("rb") as f:
        return pickle.load(f)


def predict(model: Any, record: dict) -> dict:
    """
    Run the loaded model on a single record and return a structured prediction.

    TODO — implement this function.

    Args:
        model:  The loaded model returned by load_model().
        record: A dict matching the fields in records.csv (without loss_ratio
                and is_loss_making).

    Returns:
        A dict with at minimum:
            is_loss_making_prediction  (bool)   — the model's prediction
            confidence                 (float)  — probability of the positive class, 0–1
            top_features               (list)   — names of the most influential features

    Hints:
        - You will need to apply the same feature engineering and preprocessing
          you used in the notebook before calling model.predict() / predict_proba().
        - For feature importance, consider using model.feature_importances_ (tree models)
          or model.coef_ (linear models), or SHAP values for more detail.
        - Return confidence as the probability of is_loss_making=True, not just 0/1.
    """
    raise NotImplementedError(
        "Implement predict() in app/model.py. "
        "See the docstring above for the expected return format."
    )
