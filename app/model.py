"""
Model loader — loads a trained model artifact from disk.

After completing Part 1 (the notebook), save your model to:
    app/artifacts/model.pkl

This module provides:
    load_model()  — loads and returns the saved model
    predict()     — runs the model on a single record and returns a structured result
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shap

_ARTIFACT_PATH = Path(__file__).parent / "artifacts" / "model.pkl"

# Raw numeric columns that must exist before feature engineering
_RAW_NUMERIC_COLS = ["limit", "premium", "prior_claims", "years_trading"]
# Categorical columns handled by the saved TargetEncoder
_CAT_COLS = ["risk_type", "territory", "industry", "broker"]
# All expected raw input columns (excluding target and loss_ratio)
_EXPECTED_COLS = _RAW_NUMERIC_COLS + _CAT_COLS

# Warnings that start with this prefix mean a raw field looks implausible; the agent
# surfaces them explicitly to the underwriter (values are not imputed at runtime).
EXTREME_RAW_WARNING_PREFIX = "Extreme raw value —"


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

    Args:
        model:  The artifact dict returned by load_model().
        record: A dict matching the fields in records.csv (without loss_ratio
                and is_loss_making).

    Returns:
        A dict containing:
            is_loss_making_prediction  (bool)   — the model's binary prediction
            confidence                 (float)  — P(loss-making), 0–1
            top_features               (list)   — feature names ranked by |SHAP value|
            shap_values                (dict)   — {feature: shap_value} for this record
            warnings                   (list)   — data quality / OOD notices for the caller
                (implausible raw values: out-of-range vs training min–max, impossible
                negatives, or unseen categories — flagged, not imputed)
    """
    clf = model["model"]
    encoder = model["encoder"]
    feature_names = model["feature_names"]
    raw_medians = model["raw_medians"]
    feature_medians = model["feature_medians"]
    outlier_bounds = model["outlier_bounds"]

    warnings: list[str] = []

    # ------------------------------------------------------------------
    # 1. Build a single-row DataFrame from the raw record
    # ------------------------------------------------------------------
    row = dict(record)

    # Normalise categoricals (same as notebook load step)
    for col in _CAT_COLS:
        if col in row and row[col] is not None:
            row[col] = str(row[col]).strip().lower()

    # ------------------------------------------------------------------
    # 2. Check for missing / unexpected columns and impute
    # ------------------------------------------------------------------
    for col in _EXPECTED_COLS:
        if col not in row or row[col] is None or (
            isinstance(row[col], float) and np.isnan(row[col])
        ):
            if col in _RAW_NUMERIC_COLS:
                imputed = raw_medians.get(col, 0.0)
                warnings.append(
                    f"Missing value for '{col}' — imputed with training median ({imputed})."
                )
                row[col] = imputed
            else:
                warnings.append(
                    f"Missing value for '{col}' — imputed with 'unknown'."
                )
                row[col] = "unknown"

    # ------------------------------------------------------------------
    # 3. Flag implausible raw values: impossible (e.g. negative) or outside the
    #    training min–max range / unknown category (stored in artifact). Values
    #    are never imputed here — only warnings.
    # ------------------------------------------------------------------
    raw_numeric_bounds: dict[str, Any] = model.get("raw_numeric_bounds") or {}
    raw_categorical_values: dict[str, Any] = model.get("raw_categorical_values") or {}

    for col in _RAW_NUMERIC_COLS:
        val = row.get(col)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            continue
        try:
            v = float(val)
        except (TypeError, ValueError):
            continue
        if v < 0:
            warnings.append(
                f"{EXTREME_RAW_WARNING_PREFIX} '{col}' is {val} (negative values are impossible "
                "for this field). The model was run using the value as submitted; verify before "
                "relying on the score."
            )
            continue
        bounds = raw_numeric_bounds.get(col)
        if bounds is not None:
            lo, hi = float(bounds[0]), float(bounds[1])
            if v < lo or v > hi:
                warnings.append(
                    f"{EXTREME_RAW_WARNING_PREFIX} '{col}' is {val}, outside the training "
                    f"min–max range [{lo:g}, {hi:g}]. The model was run using the value as "
                    "submitted; verify before relying on the score."
                )

    for col in _CAT_COLS:
        val = row.get(col)
        if val is None:
            continue
        known = raw_categorical_values.get(col)
        if not known:
            continue
        sval = str(val).strip().lower()
        known_set = known if isinstance(known, set) else set(known)
        if sval not in known_set:
            warnings.append(
                f"{EXTREME_RAW_WARNING_PREFIX} '{col}' value {val!r} does not appear in the "
                "training data for this category. The model was run using the value as "
                "submitted; verify before relying on the score."
            )

    # ------------------------------------------------------------------
    # 4. Target-encode categoricals using the saved encoder
    # ------------------------------------------------------------------
    df_raw = pd.DataFrame([row])[_EXPECTED_COLS]
    df_enc = encoder.transform(df_raw)

    # ------------------------------------------------------------------
    # 5. Feature engineering (must mirror the notebook exactly)
    # ------------------------------------------------------------------
    df_enc["premium_rate"] = (
        df_enc["premium"] / df_enc["limit"].replace(0, np.nan)
    ).fillna(0)
    df_enc["prior_claim_rate"] = df_enc["prior_claims"] / df_enc["years_trading"].replace(0, 1)

    # ------------------------------------------------------------------
    # 6. Select and order features to match what the model was trained on
    # ------------------------------------------------------------------
    missing_engineered = [f for f in feature_names if f not in df_enc.columns]
    for col in missing_engineered:
        imputed = feature_medians.get(col, 0.0)
        warnings.append(
            f"Engineered feature '{col}' could not be computed — "
            f"imputed with training median ({imputed})."
        )
        df_enc[col] = imputed

    X_input = df_enc[feature_names]

    # ------------------------------------------------------------------
    # 7. Outlier / out-of-distribution flagging
    # ------------------------------------------------------------------
    for feat in feature_names:
        val = X_input[feat].iloc[0]
        lo, hi = outlier_bounds.get(feat, (None, None))
        if lo is not None and (val < lo or val > hi):
            warnings.append(
                f"Feature '{feat}' value {val:.4g} is outside the training range "
                f"[{lo:.4g}, {hi:.4g}] (5th–95th percentile on the training split). "
                "Prediction may be less reliable."
            )

    # ------------------------------------------------------------------
    # 8. Predict
    # ------------------------------------------------------------------
    confidence = float(clf.predict_proba(X_input)[0, 1])
    threshold = float(model.get("threshold", 0.5))
    is_loss_making_prediction = confidence >= threshold

    # ------------------------------------------------------------------
    # 9. SHAP explanations — average across the 5 calibration folds
    # ------------------------------------------------------------------
    shap_vals_list = []
    for cc in clf.calibrated_classifiers_:
        lr = cc.estimator.named_steps["clf"]
        scaler = cc.estimator.named_steps["scaler"]
        X_scaled = scaler.transform(X_input)
        background = np.zeros((1, X_scaled.shape[1]))
        explainer = shap.LinearExplainer(
            lr, background, feature_names=feature_names
        )
        shap_vals_list.append(explainer.shap_values(X_scaled)[0])

    shap_array = np.mean(shap_vals_list, axis=0)
    shap_dict = {feat: float(val) for feat, val in zip(feature_names, shap_array)}
    top_features = sorted(shap_dict, key=lambda f: abs(shap_dict[f]), reverse=True)

    return {
        "is_loss_making_prediction": is_loss_making_prediction,
        "confidence": round(confidence, 4),
        "top_features": top_features,
        "shap_values": {f: round(shap_dict[f], 6) for f in top_features},
        "warnings": warnings,
    }
