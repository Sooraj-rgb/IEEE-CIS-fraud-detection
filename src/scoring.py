"""Prepare raw transactions and score them with the saved fraud model."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
try:
    from src.features import apply_preprocessor
except ModuleNotFoundError:
    from features import apply_preprocessor

HELPER = ["TransactionDT", "TransactionAmt", "card1", "card2", "addr1", "D1", "P_emaildomain"]
REQUIRED = ["TransactionDT", "TransactionAmt", "card1"]


def prepare(df: pd.DataFrame, model: Any, state: Mapping[str, Any]) -> pd.DataFrame:
    """Normalize raw columns and fill every input expected by feature engineering."""
    frame = df.drop(columns=[c for c in ("isFraud", "TransactionID") if c in df.columns]).copy()
    frame.columns = [column.replace("-", "_") for column in frame.columns]
    expected = list(dict.fromkeys(list(model.feature_name_) + HELPER + list(state["cats"])))
    missing = [column for column in expected if column not in frame.columns]
    if missing:
        additions = pd.DataFrame(np.nan, index=frame.index, columns=missing)
        frame = pd.concat([frame, additions], axis=1)
    text_columns = set(state["cats"])
    for column in frame.columns:
        if column in text_columns:
            frame[column] = frame[column].astype("object")
        elif column == "card1":
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("int64")
        else:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float64")
    return frame


def validate_required(df: pd.DataFrame) -> pd.Series:
    """Return a row mask for transactions with all required raw fields present."""
    mask = pd.Series(True, index=df.index, dtype=bool)
    for column in REQUIRED:
        if column not in df.columns:
            return pd.Series(False, index=df.index, dtype=bool)
        mask &= df[column].notna()
    return mask


def transformed_frame(df: pd.DataFrame, model: Any, state: Mapping[str, Any]) -> pd.DataFrame:
    """Return the complete feature-engineered frame before model-column selection."""
    return apply_preprocessor(prepare(df, model, state), state)


def model_frame(df: pd.DataFrame, model: Any, state: Mapping[str, Any]) -> pd.DataFrame:
    """Return engineered features in the exact order expected by the model."""
    return transformed_frame(df, model, state).reindex(columns=list(model.feature_name_))


def score_frame(
    df: pd.DataFrame,
    model: Any,
    state: Mapping[str, Any],
    contrib: bool = False,
) -> pd.DataFrame:
    """Score raw rows and optionally attach per-feature log-odds contributions."""
    features = model_frame(df, model, state)
    probabilities = model.predict_proba(features)[:, 1]
    result = pd.DataFrame({"fraud_probability": probabilities}, index=df.index)
    if contrib:
        contributions = np.asarray(model.predict(features, pred_contrib=True))
        if contributions.ndim == 1:
            contributions = contributions.reshape(1, -1)
        contributions = contributions[:, :-1]
        if contributions.shape[1] != len(model.feature_name_):
            raise ValueError("The model returned an unexpected contribution shape.")
        contribution_frame = pd.DataFrame(
            contributions,
            columns=[f"contrib:{feature}" for feature in model.feature_name_],
            index=df.index,
        )
        result = pd.concat([result, contribution_frame], axis=1)
    return result


def risk_band(probability: float, review: float, high: float) -> str:
    """Map a probability to its configured Low, Review, or High risk band."""
    if probability >= high:
        return "High"
    if probability >= review:
        return "Review"
    return "Low"


def group_features_available(x_row: Any) -> bool:
    """Return whether history-based pseudo-user features exist for this row."""
    value = x_row.get("uid_count", np.nan)
    return not pd.isna(value)
