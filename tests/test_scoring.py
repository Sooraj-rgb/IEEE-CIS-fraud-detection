"""Acceptance tests for raw transaction preparation and scoring."""

from __future__ import annotations

import os
from pathlib import Path

import joblib
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(os.environ.get("MODEL_PATH", ROOT / "model.joblib"))
STATE_PATH = Path(os.environ.get("STATE_PATH", ROOT / "preprocessor.joblib"))
if not MODEL_PATH.is_absolute():
    MODEL_PATH = ROOT / MODEL_PATH
if not STATE_PATH.is_absolute():
    STATE_PATH = ROOT / STATE_PATH
if not MODEL_PATH.exists() or not STATE_PATH.exists():
    pytest.skip("Local model artifacts are missing; scoring tests need no training run.", allow_module_level=True)

from src.scoring import (  # noqa: E402
    HELPER,
    REQUIRED,
    group_features_available,
    prepare,
    score_frame,
    transformed_frame,
    validate_required,
)

MODEL = joblib.load(MODEL_PATH)
STATE = joblib.load(STATE_PATH)


def test_prepare_adds_model_helper_and_categorical_inputs() -> None:
    """Preparation fills every expected column from only required raw fields."""
    frame = pd.DataFrame([{ "TransactionDT": 8_895_414, "TransactionAmt": 209.95, "card1": 14_265 }])
    prepared = prepare(frame, MODEL, STATE)
    assert set(MODEL.feature_name_).issubset(prepared.columns)
    assert set(HELPER).issubset(prepared.columns)
    assert set(STATE["cats"]).issubset(prepared.columns)
    assert set(REQUIRED).issubset(prepared.columns)


def test_required_only_row_scores_in_probability_range() -> None:
    """A raw row with just the three required fields can be scored."""
    frame = pd.DataFrame([{ "TransactionDT": 8_895_414, "TransactionAmt": 209.95, "card1": 14_265 }])
    probability = float(score_frame(frame, MODEL, STATE)["fraud_probability"].iloc[0])
    assert 0.0 <= probability <= 1.0


def test_prepare_normalizes_hyphenated_identity_headers() -> None:
    """Kaggle identity columns with hyphens become underscore names."""
    frame = pd.DataFrame([{
        "TransactionDT": 8_895_414, "TransactionAmt": 209.95, "card1": 14_265, "id-01": 0.0,
    }])
    prepared = prepare(frame, MODEL, STATE)
    assert "id_01" in prepared.columns
    assert "id-01" not in prepared.columns


def test_validate_required_masks_rows_with_missing_values() -> None:
    """Rows missing any required transaction value are masked out."""
    frame = pd.DataFrame([
        {"TransactionDT": 1, "TransactionAmt": 2.0, "card1": 3},
        {"TransactionDT": 1, "TransactionAmt": None, "card1": 3},
        {"TransactionDT": 1, "TransactionAmt": 2.0, "card1": None},
    ])
    assert validate_required(frame).tolist() == [True, False, False]


def test_unseen_card_has_no_history_features() -> None:
    """A deliberately unknown card does not map to pseudo-user history."""
    frame = pd.DataFrame([{ "TransactionDT": 8_895_414, "TransactionAmt": 209.95, "card1": 999_999_999 }])
    transformed = transformed_frame(frame, MODEL, STATE)
    assert not group_features_available(transformed.iloc[0])


def test_contributions_exclude_the_bias_column() -> None:
    """Contribution output includes one value for each model feature."""
    frame = pd.DataFrame([{ "TransactionDT": 8_895_414, "TransactionAmt": 209.95, "card1": 14_265 }])
    result = score_frame(frame, MODEL, STATE, contrib=True)
    contribution_columns = [column for column in result.columns if column.startswith("contrib:")]
    assert len(contribution_columns) == len(MODEL.feature_name_)
