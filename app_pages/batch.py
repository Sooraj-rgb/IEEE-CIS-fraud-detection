"""CSV batch scoring and optional evaluation page."""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)

from app_pages.common import footer, page_context
from src.scoring import REQUIRED, risk_band, score_frame, validate_required

MAX_ROWS = 100_000
CHUNK_SIZE = 20_000
ACCENT = "#1F4E79"
NEUTRAL = "#9099A5"


def _valid_numeric_rows(frame: pd.DataFrame) -> pd.Series:
    """Require numeric, finite values in every required field."""
    mask = validate_required(frame)
    for column in REQUIRED:
        if column not in frame:
            return pd.Series(False, index=frame.index, dtype=bool)
        numeric = pd.to_numeric(frame[column], errors="coerce")
        mask &= numeric.notna() & np.isfinite(numeric)
    card1 = pd.to_numeric(frame["card1"], errors="coerce")
    mask &= (card1 % 1 == 0)
    return mask


def _read_upload(upload: Any) -> pd.DataFrame:
    """Read an uploaded CSV and provide a clear error for malformed input."""
    try:
        frame = pd.read_csv(upload)
    except Exception as exc:
        raise ValueError("The file could not be read as CSV. Check its delimiter, header row, and encoding.") from exc
    if frame.empty:
        raise ValueError("The uploaded CSV has a header but no transaction rows.")
    frame.columns = [str(column).strip() for column in frame.columns]
    return frame


def _score_upload(frame: pd.DataFrame, model: Any, state: dict, review: float, high: float) -> pd.DataFrame:
    """Score valid rows in bounded chunks and update the progress indicator."""
    if not all(column in frame.columns for column in REQUIRED):
        missing = [column for column in REQUIRED if column not in frame.columns]
        raise ValueError(f"CSV is missing required columns: {', '.join(missing)}.")
    limited = frame.iloc[:MAX_ROWS].copy()
    valid_mask = _valid_numeric_rows(limited)
    valid = limited.loc[valid_mask].copy()
    if valid.empty:
        return pd.DataFrame(columns=["fraud_probability", "risk_band"]), limited, 0
    chunk_count = (len(valid) + CHUNK_SIZE - 1) // CHUNK_SIZE
    progress = st.progress(0.0, text="Scoring transactions…")
    scored: list[pd.DataFrame] = []
    try:
        for chunk_index, start in enumerate(range(0, len(valid), CHUNK_SIZE)):
            chunk = valid.iloc[start : start + CHUNK_SIZE]
            prediction = score_frame(chunk, model, state)
            scored.append(prediction)
            progress.progress((chunk_index + 1) / chunk_count, text=f"Scored {min(start + len(chunk), len(valid)):,} of {len(valid):,} valid rows")
    except Exception as exc:
        progress.empty()
        raise ValueError("Scoring failed. Check that required values are numeric and the CSV uses raw transaction columns.") from exc
    progress.empty()
    results = pd.concat(scored).reindex(valid.index)
    results["risk_band"] = [risk_band(float(p), review, high) for p in results["fraud_probability"]]
    if "TransactionID" in valid.columns:
        results.insert(0, "TransactionID", valid["TransactionID"])
    return results, limited, int((~valid_mask).sum())


def _render_evaluation(source: pd.DataFrame, result: pd.DataFrame, high: float) -> None:
    """Show holdout metrics for valid binary labels in the uploaded rows."""
    labels = pd.to_numeric(source.loc[result.index, "isFraud"], errors="coerce")
    valid_labels = labels.isin([0, 1])
    if not bool(valid_labels.any()):
        st.info("isFraud was present, but no valid 0/1 labels were available for evaluation.")
        return
    y_true = labels.loc[valid_labels].astype(int)
    probabilities = result.loc[y_true.index, "fraud_probability"].astype(float)
    if y_true.nunique() < 2:
        st.info("ROC AUC and PR AUC need both fraud and legitimate labels; this upload has only one class.")
        return
    predicted = (probabilities >= high).astype(int)
    roc = roc_auc_score(y_true, probabilities)
    pr = average_precision_score(y_true, probabilities)
    precision = precision_score(y_true, predicted, zero_division=0)
    recall = recall_score(y_true, predicted, zero_division=0)
    st.subheader("Evaluation")
    a, b, c, d = st.columns(4)
    a.metric("ROC AUC", f"{roc:.4f}")
    b.metric("PR AUC", f"{pr:.4f}")
    c.metric("Precision at high threshold", f"{precision:.2%}")
    d.metric("Recall at high threshold", f"{recall:.2%}")
    matrix = confusion_matrix(y_true, predicted, labels=[0, 1])
    matrix_frame = pd.DataFrame(matrix, index=["Actual legitimate", "Actual fraud"], columns=["Predicted legitimate", "Predicted fraud"])
    st.dataframe(
        matrix_frame,
        column_config={column: st.column_config.NumberColumn(column, format="%d") for column in matrix_frame.columns},
        use_container_width=True,
    )
    st.caption("Rows from the training period look seen to the history features, so scores are optimistic.")


def main() -> None:
    """Render CSV upload, batch scoring, and optional labeled evaluation."""
    st.title("Batch scoring")
    st.caption("Score raw transaction CSVs in chunks and review the highest-risk rows.")
    context = page_context()
    if context is None:
        footer()
        return
    model, state, review, high = context
    upload = st.file_uploader("Upload a CSV", type=["csv"], help="TransactionID and isFraud are optional.")
    if upload is None:
        st.info("Upload a CSV with TransactionDT, TransactionAmt, and card1 to begin.")
        footer()
        return
    try:
        source = _read_upload(upload)
    except ValueError as exc:
        st.error(str(exc))
        footer()
        return
    if len(source) > MAX_ROWS:
        st.info(f"The file has {len(source):,} rows. Scoring its first {MAX_ROWS:,} rows.")
    try:
        result, limited, skipped = _score_upload(source, model, state, review, high)
    except ValueError as exc:
        st.error(str(exc))
        footer()
        return
    st.write(f"Rows scored: {len(result):,} · Rows skipped for missing or invalid required fields: {skipped:,}")
    if result.empty:
        st.warning("No rows had all required numeric fields, so there was nothing to score.")
        footer()
        return
    probabilities = result["fraud_probability"].astype(float)
    review_count = int((probabilities >= review).sum())
    high_count = int((probabilities >= high).sum())
    a, b, c, d = st.columns(4)
    a.metric("Rows scored", f"{len(result):,}")
    b.metric("At review threshold", f"{review_count:,} ({review_count / len(result):.2%})")
    c.metric("At high threshold", f"{high_count:,} ({high_count / len(result):.2%})")
    d.metric("Mean probability", f"{probabilities.mean():.2%}")

    st.subheader("Probability distribution")
    chart_data = pd.DataFrame({"Fraud probability": probabilities})
    histogram = alt.Chart(chart_data).mark_bar(color=ACCENT).encode(
        x=alt.X("Fraud probability:Q", bin=alt.Bin(maxbins=40), title="Fraud probability", axis=alt.Axis(format=".0%")),
        y=alt.Y("count():Q", title="Rows (log scale)", scale=alt.Scale(type="log")),
        tooltip=[alt.Tooltip("count():Q", title="Rows")],
    )
    thresholds = alt.Chart(pd.DataFrame({"threshold": [review, high], "name": ["Review", "High risk"]})).mark_rule(
        color=NEUTRAL, strokeDash=[4, 3]
    ).encode(
        x=alt.X("threshold:Q", axis=alt.Axis(format=".0%")),
        tooltip=[alt.Tooltip("name:N"), alt.Tooltip("threshold:Q", format=".2%")],
    )
    st.altair_chart((histogram + thresholds).properties(height=300), use_container_width=True)

    st.subheader("Highest-risk rows")
    display_columns = [column for column in ("TransactionID", "TransactionAmt", "card1", "fraud_probability", "risk_band") if column in result.columns]
    top = result.sort_values("fraud_probability", ascending=False).head(100)[display_columns]
    config = {
        "TransactionAmt": st.column_config.NumberColumn("Transaction amount", format="%.3f"),
        "card1": st.column_config.NumberColumn("card1", format="%d"),
        "fraud_probability": st.column_config.ProgressColumn("Fraud probability", format="percent", min_value=0.0, max_value=1.0),
        "risk_band": st.column_config.TextColumn("Risk band"),
    }
    if "TransactionID" in display_columns:
        config["TransactionID"] = st.column_config.NumberColumn("TransactionID", format="%d")
    st.dataframe(top, column_config=config, hide_index=True, use_container_width=True, height=420)
    download_columns = (["TransactionID"] if "TransactionID" in result.columns else []) + ["fraud_probability", "risk_band"]
    st.download_button(
        "Download scored results",
        data=result[download_columns].to_csv(index=False).encode("utf-8"),
        file_name="fraud_scores.csv",
        mime="text/csv",
    )
    if "isFraud" in limited.columns:
        _render_evaluation(limited, result, high)
    footer()


main()
