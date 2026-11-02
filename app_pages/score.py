"""Interactive single-transaction scoring page."""

from __future__ import annotations

import json
from typing import Any

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from app_pages.common import footer, page_context
from src.scoring import (
    REQUIRED,
    group_features_available,
    model_frame,
    risk_band,
    score_frame,
    transformed_frame,
    validate_required,
)

ACCENT = "#1F4E79"
NEUTRAL = "#9099A5"
RISK_COLORS = {"Low": "#2F6B4F", "Review": "#A66A00", "High": "#A12B2B"}
NUMERIC_FIELDS = ("card1", "card2", "card3", "card5", "addr1", "addr2")
TEXT_FIELDS = (
    "ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain",
    "DeviceType", "DeviceInfo", "id_31",
)
PRESETS = [
    {
        "label": "Sample from training data - actual label: legitimate",
        "values": {
            "TransactionDT": 8895414, "TransactionAmt": 209.95, "ProductCD": "W",
            "card1": 14265, "card2": 543, "card3": 150, "card4": "mastercard", "card5": 224,
        },
    },
    {
        "label": "Sample from training data - actual label: legitimate",
        "values": {
            "TransactionDT": 8703573, "TransactionAmt": 6.767, "ProductCD": "C",
            "card1": 3154, "card2": 408, "card3": 185, "card4": "mastercard", "card5": 224,
            "DeviceType": "mobile", "id_31": "mobile safari generic",
        },
    },
    {
        "label": "Sample from training data - actual label: fraud",
        "values": {
            "TransactionDT": 6181776, "TransactionAmt": 59.0, "ProductCD": "W",
            "card1": 16034, "card2": 481, "card3": 150, "card4": "mastercard", "card5": 102,
        },
    },
]


def _categorical_options(column: str, state: dict[str, Any]) -> list[str]:
    """Return allowed categories without the catch-all value."""
    options = [value for value in state.get("cats", {}).get(column, ["missing"]) if value != "other"]
    return options or ["missing"]


def _set_preset(index: int) -> None:
    """Load one provided sample into the form's session-state fields."""
    sample = PRESETS[index]["values"]
    dt = int(sample["TransactionDT"])
    st.session_state["score_amount"] = str(sample["TransactionAmt"])
    st.session_state["score_day"] = dt // 86400
    st.session_state["score_hour"] = (dt // 3600) % 24
    for field in NUMERIC_FIELDS:
        value = sample.get(field)
        st.session_state[f"score_{field}"] = "" if value is None else str(value)
    for field in TEXT_FIELDS:
        st.session_state[f"score_{field}"] = sample.get(field, "missing")
    st.session_state["score_extra_json"] = "{}"
    st.session_state["_score_result"] = None
    st.session_state["_score_values"] = None


def _parse_number(value: str, field: str) -> float:
    """Parse a form number while preserving a blank as NaN."""
    if not value.strip():
        return float("nan")
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be a number or left blank.") from exc


def _parse_extra_fields(value: str, controlled: set[str]) -> dict[str, Any]:
    """Validate the advanced JSON object and return additional raw fields."""
    try:
        data = json.loads(value or "{}", parse_constant=lambda item: (_ for _ in ()).throw(ValueError(item)))
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError("Advanced fields must be valid JSON (for example, {\"M1\": \"T\"}).") from exc
    if not isinstance(data, dict):
        raise ValueError("Advanced fields must be a JSON object with column names and values.")
    if any(not isinstance(key, str) for key in data):
        raise ValueError("Advanced field names must be text column names.")
    if set(data) & controlled:
        raise ValueError("Advanced fields can only add columns; remove fields already entered above.")
    if any(isinstance(item, (dict, list)) for item in data.values()):
        raise ValueError("Advanced field values must be simple text, numbers, true, false, or null values.")
    return data


def _render_form(state: dict[str, Any]) -> tuple[bool, dict[str, Any] | None, str | None]:
    """Render the grouped transaction form and return its submitted raw row."""
    for index, preset in enumerate(PRESETS):
        st.button(preset["label"], key=f"score_sample_{index}", on_click=_set_preset, args=(index,))
    submitted = False
    raw: dict[str, Any] | None = None
    error: str | None = None
    with st.form("single_transaction_form"):
        with st.expander("Transaction", expanded=True):
            amount = st.text_input("Transaction amount", key="score_amount", placeholder="Required")
            day_col, hour_col = st.columns(2)
            day = day_col.slider("Day index", 0, 181, key="score_day", value=0)
            hour = hour_col.slider("Hour", 0, 23, key="score_hour", value=0)
            product = st.selectbox(
                "ProductCD", _categorical_options("ProductCD", state), key="score_ProductCD",
                format_func=lambda value: "Not provided" if value == "missing" else value,
            )
        with st.expander("Card", expanded=True):
            cols = st.columns(3)
            values: dict[str, Any] = {"ProductCD": product}
            for field, column in zip(("card1", "card2", "card3", "card4", "card5", "card6"), cols * 2):
                if field in {"card1", "card2", "card3", "card5"}:
                    values[field] = column.text_input(
                        field, key=f"score_{field}", placeholder="Required" if field == "card1" else "Optional"
                    )
                else:
                    values[field] = column.selectbox(
                        field, _categorical_options(field, state), key=f"score_{field}",
                        format_func=lambda value: "Not provided" if value == "missing" else value,
                    )
        with st.expander("Billing and email"):
            left, right = st.columns(2)
            values["addr1"] = left.text_input("addr1", key="score_addr1")
            values["addr2"] = right.text_input("addr2", key="score_addr2")
            values["P_emaildomain"] = left.selectbox(
                "P_emaildomain", _categorical_options("P_emaildomain", state), key="score_P_emaildomain",
                format_func=lambda value: "Not provided" if value == "missing" else value,
            )
            values["R_emaildomain"] = right.selectbox(
                "R_emaildomain", _categorical_options("R_emaildomain", state), key="score_R_emaildomain",
                format_func=lambda value: "Not provided" if value == "missing" else value,
            )
        with st.expander("Device"):
            cols = st.columns(3)
            for field, column in zip(("DeviceType", "DeviceInfo", "id_31"), cols):
                values[field] = column.selectbox(
                    field, _categorical_options(field, state), key=f"score_{field}",
                    format_func=lambda value: "Not provided" if value == "missing" else value,
                )
        with st.expander("Advanced: other raw fields"):
            extra_json = st.text_area("Additional raw columns as JSON", value="{}", key="score_extra_json")
        submitted = st.form_submit_button("Score transaction", type="primary", use_container_width=True)
    if submitted:
        try:
            row: dict[str, Any] = {
                "TransactionDT": int(day) * 86400 + int(hour) * 3600,
                "TransactionAmt": _parse_number(amount, "Transaction amount"),
                "ProductCD": product,
            }
            for field in NUMERIC_FIELDS:
                source = values[field]
                row[field] = _parse_number(source, field)
            for field in TEXT_FIELDS:
                row[field] = values[field]
            if not row["card1"] or not np.isfinite(row["card1"]):
                raise ValueError("card1 is required and must be a whole number.")
            if row["card1"] % 1 != 0:
                raise ValueError("card1 must be a whole number.")
            if not np.isfinite(row["TransactionAmt"]):
                raise ValueError("Transaction amount is required and must be a number.")
            controlled = set(row) | {"TransactionID", "isFraud"}
            row.update(_parse_extra_fields(extra_json, controlled))
            submitted_frame = pd.DataFrame([row])
            if not bool(validate_required(submitted_frame).iloc[0]):
                raise ValueError("Transaction amount and card1 are required.")
            raw = row
        except ValueError as exc:
            error = str(exc)
    return submitted, raw, error


def _contribution_chart(result: pd.DataFrame) -> alt.Chart:
    """Build a compact signed contribution chart for the ten strongest effects."""
    columns = [column for column in result.columns if column.startswith("contrib:")]
    contributions = result.loc[result.index[0], columns]
    top = contributions.reindex(contributions.abs().sort_values(ascending=False).head(10).index)
    frame = pd.DataFrame({
        "feature": [column.removeprefix("contrib:") for column in top.index],
        "contribution": top.to_numpy(dtype=float),
    }).sort_values("contribution", ascending=True)
    return (
        alt.Chart(frame)
        .mark_bar(size=17)
        .encode(
            x=alt.X("contribution:Q", title="Log-odds contribution", axis=alt.Axis(format=".2f")),
            y=alt.Y("feature:N", sort=None, title=None),
            color=alt.condition("datum.contribution > 0", alt.value(ACCENT), alt.value(NEUTRAL)),
            tooltip=[alt.Tooltip("feature:N", title="Feature"), alt.Tooltip("contribution:Q", format=".4f")],
        )
        .properties(height=290)
    )


def _render_result(model: Any, state: dict[str, Any], row: dict[str, Any], review: float, high: float) -> None:
    """Score one valid row and render its probability, risk, and explanation."""
    frame = pd.DataFrame([row])
    try:
        result = score_frame(frame, model, state, contrib=True)
        transformed = transformed_frame(frame, model, state)
        ready = model_frame(frame, model, state)
    except Exception:
        st.error("The transaction could not be prepared for scoring. Check the raw field values and try again.")
        return
    probability = float(result["fraud_probability"].iloc[0])
    band = risk_band(probability, review, high)
    st.metric("Fraud probability", f"{probability:.2%}")
    st.markdown(
        f"<div style='font-size:1.15rem;font-weight:650;color:{RISK_COLORS[band]}'>Risk band: {band}</div>",
        unsafe_allow_html=True,
    )
    bar_data = pd.DataFrame({"label": ["Fraud probability"], "probability": [probability]})
    threshold_data = pd.DataFrame({"threshold": [review, high], "name": ["Review", "High risk"]})
    bar = alt.Chart(bar_data).mark_bar(size=22, color=ACCENT).encode(
        x=alt.X("probability:Q", scale=alt.Scale(domain=[0, 1]), title="Probability", axis=alt.Axis(format=".0%")),
        y=alt.Y("label:N", title=None, axis=None),
        tooltip=[alt.Tooltip("probability:Q", format=".2%")],
    )
    rules = alt.Chart(threshold_data).mark_rule(color=NEUTRAL, strokeDash=[4, 3]).encode(
        x=alt.X("threshold:Q", scale=alt.Scale(domain=[0, 1])),
        tooltip=[alt.Tooltip("name:N"), alt.Tooltip("threshold:Q", format=".0%")],
    )
    st.altair_chart((bar + rules).properties(height=75), use_container_width=True)
    guidance = {
        "Low": "Continue standard controls; this score does not call for elevated review.",
        "Review": "Review this transaction alongside supporting context.",
        "High": "Prioritize analyst review before taking action.",
    }
    st.write(guidance[band])
    if not group_features_available(transformed.iloc[0]):
        st.info(
            "This card or customer was not seen in the training period, so history-based features are blank; "
            "the score relies on the remaining fields."
        )
    st.subheader("Why this score")
    st.caption("Positive contributions move the score toward fraud; negative contributions move it toward legitimate.")
    st.altair_chart(_contribution_chart(result), use_container_width=True)
    with st.expander("Model-ready feature values"):
        feature_values = pd.DataFrame({"Feature": ready.columns, "Model-ready value": ready.iloc[0].to_numpy()})
        st.dataframe(
            feature_values,
            column_config={
                "Feature": st.column_config.TextColumn("Feature"),
                "Model-ready value": st.column_config.TextColumn("Model-ready value"),
            },
            hide_index=True,
            use_container_width=True,
            height=360,
        )


def main() -> None:
    """Render the single-transaction scoring experience."""
    st.title("Score a transaction")
    st.caption("Estimate fraud risk for one transaction using the saved LightGBM model.")
    context = page_context()
    if context is None:
        footer()
        return
    model, state, review, high = context
    left, right = st.columns([1, 1.05], gap="large")
    with left:
        submitted, row, error = _render_form(state)
        if error:
            st.error(error)
        elif submitted and row is not None:
            st.session_state["_score_values"] = row
            st.session_state["_score_result"] = True
    with right:
        st.subheader("Result")
        saved = st.session_state.get("_score_values")
        if saved is None:
            st.info("Enter the required transaction amount and card1, then select Score transaction.")
        else:
            _render_result(model, state, saved, review, high)
    footer()


main()
