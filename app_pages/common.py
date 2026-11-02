"""Shared display helpers for the Streamlit pages."""

from __future__ import annotations

from typing import Any

import streamlit as st


def page_context() -> tuple[Any, dict[str, Any], float, float] | None:
    """Return loaded artifacts and thresholds, or display the shared error panel."""
    model = st.session_state.get("_fraud_model")
    state = st.session_state.get("_fraud_state")
    if model is None or state is None:
        st.error("Artifacts could not be loaded. Run python src/train.py first")
        return None
    return (
        model,
        state,
        float(st.session_state.get("review_threshold", 0.10)),
        float(st.session_state.get("high_threshold", 0.50)),
    )


def footer() -> None:
    """Show the shared decision-support notice."""
    st.markdown(
        "<div style='color:#5B6675;font-size:0.82rem;border-top:1px solid #D9DEE5;"
        "padding-top:0.65rem;margin-top:2rem'>Decision-support tool. Scores are model estimates "
        "and should be reviewed by an analyst.</div>",
        unsafe_allow_html=True,
    )
