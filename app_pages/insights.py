"""Validation metrics and feature importance page."""

from __future__ import annotations

import re

import altair as alt
import pandas as pd
import streamlit as st

from app_pages.common import footer, page_context
from src.model_card import EXPERIMENTS, VALIDATION_PR_AUC, VALIDATION_ROC_AUC

ACCENT = "#1F4E79"
NEUTRAL = "#9099A5"


def _is_engineered(name: str) -> bool:
    """Identify the engineered feature families highlighted in the chart."""
    return (
        name.startswith("uid_")
        or name.endswith("_count")
        or name.startswith("card1_amt_")
        or name in {"amt_log", "hour", "weekday", "amt_vs_uid_mean"}
    )


def main() -> None:
    """Render validation facts, top gain features, and experiment history."""
    st.title("Model insights")
    st.caption("Validation performance and the feature patterns used by the trained model.")
    context = page_context()
    if context is None:
        footer()
        return
    model, _state, _review, _high = context
    a, b, c, d = st.columns(4)
    a.metric("Validation ROC AUC", f"{VALIDATION_ROC_AUC:.4f}")
    b.metric("Validation PR AUC", f"{VALIDATION_PR_AUC:.4f}")
    c.metric("Trees", f"{int(model.best_iteration_):,}")
    d.metric("Features", f"{len(model.feature_name_):,}")

    importance = model.booster_.feature_importance(importance_type="gain")
    feature_data = pd.DataFrame({"Feature": model.feature_name_, "Gain": importance})
    feature_data = feature_data.sort_values("Gain", ascending=False).head(20).sort_values("Gain", ascending=True)
    feature_data["Family"] = feature_data["Feature"].map(lambda name: "Engineered" if _is_engineered(name) else "Other")
    chart = alt.Chart(feature_data).mark_bar(size=17).encode(
        x=alt.X("Gain:Q", title="Split gain", axis=alt.Axis(format="~s")),
        y=alt.Y("Feature:N", sort=None, title=None),
        color=alt.Color(
            "Family:N",
            scale=alt.Scale(domain=["Engineered", "Other"], range=[ACCENT, NEUTRAL]),
            legend=None,
        ),
        tooltip=["Feature:N", alt.Tooltip("Gain:Q", format=".3f"), "Family:N"],
    ).properties(height=500)
    st.subheader("Top 20 features by gain")
    st.caption("Accent marks engineered features; neutral grey marks other model features.")
    st.altair_chart(chart, use_container_width=True)

    st.subheader("Experiment history")
    experiments = pd.DataFrame(EXPERIMENTS)
    st.dataframe(
        experiments,
        column_config={
            "Experiment": st.column_config.TextColumn("Experiment"),
            "ROC AUC": st.column_config.NumberColumn("ROC AUC", format="%.4f"),
            "PR AUC": st.column_config.NumberColumn("PR AUC", format="%.4f"),
        },
        hide_index=True,
        use_container_width=True,
    )
    st.caption("Class weighting with scale_pos_weight 27.6 reduced PR AUC to 0.22 and was rejected.")
    footer()


main()
