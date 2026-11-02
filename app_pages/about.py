"""Purpose, dataset, modeling method, and limitations page."""

from __future__ import annotations

import streamlit as st

from app_pages.common import footer, page_context
from src.model_card import FRAUD_RATE, LIMITATIONS, TRANSACTION_COUNT


def main() -> None:
    """Render the short model card and its practical limitations."""
    st.title("About and limitations")
    st.caption("A concise model card for interpreting scores from this application.")
    if page_context() is None:
        footer()
        return
    st.subheader("Purpose")
    st.write("Estimate transaction fraud risk to support analyst review.")
    st.subheader("Data")
    st.write(
        f"IEEE-CIS fraud detection data from Vesta: {TRANSACTION_COUNT:,} transactions, "
        f"with a {FRAUD_RATE:.1%} fraud rate."
    )
    st.subheader("Method")
    st.write("A time-based train/validation split with a LightGBM classifier and pseudo user-ID history features.")
    st.subheader("Limitations")
    for limitation in LIMITATIONS:
        st.markdown(f"- {limitation}")
    footer()


main()
