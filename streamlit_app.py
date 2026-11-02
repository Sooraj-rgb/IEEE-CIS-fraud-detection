"""Run the Streamlit fraud scoring application."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import joblib
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ARTIFACT_FIX = "Run python src/train.py first"


def _artifact_path(variable: str, default_name: str) -> Path:
    """Resolve an artifact override or its project-root default."""
    configured = os.environ.get(variable)
    path = Path(configured) if configured else ROOT / default_name
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


@st.cache_resource(show_spinner=False)
def load_artifacts(model_path: str, state_path: str) -> tuple[Any, dict[str, Any]]:
    """Load and validate the local model and preprocessor once per path."""
    model = joblib.load(Path(model_path))
    state = joblib.load(Path(state_path))
    if not hasattr(model, "feature_name_") or not hasattr(model, "booster_"):
        raise ValueError("The model artifact does not contain the expected LightGBM model.")
    if not isinstance(state, dict) or not {"cats", "drop_cols"}.issubset(state):
        raise ValueError("The preprocessor artifact does not contain the expected state.")
    return model, state


def _repair_thresholds(changed: str) -> None:
    """Keep the review threshold strictly below the high-risk threshold."""
    review = float(st.session_state["review_threshold"])
    high = float(st.session_state["high_threshold"])
    if review < high:
        return
    if changed == "review_threshold":
        if review >= 0.99:
            st.session_state["review_threshold"] = 0.98
            st.session_state["high_threshold"] = 0.99
        else:
            st.session_state["high_threshold"] = min(0.99, review + 0.01)
    elif high <= 0.01:
        st.session_state["review_threshold"] = 0.01
        st.session_state["high_threshold"] = 0.02
    else:
        st.session_state["review_threshold"] = max(0.01, high - 0.01)


def _apply_cost_threshold() -> None:
    """Apply the cost-based suggestion as the review threshold."""
    missed = float(st.session_state["cost_missed_fraud"])
    false_alarm = float(st.session_state["cost_false_alarm"])
    total = missed + false_alarm
    if total <= 0:
        return
    suggested = false_alarm / total
    high = float(st.session_state["high_threshold"])
    st.session_state["review_threshold"] = min(0.99, max(0.01, min(suggested, high - 0.01)))


def _render_sidebar(model_loaded: bool, load_error: str | None) -> None:
    """Render shared artifact status and decision settings."""
    with st.sidebar:
        st.header("Scoring settings")
        st.write("Artifacts: Loaded" if model_loaded else "Artifacts: Error")
        if not model_loaded:
            st.caption(ARTIFACT_FIX)
        st.slider(
            "Review threshold",
            min_value=0.01,
            max_value=0.99,
            step=0.01,
            key="review_threshold",
            on_change=_repair_thresholds,
            args=("review_threshold",),
            help="Scores at or above this value enter the review band.",
        )
        st.slider(
            "High-risk threshold",
            min_value=0.01,
            max_value=0.99,
            step=0.01,
            key="high_threshold",
            on_change=_repair_thresholds,
            args=("high_threshold",),
            help="Scores at or above this value enter the high-risk band.",
        )
        with st.expander("Threshold from costs"):
            st.number_input(
                "Cost of a missed fraud",
                min_value=0.0,
                value=100.0,
                step=10.0,
                key="cost_missed_fraud",
            )
            st.number_input(
                "Cost of a false alarm",
                min_value=0.0,
                value=10.0,
                step=1.0,
                key="cost_false_alarm",
            )
            missed = float(st.session_state["cost_missed_fraud"])
            false_alarm = float(st.session_state["cost_false_alarm"])
            total = missed + false_alarm
            if total > 0:
                suggested = false_alarm / total
                st.caption(f"Suggested threshold: {suggested:.2%}.")
            else:
                st.caption("Enter a non-zero cost to calculate a suggested threshold.")
            st.caption("This suggestion assumes calibrated probabilities.")
            st.button(
                "Use suggested threshold",
                disabled=total <= 0,
                on_click=_apply_cost_threshold,
                use_container_width=True,
            )


def main() -> None:
    """Configure the app, load artifacts, and run the selected page."""
    st.set_page_config(page_title="Fraud scoring", layout="wide")
    st.markdown(
        """<style>
        section.main > div.block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
        #MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDeployButton"] {visibility: hidden;}
        [data-testid="stMetricValue"] {font-variant-numeric: tabular-nums;}
        [data-testid="stVerticalBlockBorderWrapper"] {border: 1px solid #D9DEE5; border-radius: 6px; box-shadow: none;}
        h1, h2, h3 {letter-spacing: -0.015em;}
        </style>""",
        unsafe_allow_html=True,
    )
    st.session_state.setdefault("review_threshold", 0.10)
    st.session_state.setdefault("high_threshold", 0.50)
    model = None
    state = None
    load_error = None
    model_path = _artifact_path("MODEL_PATH", "model.joblib")
    state_path = _artifact_path("STATE_PATH", "preprocessor.joblib")
    try:
        model, state = load_artifacts(str(model_path), str(state_path))
    except Exception:
        load_error = ARTIFACT_FIX
    st.session_state["_fraud_model"] = model
    st.session_state["_fraud_state"] = state
    st.session_state["_artifact_error"] = load_error
    _render_sidebar(model is not None and state is not None, load_error)

    pages = [
        st.Page("app_pages/score.py", title="Score a transaction", default=True),
        st.Page("app_pages/batch.py", title="Batch scoring"),
        st.Page("app_pages/insights.py", title="Model insights"),
        st.Page("app_pages/about.py", title="About and limitations"),
    ]
    st.navigation(pages, position="sidebar").run()


if __name__ == "__main__":
    main()
