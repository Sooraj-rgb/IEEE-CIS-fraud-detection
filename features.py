"""Feature engineering and cleaning for the IEEE-CIS fraud project.

Two functions do the work:

    state = fit_preprocessor(train_df, extra_dfs=[valid_df])
    clean = apply_preprocessor(any_df, state)

fit_preprocessor LEARNS facts (group averages, which categories are common,
which columns to drop) and stores them in a dictionary called `state`.
apply_preprocessor only READS that dictionary, so train, validation and the
Kaggle test files all get exactly the same treatment.

Input frames must NOT contain `isFraud` or `TransactionID`.
"""

import numpy as np
import pandas as pd

COUNT_COLS = ["card1", "card2", "addr1", "P_emaildomain"]
RARE_THRESHOLD = 100      # categories with fewer rows than this become "other"
NULL_DROP_THRESHOLD = 0.99  # columns more empty than this are dropped


def _make_uid(df):
    """Fake 'person ID': same card + same address + same account start day."""
    day = df["TransactionDT"] // 86400
    # .map(str) turns a missing value into the text "nan" on every pandas
    # version, so rows with a blank address or D1 still get a person ID
    return (df["card1"].map(str) + "_" + df["addr1"].map(str) + "_"
            + (day - df["D1"]).round().map(str))


def _add_numeric_features(df, state):
    """Group features + time/amount features. Uses only the numbers in `state`."""
    df = df.copy()

    # person-ID features
    uid = _make_uid(df)
    df["uid_count"] = uid.map(state["uid_stats"]["count"])
    df["uid_amt_mean"] = uid.map(state["uid_stats"]["mean"])
    df["uid_amt_std"] = uid.map(state["uid_stats"]["std"])
    df["amt_vs_uid_mean"] = df["TransactionAmt"] / df["uid_amt_mean"]

    # "how common is this card / address / email?"
    for col in COUNT_COLS:
        df[col + "_count"] = df[col].map(state["count_maps"][col])

    # amount behaviour per card
    df["card1_amt_mean"] = df["card1"].map(state["card1_stats"]["mean"])
    df["card1_amt_std"] = df["card1"].map(state["card1_stats"]["std"])

    # time and amount features
    df["amt_log"] = np.log1p(df["TransactionAmt"])
    df["hour"] = (df["TransactionDT"] // 3600) % 24
    df["weekday"] = (df["TransactionDT"] // 86400) % 7
    return df.drop(columns=["TransactionDT"])


def _clean_categoricals(df, state):
    """Blank -> 'missing', rare -> 'other', then mark as categorical."""
    df = df.copy()
    for col, cats in state["cats"].items():
        s = df[col].astype("object").fillna("missing")
        s = s.where(s.isin(cats), "other")
        df[col] = pd.Categorical(s, categories=cats)
    return df


def fit_preprocessor(train, extra_dfs=()):
    """Learn everything the cleaning needs.

    The group statistics (person counts, card averages, ...) are computed on
    `train` plus any frames in `extra_dfs`. The notebook computed them on
    train + validation together, so train.py passes the validation set here
    to reproduce the same numbers. That uses a bit of "future" information,
    which a live system would not have. See the README limitations.
    """
    all_df = pd.concat([train, *extra_dfs])
    state = {}

    uid = _make_uid(all_df)
    state["uid_stats"] = (all_df["TransactionAmt"].groupby(uid)
                          .agg(["count", "mean", "std"]))
    state["count_maps"] = {c: all_df[c].value_counts() for c in COUNT_COLS}
    state["card1_stats"] = (all_df.groupby("card1")["TransactionAmt"]
                            .agg(["mean", "std"]))

    # learn which text values are common (from train only)
    num = _add_numeric_features(train, state)
    state["cats"] = {}
    for col in num.select_dtypes(include=["object", "string"]).columns:
        counts = num[col].value_counts()
        keep = set(counts[counts >= RARE_THRESHOLD].index)
        state["cats"][col] = sorted(keep | {"missing", "other"})

    # learn which numeric columns are almost completely empty (train only)
    cleaned = _clean_categoricals(num, state)
    null_pct = cleaned.isnull().mean()
    state["drop_cols"] = list(null_pct[null_pct > NULL_DROP_THRESHOLD].index)
    return state


def apply_preprocessor(df, state):
    """Apply the saved cleaning steps to any dataframe."""
    df = _add_numeric_features(df, state)
    df = _clean_categoricals(df, state)
    return df.drop(columns=state["drop_cols"], errors="ignore")
