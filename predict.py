"""Final fit on all labeled data + predictions for the Kaggle test files.

Run from the project folder AFTER `python src/train.py` has produced
model.joblib and preprocessor.joblib:

    python src/predict.py

What it does
  1. Loads train_*.csv and test_*.csv from data/ (left-joining identity).
  2. Learns the cleaning steps with the group statistics computed over
     train + test together (the approach top Kaggle solutions used), so test
     transactions get real person/card counts instead of blanks.
  3. Trains one final model on ALL labeled rows. There is no validation set
     left to stop early on, so the tree count is the one found by train.py
     (best_iteration_) scaled up by 1.25 because there is ~25% more data.
  4. Writes submission.csv (TransactionID, isFraud probability).

This is a leaderboard-style run, not a live-system design: it uses test rows
to compute features. Keep that in the README limitations. Your honest
validation number is still the one from train.py.
"""

import argparse
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from features import apply_preprocessor, fit_preprocessor  # noqa: E402
from train import PARAMS  # noqa: E402

DATA_DIR = Path("data")
TREE_SCALE = 1.25


def load_merged(kind):
    """Read {kind}_transaction.csv and {kind}_identity.csv and left-join them."""
    trans = pd.read_csv(DATA_DIR / f"{kind}_transaction.csv")
    ident = pd.read_csv(DATA_DIR / f"{kind}_identity.csv")
    # Kaggle's test_identity.csv names columns id-01, id-02 ... (hyphens),
    # while train_identity.csv uses id_01, id_02 ... (underscores).
    ident.columns = [c.replace("-", "_") for c in ident.columns]
    return trans.merge(ident, on="TransactionID", how="left")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trees", type=int, default=None,
                        help="number of trees; default = train.py's best_iteration_ x 1.25")
    parser.add_argument("--out", default="submission.csv")
    args = parser.parse_args()

    if not Path("preprocessor.joblib").exists():
        raise SystemExit("preprocessor.joblib not found. Run python src/train.py first.")
    old_state = joblib.load("preprocessor.joblib")

    if args.trees is None:
        if not Path("model.joblib").exists():
            raise SystemExit("model.joblib not found. Run python src/train.py first, "
                             "or pass --trees N.")
        n_trees = int(joblib.load("model.joblib").best_iteration_ * TREE_SCALE)
    else:
        n_trees = args.trees
    print("trees for the final model:", n_trees)

    # 1. data
    train = load_merged("train").sort_values("TransactionDT").reset_index(drop=True)
    test = load_merged("test")
    y = train["isFraud"]
    x_train = train.drop(columns=["isFraud", "TransactionID"])
    ids = test["TransactionID"]
    x_test = test.drop(columns=["TransactionID"])

    missing = set(x_train.columns) - set(x_test.columns)
    if missing:
        raise SystemExit(f"test files are missing {len(missing)} columns, "
                         f"for example {sorted(missing)[:5]}")
    x_test = x_test[x_train.columns]
    print("train rows:", len(x_train), "| test rows:", len(x_test))

    # 2. cleaning steps, learned with test included for the group statistics
    state = fit_preprocessor(x_train, extra_dfs=[x_test])
    # keep the columns train.py found useless (zero importance)
    state["drop_cols"] = sorted(set(state["drop_cols"]) | set(old_state["drop_cols"]))
    xt = apply_preprocessor(x_train, state)
    xs = apply_preprocessor(x_test, state)
    print("features:", xt.shape[1])

    # 3. final model on all labeled data
    model = lgb.LGBMClassifier(**{**PARAMS, "n_estimators": n_trees})
    model.fit(xt, y)

    # 4. predictions
    p = model.predict_proba(xs)[:, 1]
    sub = pd.DataFrame({"TransactionID": ids, "isFraud": p})
    sub.to_csv(args.out, index=False)
    print(f"saved {args.out}: {len(sub)} rows | mean probability {p.mean():.4f} "
          f"| share above 0.5: {(p > 0.5).mean():.4f}")

    joblib.dump(model, "final_model.joblib")
    joblib.dump(state, "final_preprocessor.joblib")


if __name__ == "__main__":
    main()
