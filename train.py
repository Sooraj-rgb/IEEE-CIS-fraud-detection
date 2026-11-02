"""Train the fraud model end to end.

Run from the project folder:  python src/train.py
Expected result on train_merged.csv: ROC AUC 0.9334 | PR AUC 0.6195
"""

import csv
import datetime
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent))
from features import apply_preprocessor, fit_preprocessor  # noqa: E402

DATA_PATH = "data/train_merged.csv"
PARAMS = dict(
    n_estimators=6000, learning_rate=0.02, num_leaves=128,
    min_child_samples=20, colsample_bytree=0.5, subsample=0.8,
    subsample_freq=1, scale_pos_weight=1, random_state=42, verbose=-1,
)


def train_model(x_train, y_train, x_valid, y_valid):
    model = lgb.LGBMClassifier(**PARAMS)
    model.fit(
        x_train, y_train,
        eval_set=[(x_valid, y_valid)], eval_metric="auc",
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    p = model.predict_proba(x_valid)[:, 1]
    return model, roc_auc_score(y_valid, p), average_precision_score(y_valid, p)


def main():
    # 1. load, sort by time, split 80/20 without shuffling
    df = pd.read_csv(DATA_PATH)
    df = df.sort_values("TransactionDT").reset_index(drop=True)
    x = df.drop(columns=["isFraud", "TransactionID"])
    y = df["isFraud"]
    x_train, x_valid, y_train, y_valid = train_test_split(
        x, y, test_size=0.2, random_state=42, shuffle=False)

    # 2. learn the cleaning steps, then apply them
    state = fit_preprocessor(x_train, extra_dfs=[x_valid])
    xt = apply_preprocessor(x_train, state)
    xv = apply_preprocessor(x_valid, state)
    print("features after cleaning:", xt.shape[1])

    # 3. first model, only to find columns the model never uses
    model, roc, pr = train_model(xt, y_train, xv, y_valid)
    print(f"first model  -> ROC AUC {roc:.4f} | PR AUC {pr:.4f} "
          f"| trees {model.best_iteration_}")
    gain = pd.Series(model.booster_.feature_importance(importance_type="gain"),
                     index=xt.columns)
    useless = list(gain[gain == 0].index)
    print("zero-importance columns dropped:", len(useless))

    # 4. remember them in the state, drop, retrain: this is the final model
    state["drop_cols"] += useless
    xt = xt.drop(columns=useless)
    xv = xv.drop(columns=useless)
    model, roc, pr = train_model(xt, y_train, xv, y_valid)
    print(f"final model  -> ROC AUC {roc:.4f} | PR AUC {pr:.4f} "
          f"| trees {model.best_iteration_}")
    print("notebook result was ROC AUC 0.9334 | PR AUC 0.6195")

    # 5. save everything a prediction service needs
    joblib.dump(model, "model.joblib")
    joblib.dump(state, "preprocessor.joblib")

if __name__ == "__main__":
    main()
