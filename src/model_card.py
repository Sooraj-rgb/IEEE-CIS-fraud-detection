"""Static validation facts and experiment history for the trained model."""

VALIDATION_ROC_AUC = 0.9334
VALIDATION_PR_AUC = 0.6195
FRAUD_RATE = 0.035
TRANSACTION_COUNT = 590_540

EXPERIMENTS = [
    {"Experiment": "Baseline", "ROC AUC": 0.9178, "PR AUC": 0.5790},
    {"Experiment": "Drop 7 empty columns", "ROC AUC": 0.9187, "PR AUC": 0.5764},
    {"Experiment": "+ pseudo user-ID features", "ROC AUC": 0.9245, "PR AUC": 0.5987},
    {"Experiment": "+ frequency and per-card amount features", "ROC AUC": 0.9263, "PR AUC": 0.6105},
    {"Experiment": "num_leaves 128", "ROC AUC": 0.9300, "PR AUC": 0.6126},
    {"Experiment": "learning rate 0.02", "ROC AUC": 0.9340, "PR AUC": 0.6159},
    {"Experiment": "Drop 15 zero-importance columns", "ROC AUC": 0.9334, "PR AUC": 0.6195},
]

LIMITATIONS = [
    "Group features were computed over train and validation together, which is not possible live.",
    "Early stopping used the validation set, so the score is slightly optimistic.",
    "card1 is a top feature, so the model may recognise specific cards.",
    "The pseudo user ID is a heuristic.",
    "Unseen cards receive blank history features.",
    "Feature contributions explain the model, not causes of fraud.",
    "This app is a decision aid, not an automated decision.",
]
