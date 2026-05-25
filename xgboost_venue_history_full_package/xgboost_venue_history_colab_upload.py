
# ============================================================
# XGBoost Cricket Match Winner Prediction
# Single Feature Engineering: Pre-match Venue History Features
# ============================================================

# -----------------------------
# Cell 1: Import libraries and setup
# -----------------------------
import os
import re
import json
import warnings
from pathlib import Path
from collections import defaultdict

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    roc_curve,
)
import joblib

RANDOM_STATE = 42
OUTPUT_DIR = Path("xgb_venue_history_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Cell 2: Upload/read dataset
# -----------------------------
def get_dataset_path():
    """Colab-friendly upload. Locally, falls back to Dataset-1.csv."""
    try:
        from google.colab import files
        print("Upload your dataset CSV file now, for example: Dataset-1.csv")
        uploaded = files.upload()
        if len(uploaded) == 0:
            raise ValueError("No file was uploaded.")
        uploaded_filename = list(uploaded.keys())[0]
        print(f"Uploaded file detected: {uploaded_filename}")
        return uploaded_filename
    except ModuleNotFoundError:
        local_path = "Dataset-1.csv"
        if not os.path.exists(local_path):
            local_path = "/mnt/data/Dataset-1.csv"
        print(f"Not running inside Colab. Using local file: {local_path}")
        return local_path

DATA_PATH = get_dataset_path()


# -----------------------------
# Cell 3: Cleaning helpers
# -----------------------------
def clean_team_name(x):
    if pd.isna(x):
        return np.nan
    x = str(x).strip()
    x = re.sub(r"\s+", " ", x)
    replacements = {
        "West": "West Indies",
        "U.S.A.": "USA",
        "United States of America": "USA",
        "United Arab Emirates": "UAE",
    }
    return replacements.get(x, x)


def clean_venue_name(x):
    if pd.isna(x):
        return "Unknown"
    x = str(x).strip()
    x = re.sub(r"\s+", " ", x)
    broken_tokens = [
        "Players Series Matches",
        "Statistics",
        "All Rounders",
        "World Cup",
        "Indian Premier League",
    ]
    if len(x) > 90 or any(tok in x for tok in broken_tokens):
        return "Unknown"
    if x == "" or x.lower() in {"nan", "none", "null"}:
        return "Unknown"
    return x


def parse_match_date(series):
    return pd.to_datetime(series, errors="coerce", dayfirst=True)


def infer_valid_winner(row):
    winner = clean_team_name(row.get("match_winner"))
    t1 = row.get("team1")
    t2 = row.get("team2")
    if winner == t1:
        return t1
    if winner == t2:
        return t2
    return np.nan


def row_first_innings_score(row):
    """Estimate previous first-innings score from team1 batting runs.

    Important: this score is never used from the current match as a model input.
    It is used only after feature creation to update venue history for later matches.
    """
    total = 0.0
    found = False
    for i in range(1, 12):
        col = f"team1_bat{i}_runs"
        if col in row.index:
            val = pd.to_numeric(row[col], errors="coerce")
            if pd.notna(val):
                total += float(val)
                found = True
    return total if found else np.nan


# -----------------------------
# Cell 4: Load and clean dataset
# -----------------------------
def load_and_clean_dataset(path):
    df = pd.read_csv(path)
    original_rows = len(df)

    required = ["team1", "team2", "venue", "date", "match_winner"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["team1"] = df["team1"].apply(clean_team_name)
    df["team2"] = df["team2"].apply(clean_team_name)
    df["match_winner"] = df["match_winner"].apply(clean_team_name)
    df["venue"] = df["venue"].apply(clean_venue_name)
    df["date"] = parse_match_date(df["date"])

    df = df.dropna(subset=["team1", "team2", "venue", "date"])
    df = df[df["team1"] != df["team2"]].copy()

    df["valid_winner"] = df.apply(infer_valid_winner, axis=1)
    df = df.dropna(subset=["valid_winner"]).copy()
    df["team1_won"] = (df["valid_winner"] == df["team1"]).astype(int)

    # Precompute first innings score once. This value is used only to update
    # future venue history after each match, never as a direct current-match input.
    run_cols = [f"team1_bat{i}_runs" for i in range(1, 12) if f"team1_bat{i}_runs" in df.columns]
    if run_cols:
        df["_first_innings_score_for_history"] = df[run_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1)
        df.loc[df["_first_innings_score_for_history"] <= 0, "_first_innings_score_for_history"] = np.nan
    else:
        df["_first_innings_score_for_history"] = np.nan

    sort_cols = ["date"]
    if "match_code" in df.columns:
        sort_cols.append("match_code")
    df = df.sort_values(sort_cols).reset_index(drop=True)

    removed_rows = original_rows - len(df)
    print(f"Original rows: {original_rows}")
    print(f"Rows after cleaning/resolved matches: {len(df)}")
    print(f"Removed rows: {removed_rows}")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    return df, removed_rows

clean_df, removed_rows = load_and_clean_dataset(DATA_PATH)


# -----------------------------
# Cell 5: Feature engineering - venue history only
# -----------------------------
def create_venue_history_features(df, prior_matches=10.0):
    """Create venue-history features chronologically using only past matches.

    Engineered feature family: Venue history only.
    The current match updates all venue/team-at-venue histories only after its pre-match
    feature values have already been written.
    """
    df = df.copy().reset_index(drop=True)

    all_scores = df["_first_innings_score_for_history"].dropna().astype(float).tolist()
    global_first_score_prior = float(np.median(all_scores)) if all_scores else 250.0

    venue_matches = defaultdict(int)
    venue_first_score_sum = defaultdict(float)
    venue_score_count = defaultdict(int)
    venue_bat_first_wins = defaultdict(int)
    venue_chasing_wins = defaultdict(int)

    team_venue_matches = defaultdict(int)
    team_venue_wins = defaultdict(int)

    feature_rows = []
    for _, row in df.iterrows():
        venue = row["venue"]
        team1 = row["team1"]
        team2 = row["team2"]
        y = int(row["team1_won"])

        vm = venue_matches[venue]
        sc = venue_score_count[venue]
        ss = venue_first_score_sum[venue]

        team1_key = (team1, venue)
        team2_key = (team2, venue)
        team1_venue_matches = team_venue_matches[team1_key]
        team2_venue_matches = team_venue_matches[team2_key]

        venue_avg_first_innings_score_before = (
            ss + prior_matches * global_first_score_prior
        ) / (sc + prior_matches)

        venue_bat_first_win_rate_before = (
            venue_bat_first_wins[venue] + 0.5 * prior_matches
        ) / (vm + prior_matches)

        venue_chasing_win_rate_before = (
            venue_chasing_wins[venue] + 0.5 * prior_matches
        ) / (vm + prior_matches)

        team1_venue_win_rate_before = (
            team_venue_wins[team1_key] + 0.5 * prior_matches
        ) / (team1_venue_matches + prior_matches)

        team2_venue_win_rate_before = (
            team_venue_wins[team2_key] + 0.5 * prior_matches
        ) / (team2_venue_matches + prior_matches)

        feature_rows.append({
            "venue_avg_first_innings_score_before": venue_avg_first_innings_score_before,
            "venue_chasing_win_rate_before": venue_chasing_win_rate_before,
            "venue_bat_first_win_rate_before": venue_bat_first_win_rate_before,
            "team1_venue_win_rate_before": team1_venue_win_rate_before,
            "team2_venue_win_rate_before": team2_venue_win_rate_before,
            "venue_win_rate_diff_before": team1_venue_win_rate_before - team2_venue_win_rate_before,
            "venue_matches_played_before": vm,
            "team1_venue_matches_before": team1_venue_matches,
            "team2_venue_matches_before": team2_venue_matches,
            "team_venue_matches_diff_before": team1_venue_matches - team2_venue_matches,
        })

        # Update histories after saving current pre-match features.
        venue_matches[venue] += 1
        if y == 1:
            venue_bat_first_wins[venue] += 1
        else:
            venue_chasing_wins[venue] += 1

        first_innings_score = row.get("_first_innings_score_for_history", np.nan)
        if pd.notna(first_innings_score) and np.isfinite(first_innings_score) and first_innings_score > 0:
            venue_first_score_sum[venue] += float(first_innings_score)
            venue_score_count[venue] += 1

        team_venue_matches[team1_key] += 1
        team_venue_matches[team2_key] += 1
        if y == 1:
            team_venue_wins[team1_key] += 1
        else:
            team_venue_wins[team2_key] += 1

    return pd.concat([df.reset_index(drop=True), pd.DataFrame(feature_rows)], axis=1)


# -----------------------------
# Cell 6: Train/validation/test split and encoding helpers
# -----------------------------
def chronological_split(df, train_frac=0.64, val_frac=0.16):
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    return (
        df.iloc[:train_end].copy(),
        df.iloc[train_end:val_end].copy(),
        df.iloc[:val_end].copy(),
        df.iloc[val_end:].copy(),
    )


def make_design_matrices(train_df, other_df, numeric_features, categorical_features):
    selected = numeric_features + categorical_features
    train_x = pd.get_dummies(train_df[selected], columns=categorical_features, dummy_na=False)
    other_x = pd.get_dummies(other_df[selected], columns=categorical_features, dummy_na=False)
    train_x, other_x = train_x.align(other_x, join="left", axis=1, fill_value=0)
    return train_x, other_x


def make_trainval_test_matrices(trainval_df, test_df, numeric_features, categorical_features):
    selected = numeric_features + categorical_features
    trainval_x = pd.get_dummies(trainval_df[selected], columns=categorical_features, dummy_na=False)
    test_x = pd.get_dummies(test_df[selected], columns=categorical_features, dummy_na=False)
    trainval_x, test_x = trainval_x.align(test_x, join="left", axis=1, fill_value=0)
    return trainval_x, test_x


def evaluate_threshold(y_true, proba, threshold):
    pred = (proba >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "roc_auc": roc_auc_score(y_true, proba),
    }


def choose_threshold_from_validation(y_true, proba):
    best = None
    for threshold in np.linspace(0.35, 0.65, 301):
        metrics = evaluate_threshold(y_true, proba, threshold)
        key = (metrics["accuracy"], metrics["balanced_accuracy"], metrics["roc_auc"])
        if best is None or key > best["key"]:
            best = {"threshold": float(threshold), "metrics": metrics, "key": key}
    return best["threshold"], best["metrics"]


def build_xgb(config):
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=1,
        tree_method="hist",
        **config,
    )


# -----------------------------
# Cell 7: Validation search using venue-history features only
# -----------------------------
VENUE_CORE_FEATURES = [
    "venue_avg_first_innings_score_before",
    "venue_chasing_win_rate_before",
    "venue_bat_first_win_rate_before",
    "team1_venue_win_rate_before",
    "team2_venue_win_rate_before",
]

VENUE_EXTENDED_FEATURES = VENUE_CORE_FEATURES + [
    "venue_win_rate_diff_before",
    "venue_matches_played_before",
    "team1_venue_matches_before",
    "team2_venue_matches_before",
    "team_venue_matches_diff_before",
]

FEATURE_SETS = {
    "venue_core": VENUE_CORE_FEATURES,
    "venue_extended": VENUE_EXTENDED_FEATURES,
}

# Raw pre-match context columns. These are not additional engineered cricket features.
CATEGORICAL_FEATURES = ["team1", "team2", "venue"]

XGB_CONFIGS = {
    "regularized_depth2_small": {
        "n_estimators": 120,
        "max_depth": 2,
        "learning_rate": 0.05,
        "subsample": 0.80,
        "colsample_bytree": 0.80,
        "min_child_weight": 8,
        "reg_alpha": 1.0,
        "reg_lambda": 8.0,
    },
    "regularized_depth2_medium": {
        "n_estimators": 180,
        "max_depth": 2,
        "learning_rate": 0.035,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "min_child_weight": 6,
        "reg_alpha": 0.5,
        "reg_lambda": 6.0,
    },
    "regularized_depth3_small": {
        "n_estimators": 140,
        "max_depth": 3,
        "learning_rate": 0.03,
        "subsample": 0.80,
        "colsample_bytree": 0.80,
        "min_child_weight": 10,
        "reg_alpha": 1.0,
        "reg_lambda": 10.0,
    },
}

PRIORS_TO_TRY = [2.0, 5.0, 8.0, 10.0, 16.0, 24.0, 32.0]

validation_records = []
best = None

for prior in PRIORS_TO_TRY:
    print(f"Trying venue prior {prior}...", flush=True)
    feature_df = create_venue_history_features(clean_df, prior_matches=prior)
    train_df, val_df, trainval_df, test_df = chronological_split(feature_df)

    for feature_set_name, numeric_features in FEATURE_SETS.items():
        print(f"  Feature set {feature_set_name}", flush=True)
        for config_name, config in XGB_CONFIGS.items():
            x_train, x_val = make_design_matrices(train_df, val_df, numeric_features, CATEGORICAL_FEATURES)
            y_train = train_df["team1_won"]
            y_val = val_df["team1_won"]

            model = build_xgb(config)
            model.fit(x_train, y_train)
            val_proba = model.predict_proba(x_val)[:, 1]
            threshold, selected_metrics = choose_threshold_from_validation(y_val, val_proba)
            default_metrics = evaluate_threshold(y_val, val_proba, 0.50)

            record = {
                "prior_matches": prior,
                "feature_set": feature_set_name,
                "xgb_config": config_name,
                "selected_threshold": threshold,
                "val_accuracy_selected_threshold": selected_metrics["accuracy"],
                "val_balanced_accuracy_selected_threshold": selected_metrics["balanced_accuracy"],
                "val_roc_auc": selected_metrics["roc_auc"],
                "val_accuracy_at_0_50": default_metrics["accuracy"],
                "val_balanced_accuracy_at_0_50": default_metrics["balanced_accuracy"],
            }
            validation_records.append(record)

            key = (
                record["val_accuracy_selected_threshold"],
                record["val_balanced_accuracy_selected_threshold"],
                record["val_roc_auc"],
            )
            if best is None or key > best["key"]:
                best = {
                    "key": key,
                    "record": record,
                    "prior": prior,
                    "feature_set_name": feature_set_name,
                    "numeric_features": numeric_features,
                    "config_name": config_name,
                    "config": config,
                    "threshold": threshold,
                }

validation_results = pd.DataFrame(validation_records).sort_values(
    ["val_accuracy_selected_threshold", "val_balanced_accuracy_selected_threshold", "val_roc_auc"],
    ascending=False,
).reset_index(drop=True)
validation_results.to_csv(OUTPUT_DIR / "validation_venue_model_selection_results.csv", index=False)

print("Best validation setting:")
print(json.dumps(best["record"], indent=2))


# -----------------------------
# Cell 8: Final training on train+validation and test evaluation
# -----------------------------
final_df = create_venue_history_features(clean_df, prior_matches=best["prior"])
train_df, val_df, trainval_df, test_df = chronological_split(final_df)

numeric_features = best["numeric_features"]
categorical_features = CATEGORICAL_FEATURES

x_trainval, x_test = make_trainval_test_matrices(trainval_df, test_df, numeric_features, categorical_features)
y_trainval = trainval_df["team1_won"]
y_test = test_df["team1_won"]

final_model = build_xgb(best["config"])
final_model.fit(x_trainval, y_trainval)

trainval_proba = final_model.predict_proba(x_trainval)[:, 1]
test_proba = final_model.predict_proba(x_test)[:, 1]

trainval_pred = (trainval_proba >= best["threshold"]).astype(int)
test_pred = (test_proba >= best["threshold"]).astype(int)

trainval_accuracy = accuracy_score(y_trainval, trainval_pred)
test_accuracy = accuracy_score(y_test, test_pred)
test_balanced_accuracy = balanced_accuracy_score(y_test, test_pred)
test_roc_auc = roc_auc_score(y_test, test_proba)

print("\nFinal test results:")
print(f"Train+validation accuracy: {trainval_accuracy:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")
print(f"Test balanced accuracy: {test_balanced_accuracy:.4f}")
print(f"Test ROC-AUC: {test_roc_auc:.4f}")


# -----------------------------
# Cell 9: Save all results
# -----------------------------
metrics_summary = {
    "model": "Existing XGBoost with venue history features only",
    "feature_engineering_family": "Pre-match Venue History Features",
    "leakage_safety": "Venue features are computed chronologically before each match, then histories are updated after the match.",
    "removed_feature_families": [
        "Elo",
        "Recent Form",
        "Bayesian Historical Strength",
        "Opponent-Adjusted Historical Strength",
        "Colley Matrix Strength",
        "Regularized Massey Matrix Strength",
        "Head-to-head History",
    ],
    "rows_after_cleaning": int(len(final_df)),
    "removed_rows": int(removed_rows),
    "train_rows": int(len(train_df)),
    "validation_rows": int(len(val_df)),
    "train_validation_rows": int(len(trainval_df)),
    "test_rows": int(len(test_df)),
    "test_start_date": str(test_df["date"].min().date()),
    "test_end_date": str(test_df["date"].max().date()),
    "best_prior_matches": float(best["prior"]),
    "best_feature_set": best["feature_set_name"],
    "best_xgb_config": best["config_name"],
    "selected_probability_threshold_from_validation": float(best["threshold"]),
    "train_validation_accuracy": float(trainval_accuracy),
    "test_accuracy": float(test_accuracy),
    "test_balanced_accuracy": float(test_balanced_accuracy),
    "test_roc_auc": float(test_roc_auc),
    "numeric_features_used": numeric_features,
    "categorical_features_used": categorical_features,
    "excluded_leakage_columns_note": "All batting/bowling scorecard columns, match_winner, match_result, match_outcome_type, toss_winner, and team1_bat_first are excluded from model input.",
    "strict_before_toss_warning": "The original dataset has team1_bat_first=1 for all rows, so team1/team2 order may reflect innings order. This code excludes toss and batting-first columns, but strict before-toss research should eventually randomize or canonicalize team order independently of innings order.",
}

with open(OUTPUT_DIR / "metrics_summary.json", "w") as f:
    json.dump(metrics_summary, f, indent=2)

pd.DataFrame(classification_report(y_test, test_pred, output_dict=True, zero_division=0)).transpose().to_csv(
    OUTPUT_DIR / "classification_report.csv"
)

cm = confusion_matrix(y_test, test_pred)
pd.DataFrame(
    cm,
    index=["actual_team2_win", "actual_team1_win"],
    columns=["pred_team2_win", "pred_team1_win"],
).to_csv(OUTPUT_DIR / "confusion_matrix.csv")

plt.figure(figsize=(6, 5))
plt.imshow(cm)
plt.title("Confusion Matrix - XGBoost Venue History Model")
plt.xlabel("Predicted label")
plt.ylabel("Actual label")
plt.xticks([0, 1], ["Team2 win", "Team1 win"])
plt.yticks([0, 1], ["Team2 win", "Team1 win"])
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=160)
plt.close()

fpr, tpr, roc_thresholds = roc_curve(y_test, test_proba)
pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": roc_thresholds}).to_csv(
    OUTPUT_DIR / "roc_curve_points.csv", index=False
)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"ROC-AUC = {test_roc_auc:.4f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.title("ROC Curve - XGBoost Venue History Model")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "roc_curve.png", dpi=160)
plt.close()

feature_importance = pd.DataFrame({
    "feature": x_trainval.columns,
    "importance": final_model.feature_importances_,
}).sort_values("importance", ascending=False).reset_index(drop=True)
feature_importance.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)

plt.figure(figsize=(9, 6))
top = feature_importance.head(20).iloc[::-1]
plt.barh(top["feature"], top["importance"])
plt.title("Top 20 Feature Importances - Venue History Model")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "feature_importance_top20.png", dpi=160)
plt.close()

split_summary = pd.DataFrame([
    {"split": "train", "rows": len(train_df), "start_date": train_df["date"].min(), "end_date": train_df["date"].max()},
    {"split": "validation", "rows": len(val_df), "start_date": val_df["date"].min(), "end_date": val_df["date"].max()},
    {"split": "train_validation", "rows": len(trainval_df), "start_date": trainval_df["date"].min(), "end_date": trainval_df["date"].max()},
    {"split": "test", "rows": len(test_df), "start_date": test_df["date"].min(), "end_date": test_df["date"].max()},
])
split_summary.to_csv(OUTPUT_DIR / "split_summary.csv", index=False)

id_cols = ["date", "team1", "team2", "venue", "valid_winner", "team1_won"]
if "match_code" in test_df.columns:
    id_cols = ["match_code"] + id_cols
predictions = test_df[id_cols + numeric_features].copy()
predictions["predicted_probability_team1_win"] = test_proba
predictions["predicted_team1_won"] = test_pred
predictions["predicted_winner"] = np.where(predictions["predicted_team1_won"] == 1, predictions["team1"], predictions["team2"])
predictions.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)

final_df.to_csv(OUTPUT_DIR / "cleaned_dataset_with_venue_history_features.csv", index=False)

model_artifact = {
    "model": final_model,
    "train_columns": list(x_trainval.columns),
    "numeric_features": numeric_features,
    "categorical_features": categorical_features,
    "threshold": best["threshold"],
    "best_prior_matches": best["prior"],
    "cleaning_notes": "Use the same cleaning and venue-history feature generation functions before prediction.",
}
joblib.dump(model_artifact, OUTPUT_DIR / "xgboost_venue_history_model.joblib")

summary_text = f"""
XGBoost Venue History Model Summary
===================================

Feature engineering family used:
Pre-match Venue History Features only

Selected engineered features:
{chr(10).join('- ' + f for f in numeric_features)}

Raw pre-match context columns used:
{chr(10).join('- ' + f for f in categorical_features)}

Rows after cleaning: {len(final_df)}
Removed unresolved/invalid rows: {removed_rows}
Train + validation rows: {len(trainval_df)}
Test rows: {len(test_df)}
Test period: {test_df['date'].min().date()} to {test_df['date'].max().date()}

Best prior selected from validation only: {best['prior']}
Best feature set: {best['feature_set_name']}
Best XGBoost config: {best['config_name']}
Selected probability threshold from validation only: {best['threshold']:.6f}

Train + validation accuracy: {trainval_accuracy:.4f}
Test accuracy: {test_accuracy:.4f}
Test balanced accuracy: {test_balanced_accuracy:.4f}
Test ROC-AUC: {test_roc_auc:.4f}

Leakage-control note:
Venue features are calculated before each match using only previous matches at the same venue and previous team-at-venue records. Current match scores/results are used only after the current row's features are saved, so they can affect future rows only.

Important thesis warning:
The original dataset has team1_bat_first = 1 for every row, so team1/team2 order may reflect innings order. This code excludes toss_winner and team1_bat_first, but a strict before-toss thesis should eventually canonicalize or randomize team order independently of innings order.
""".strip()

with open(OUTPUT_DIR / "run_summary.txt", "w") as f:
    f.write(summary_text)

print("\nSaved outputs to:", OUTPUT_DIR.resolve())
print(summary_text)
