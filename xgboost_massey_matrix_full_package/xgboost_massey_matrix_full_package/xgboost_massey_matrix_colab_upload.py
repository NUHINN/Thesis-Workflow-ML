# %%
# ===============================================================
# Thesis-safe Cricket Match Winner Prediction
# Plain XGBoost + ONE feature engineering family:
# Pre-match Regularized Massey Matrix Team Strength
#
# Colab-ready: upload CSV directly when running in Google Colab
# ===============================================================

# %%
# ======================================
# Cell 1: Install/import required packages
# ======================================

import os
import re
import json
import sys
import zipfile
import warnings
import subprocess
from pathlib import Path
from collections import defaultdict

warnings.filterwarnings("ignore")

try:
    import xgboost  # noqa: F401
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "xgboost", "joblib"])

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)
from xgboost import XGBClassifier

RANDOM_STATE = 42
TEST_SIZE = 0.20
VALIDATION_SIZE_WITHIN_TRAIN = 0.20

# Regularized Massey prior values.
# These are deliberately high/stable values because the Massey method can overreact
# when early match history is small. Selection is made using validation data only.
MASSEY_PRIOR_CANDIDATES = [60, 80, 100, 120, 160]

# Conservative XGBoost configurations.
# This is model selection, not extra feature engineering.
MODEL_CONFIGS = [
    {
        "model_config_name": "regularized_depth1_default",
        "params": dict(
            n_estimators=180,
            max_depth=1,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=6,
            reg_lambda=8.0,
            reg_alpha=0.20,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=2,
        ),
    },
    {
        "model_config_name": "regularized_depth1_slow",
        "params": dict(
            n_estimators=220,
            max_depth=1,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=6,
            reg_lambda=8.0,
            reg_alpha=0.20,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=2,
        ),
    },
    {
        "model_config_name": "regularized_depth2_small",
        "params": dict(
            n_estimators=160,
            max_depth=2,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=8,
            reg_lambda=12.0,
            reg_alpha=0.30,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=2,
        ),
    },
]

print("Setup complete.")

# %%
# ======================================
# Cell 2: Upload dataset in Colab
# ======================================

def get_dataset_path():
    """Upload CSV in Colab; fall back to local Dataset-1.csv outside Colab."""
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
        if not Path(local_path).exists() and Path("/mnt/data/Dataset-1.csv").exists():
            local_path = "/mnt/data/Dataset-1.csv"
        print(f"Using local dataset file: {local_path}")
        return local_path

DATA_PATH = get_dataset_path()
OUTPUT_DIR = Path(DATA_PATH).resolve().parent / "xgb_massey_matrix_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"DATA_PATH  = {DATA_PATH}")
print(f"OUTPUT_DIR = {OUTPUT_DIR}")

# %%
# ======================================
# Cell 3: Robust cleaning helpers
# ======================================

def clean_text(value):
    """Basic text cleaning without changing the real meaning."""
    if pd.isna(value):
        return np.nan
    value = str(value).replace("\xa0", " ")
    value = re.sub(r"\s+", " ", value).strip()
    if value.lower() in ["nan", "none", "null", ""]:
        return np.nan
    return value


def clean_team_name(value):
    """Fix obvious scraping inconsistency in team names."""
    value = clean_text(value)
    if pd.isna(value):
        return np.nan
    team_fixes = {
        "West": "West Indies",
    }
    return team_fixes.get(value, value)


def clean_venue(value):
    """Clean venue and replace broken scraped menu text with Unknown."""
    value = clean_text(value)
    if pd.isna(value) or value == "":
        return "Unknown"
    broken_venue_patterns = [
        "Players Series Matches",
        "All Rounders Batting Bowling",
        "Captaincy Countries Dism",
    ]
    if len(value) > 80 or any(pattern in value for pattern in broken_venue_patterns):
        return "Unknown"
    return value


def make_one_hot_encoder():
    """Compatible with old and new scikit-learn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)

# %%
# ======================================
# Cell 4: Load, clean, and inspect dataset
# ======================================

def load_and_clean_dataset(path: str) -> pd.DataFrame:
    """
    Loads and cleans dataset for honest pre-match prediction.

    Kept rows:
    - valid date
    - valid team1/team2/venue/match_winner
    - resolved matches only, where match_winner is exactly team1 or team2 after cleaning

    Excluded from model later:
    - all current-match scorecard columns
    - match_winner as a feature
    - match_result and match_outcome_type
    - toss_winner and team1_bat_first
    """
    df = pd.read_csv(path)
    original_rows = len(df)

    required_columns = ["team1", "team2", "venue", "date", "match_winner"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    for col in ["team1", "team2", "match_winner", "toss_winner"]:
        if col in df.columns:
            df[col] = df[col].apply(clean_team_name)

    df["venue"] = df["venue"].apply(clean_venue)

    for col in ["match_type", "series"]:
        if col in df.columns:
            df[col] = df[col].apply(clean_text)

    df["date_parsed"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["date_parsed", "team1", "team2", "venue", "match_winner"]).copy()
    df = df[df["team1"] != df["team2"]].copy()

    resolved_mask = (df["match_winner"] == df["team1"]) | (df["match_winner"] == df["team2"])
    df = df[resolved_mask].copy()

    if "match_code" in df.columns:
        df["match_code_num"] = pd.to_numeric(df["match_code"], errors="coerce")
        df["match_code_num"] = df["match_code_num"].fillna(pd.Series(np.arange(len(df)), index=df.index))
    else:
        df["match_code"] = np.arange(len(df))
        df["match_code_num"] = np.arange(len(df))

    df = df.sort_values(["date_parsed", "match_code_num"]).reset_index(drop=True)
    df["team1_won"] = (df["match_winner"] == df["team1"]).astype(int)

    removed_rows = original_rows - len(df)
    broken_venue_count = int((df["venue"] == "Unknown").sum())

    print(f"Original rows: {original_rows}")
    print(f"Rows after cleaning/resolved-match filtering: {len(df)}")
    print(f"Rows removed: {removed_rows}")
    print(f"Rows with venue set to Unknown after cleaning: {broken_venue_count}")
    print(f"Date range: {df['date_parsed'].min().date()} to {df['date_parsed'].max().date()}")
    print("Target distribution:")
    print(df["team1_won"].value_counts().rename(index={1: "team1_won", 0: "team2_won"}))
    print("Target distribution percentage:")
    print((df["team1_won"].value_counts(normalize=True) * 100).round(2).rename(index={1: "team1_won", 0: "team2_won"}))

    return df

clean_df = load_and_clean_dataset(DATA_PATH)
clean_df.head()

# %%
# =====================================================================
# Cell 5: ONE feature engineering only: Regularized Massey Matrix rating
# =====================================================================

# This creates one feature-engineering family only:
# pre-match Regularized Massey Matrix team-strength features.
#
# Massey rating is a formal sports ranking method. It estimates team strength
# from previous results through a linear system. Here the match margin is set
# to +1 for winner and -1 for loser because the dataset is winner-only for
# pre-match prediction. A ridge-style prior stabilizes early low-history ratings.
#
# Anti-leakage logic:
# 1. Read both teams' Massey ratings BEFORE the current match.
# 2. Save those pre-match values as features.
# 3. Update historical win/loss differential AFTER the current match.
# 4. Recompute ratings for the next row only.
#
# Created columns:
# - team1_massey_strength_pre
# - team2_massey_strength_pre
# - massey_strength_diff
# - massey_abs_strength_diff
# - team1_massey_matches_pre
# - team2_massey_matches_pre
# - massey_matches_diff
#
# No Elo, no recent-form EWMA, no Bayesian historical features,
# no opponent-adjusted features, and no Colley Matrix features are used.


def compute_massey_ratings(active_teams, pair_counts, point_diff, prior_strength: float) -> dict:
    """
    Computes regularized Massey ratings using history available so far.

    M_ii receives the team's game count plus a prior.
    M_ij receives negative pair counts.
    point_diff stores +1 for previous wins and -1 for previous losses.

    A neutral team with no history receives 0.0.
    """
    teams = list(active_teams)
    n_teams = len(teams)
    if n_teams == 0:
        return {}

    index = {team: i for i, team in enumerate(teams)}
    M = np.eye(n_teams) * prior_strength
    p = np.zeros(n_teams)

    for (team_a, team_b), count in pair_counts.items():
        if team_a in index and team_b in index:
            i = index[team_a]
            j = index[team_b]
            M[i, i] += count
            M[i, j] -= count

    for team, value in point_diff.items():
        if team in index:
            p[index[team]] += value

    # Center ratings so that team strengths are relative, not arbitrary absolute values.
    if n_teams > 1:
        M[-1, :] = 1.0
        p[-1] = 0.0

    try:
        ratings = np.linalg.solve(M, p)
    except np.linalg.LinAlgError:
        ratings = np.linalg.lstsq(M, p, rcond=None)[0]

    return {team: float(ratings[index[team]]) for team in teams}


def add_massey_strength_features(df: pd.DataFrame, prior_strength: float) -> pd.DataFrame:
    """
    Adds pre-match Regularized Massey Matrix strength features using only past results.

    Anti-leakage rule:
    Features for each row are saved BEFORE updating history with that row's result.
    """
    df = df.copy().sort_values(["date_parsed", "match_code_num"]).reset_index(drop=True)

    pair_counts = defaultdict(float)
    point_diff = defaultdict(float)
    matches = defaultdict(float)
    active_teams = set()
    current_ratings = {}

    team1_strength_values = []
    team2_strength_values = []
    strength_diff_values = []
    abs_strength_diff_values = []
    team1_match_count_values = []
    team2_match_count_values = []
    match_count_diff_values = []

    for _, row in df.iterrows():
        team1 = row["team1"]
        team2 = row["team2"]
        winner = row["match_winner"]

        team1_strength_pre = current_ratings.get(team1, 0.0)
        team2_strength_pre = current_ratings.get(team2, 0.0)
        team1_matches_pre = float(matches[team1])
        team2_matches_pre = float(matches[team2])
        strength_diff = float(team1_strength_pre - team2_strength_pre)

        # Save pre-match values first. This is the anti-leakage step.
        team1_strength_values.append(float(team1_strength_pre))
        team2_strength_values.append(float(team2_strength_pre))
        strength_diff_values.append(strength_diff)
        abs_strength_diff_values.append(abs(strength_diff))
        team1_match_count_values.append(team1_matches_pre)
        team2_match_count_values.append(team2_matches_pre)
        match_count_diff_values.append(float(team1_matches_pre - team2_matches_pre))

        # Update only AFTER saving the features for this row.
        active_teams.update([team1, team2])
        pair_counts[(team1, team2)] += 1.0
        pair_counts[(team2, team1)] += 1.0

        if winner == team1:
            point_diff[team1] += 1.0
            point_diff[team2] -= 1.0
        else:
            point_diff[team2] += 1.0
            point_diff[team1] -= 1.0

        matches[team1] += 1.0
        matches[team2] += 1.0

        current_ratings = compute_massey_ratings(
            active_teams=active_teams,
            pair_counts=pair_counts,
            point_diff=point_diff,
            prior_strength=prior_strength,
        )

    df["team1_massey_strength_pre"] = team1_strength_values
    df["team2_massey_strength_pre"] = team2_strength_values
    df["massey_strength_diff"] = strength_diff_values
    df["massey_abs_strength_diff"] = abs_strength_diff_values
    df["team1_massey_matches_pre"] = team1_match_count_values
    df["team2_massey_matches_pre"] = team2_match_count_values
    df["massey_matches_diff"] = match_count_diff_values

    return df

example_massey_df = add_massey_strength_features(clean_df, prior_strength=60)
example_massey_df[[
    "date", "team1", "team2", "match_winner",
    "team1_massey_strength_pre", "team2_massey_strength_pre",
    "massey_strength_diff", "team1_massey_matches_pre", "team2_massey_matches_pre",
    "team1_won"
]].head(10)

# %%
# ===============================================
# Cell 6: Chronological train/validation/test split
# ===============================================

def chronological_split(df: pd.DataFrame):
    n_rows = len(df)
    test_start = int(n_rows * (1.0 - TEST_SIZE))

    train_val_df = df.iloc[:test_start].copy()
    test_df = df.iloc[test_start:].copy()

    val_start = int(len(train_val_df) * (1.0 - VALIDATION_SIZE_WITHIN_TRAIN))
    train_df = train_val_df.iloc[:val_start].copy()
    val_df = train_val_df.iloc[val_start:].copy()

    return train_df, val_df, train_val_df, test_df

train_df, val_df, train_val_df, test_df = chronological_split(example_massey_df)

split_summary = pd.DataFrame([
    {
        "split": "train",
        "rows": len(train_df),
        "start_date": train_df["date_parsed"].min().date(),
        "end_date": train_df["date_parsed"].max().date(),
        "team1_win_rate": train_df["team1_won"].mean(),
    },
    {
        "split": "validation",
        "rows": len(val_df),
        "start_date": val_df["date_parsed"].min().date(),
        "end_date": val_df["date_parsed"].max().date(),
        "team1_win_rate": val_df["team1_won"].mean(),
    },
    {
        "split": "train_plus_validation",
        "rows": len(train_val_df),
        "start_date": train_val_df["date_parsed"].min().date(),
        "end_date": train_val_df["date_parsed"].max().date(),
        "team1_win_rate": train_val_df["team1_won"].mean(),
    },
    {
        "split": "test",
        "rows": len(test_df),
        "start_date": test_df["date_parsed"].min().date(),
        "end_date": test_df["date_parsed"].max().date(),
        "team1_win_rate": test_df["team1_won"].mean(),
    },
])

split_summary

# %%
# =============================================================
# Cell 7: Build XGBoost pipeline and tune Massey prior/config
# =============================================================

CATEGORICAL_FEATURES = ["team1", "team2", "venue"]
NUMERIC_FEATURES = [
    "team1_massey_strength_pre",
    "team2_massey_strength_pre",
    "massey_strength_diff",
    "massey_abs_strength_diff",
    "team1_massey_matches_pre",
    "team2_massey_matches_pre",
    "massey_matches_diff",
]
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES
TARGET_COLUMN = "team1_won"


def build_xgb_pipeline(model_params: dict) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", make_one_hot_encoder(), CATEGORICAL_FEATURES),
            ("num", "passthrough", NUMERIC_FEATURES),
        ],
        remainder="drop",
    )

    model = XGBClassifier(**model_params)
    pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
    return pipeline


def find_best_threshold(y_true, probabilities):
    """Select a probability threshold using validation data only."""
    candidates = np.unique(np.concatenate([probabilities, np.array([0.50])]))
    best_threshold = 0.50
    best_accuracy = -1.0
    best_balanced_accuracy = -1.0

    for threshold in candidates:
        preds = (probabilities >= threshold).astype(int)
        acc = accuracy_score(y_true, preds)
        bal_acc = balanced_accuracy_score(y_true, preds)
        if (acc > best_accuracy) or (acc == best_accuracy and bal_acc > best_balanced_accuracy):
            best_threshold = float(threshold)
            best_accuracy = float(acc)
            best_balanced_accuracy = float(bal_acc)

    return best_threshold, best_accuracy, best_balanced_accuracy


def evaluate_candidate(prior_strength: float, model_config: dict) -> dict:
    massey_df = add_massey_strength_features(clean_df, prior_strength=prior_strength)
    train_df, val_df, _, _ = chronological_split(massey_df)

    pipeline = build_xgb_pipeline(model_config["params"])
    pipeline.fit(train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN])

    val_prob = pipeline.predict_proba(val_df[FEATURE_COLUMNS])[:, 1]
    val_pred_fixed = (val_prob >= 0.50).astype(int)
    best_threshold, val_acc_at_threshold, val_bal_at_threshold = find_best_threshold(
        val_df[TARGET_COLUMN].values,
        val_prob,
    )

    return {
        "prior_strength": prior_strength,
        "model_config_name": model_config["model_config_name"],
        "validation_accuracy_fixed_0_50": accuracy_score(val_df[TARGET_COLUMN], val_pred_fixed),
        "validation_balanced_accuracy_fixed_0_50": balanced_accuracy_score(val_df[TARGET_COLUMN], val_pred_fixed),
        "validation_roc_auc": roc_auc_score(val_df[TARGET_COLUMN], val_prob),
        "selected_probability_threshold_from_validation": best_threshold,
        "validation_accuracy_at_selected_threshold": val_acc_at_threshold,
        "validation_balanced_accuracy_at_selected_threshold": val_bal_at_threshold,
    }

validation_rows = []
for prior_strength in MASSEY_PRIOR_CANDIDATES:
    for model_config in MODEL_CONFIGS:
        validation_rows.append(evaluate_candidate(prior_strength, model_config))

validation_results = pd.DataFrame(validation_rows)

# Selection uses validation accuracy at the selected threshold first, then validation ROC-AUC.
# Test data is untouched here.
validation_results = validation_results.sort_values(
    [
        "validation_accuracy_at_selected_threshold",
        "validation_roc_auc",
        "validation_balanced_accuracy_at_selected_threshold",
    ],
    ascending=False,
).reset_index(drop=True)

best_prior_strength = float(validation_results.loc[0, "prior_strength"])
best_model_config_name = validation_results.loc[0, "model_config_name"]
best_threshold = float(validation_results.loc[0, "selected_probability_threshold_from_validation"])
best_model_config = next(config for config in MODEL_CONFIGS if config["model_config_name"] == best_model_config_name)

print(f"Best Massey prior_strength selected from validation only: {best_prior_strength}")
print(f"Best XGBoost config selected from validation only: {best_model_config_name}")
print(f"Best probability threshold selected from validation only: {best_threshold:.6f}")
validation_results

# %%
# ================================================
# Cell 8: Train final model and evaluate on test set
# ================================================

final_df = add_massey_strength_features(clean_df, prior_strength=best_prior_strength)
train_df, val_df, train_val_df, test_df = chronological_split(final_df)

final_pipeline = build_xgb_pipeline(best_model_config["params"])
final_pipeline.fit(train_val_df[FEATURE_COLUMNS], train_val_df[TARGET_COLUMN])

train_val_prob = final_pipeline.predict_proba(train_val_df[FEATURE_COLUMNS])[:, 1]
train_val_pred = (train_val_prob >= best_threshold).astype(int)

test_prob = final_pipeline.predict_proba(test_df[FEATURE_COLUMNS])[:, 1]
test_pred = (test_prob >= best_threshold).astype(int)

train_accuracy = accuracy_score(train_val_df[TARGET_COLUMN], train_val_pred)
test_accuracy = accuracy_score(test_df[TARGET_COLUMN], test_pred)
test_balanced_accuracy = balanced_accuracy_score(test_df[TARGET_COLUMN], test_pred)
test_roc_auc = roc_auc_score(test_df[TARGET_COLUMN], test_prob)

print("Final Results")
print("-------------")
print(f"Best Massey prior_strength: {best_prior_strength}")
print(f"Best XGBoost config: {best_model_config_name}")
print(f"Validation-selected probability threshold: {best_threshold:.6f}")
print(f"Train + validation accuracy: {train_accuracy:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")
print(f"Test balanced accuracy: {test_balanced_accuracy:.4f}")
print(f"Test ROC-AUC: {test_roc_auc:.4f}")
print()
print("Classification report:")
print(classification_report(test_df[TARGET_COLUMN], test_pred, target_names=["team2_won", "team1_won"]))

confusion_matrix(test_df[TARGET_COLUMN], test_pred)

# %%
# ======================================
# Cell 9: Save all useful result files
# ======================================

split_summary.to_csv(OUTPUT_DIR / "split_summary.csv", index=False)
validation_results.to_csv(OUTPUT_DIR / "validation_massey_model_selection_results.csv", index=False)

metrics_summary = {
    "feature_engineering_used": "Pre-match Regularized Massey Matrix team-strength only",
    "elo_used": False,
    "recent_form_used": False,
    "bayesian_history_win_rate_features_used": False,
    "opponent_adjusted_historical_strength_used": False,
    "colley_matrix_features_used": False,
    "best_massey_prior_strength_selected_from_validation_only": best_prior_strength,
    "best_xgboost_config_selected_from_validation_only": best_model_config_name,
    "probability_threshold_selected_from_validation_only": best_threshold,
    "model_selection_metric": "validation_accuracy_at_selected_threshold_then_validation_roc_auc_then_validation_balanced_accuracy_at_selected_threshold",
    "rows_used_after_cleaning": int(len(final_df)),
    "train_validation_rows": int(len(train_val_df)),
    "test_rows": int(len(test_df)),
    "train_validation_accuracy": float(train_accuracy),
    "test_accuracy": float(test_accuracy),
    "test_balanced_accuracy": float(test_balanced_accuracy),
    "test_roc_auc": float(test_roc_auc),
    "test_start_date": str(test_df["date_parsed"].min().date()),
    "test_end_date": str(test_df["date_parsed"].max().date()),
    "important_warning": "team1_bat_first is constant in the source dataset and team ordering may reflect innings order. This code excludes team1_bat_first and toss_winner. For strict before-toss prediction, team ordering should ideally be made independent of innings order.",
}

with open(OUTPUT_DIR / "metrics_summary.json", "w") as f:
    json.dump(metrics_summary, f, indent=4)

classification_report_df = pd.DataFrame(
    classification_report(test_df[TARGET_COLUMN], test_pred, target_names=["team2_won", "team1_won"], output_dict=True)
).transpose()
classification_report_df.to_csv(OUTPUT_DIR / "classification_report.csv")

cm = confusion_matrix(test_df[TARGET_COLUMN], test_pred)
cm_df = pd.DataFrame(cm, index=["actual_team2_won", "actual_team1_won"], columns=["predicted_team2_won", "predicted_team1_won"])
cm_df.to_csv(OUTPUT_DIR / "confusion_matrix.csv")

prediction_cols = [
    "date", "date_parsed", "team1", "team2", "venue", "match_winner", "team1_won",
    "team1_massey_strength_pre", "team2_massey_strength_pre", "massey_strength_diff",
    "massey_abs_strength_diff", "team1_massey_matches_pre", "team2_massey_matches_pre", "massey_matches_diff",
]
test_predictions = test_df[prediction_cols].copy()
test_predictions["predicted_probability_team1_win"] = test_prob
test_predictions["predicted_team1_won"] = test_pred
test_predictions["predicted_winner"] = np.where(test_predictions["predicted_team1_won"] == 1, test_predictions["team1"], test_predictions["team2"])
test_predictions["correct_prediction"] = (test_predictions["predicted_team1_won"] == test_predictions["team1_won"]).astype(int)
test_predictions.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)

final_df.to_csv(OUTPUT_DIR / "cleaned_dataset_with_massey_matrix_features.csv", index=False)

# Save ROC curve points and plot.
fpr, tpr, thresholds = roc_curve(test_df[TARGET_COLUMN], test_prob)
roc_points = pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thresholds})
roc_points.to_csv(OUTPUT_DIR / "roc_curve_points.csv", index=False)

plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f"ROC-AUC = {test_roc_auc:.4f}")
plt.plot([0, 1], [0, 1], linestyle="--", label="Random baseline")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - XGBoost Regularized Massey Matrix Model")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "roc_curve.png", dpi=200)
plt.close()

plt.figure(figsize=(5, 4))
plt.imshow(cm)
plt.title("Confusion Matrix - XGBoost Regularized Massey Matrix Model")
plt.xticks([0, 1], ["Pred team2", "Pred team1"])
plt.yticks([0, 1], ["Actual team2", "Actual team1"])
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha="center", va="center")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=200)
plt.close()

# Feature importance after preprocessing.
try:
    preprocessor = final_pipeline.named_steps["preprocess"]
    model = final_pipeline.named_steps["model"]
    cat_feature_names = list(preprocessor.named_transformers_["cat"].get_feature_names_out(CATEGORICAL_FEATURES))
    all_feature_names = cat_feature_names + NUMERIC_FEATURES
    importance_df = pd.DataFrame({
        "feature": all_feature_names,
        "importance_gain": model.feature_importances_,
    }).sort_values("importance_gain", ascending=False)
    importance_df.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)

    top_n = min(20, len(importance_df))
    plt.figure(figsize=(8, 6))
    plot_df = importance_df.head(top_n).iloc[::-1]
    plt.barh(plot_df["feature"], plot_df["importance_gain"])
    plt.xlabel("XGBoost Feature Importance")
    plt.title("Top Feature Importances - Regularized Massey Matrix Model")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_importance_top20.png", dpi=200)
    plt.close()
except Exception as e:
    print(f"Feature importance saving skipped due to: {e}")

joblib.dump(final_pipeline, OUTPUT_DIR / "xgboost_massey_matrix_pipeline.joblib")

run_summary = f"""
XGBoost Regularized Massey Matrix Model Summary
==============================================

Feature engineering used:
- Pre-match Regularized Massey Matrix team-strength only

Not used:
- Elo ratings
- Recent-form EWMA ratings
- Bayesian historical strength features
- Opponent-adjusted historical strength features
- Colley Matrix features
- Toss winner
- team1_bat_first
- Current-match batting/bowling/scorecard columns
- match_winner/match_result as features

Best hyperparameters selected from validation only:
- Massey prior_strength: {best_prior_strength}
- XGBoost config: {best_model_config_name}
- Probability threshold: {best_threshold:.6f}

Rows used after cleaning: {len(final_df)}
Train + validation rows: {len(train_val_df)}
Test rows: {len(test_df)}
Test period: {test_df['date_parsed'].min().date()} to {test_df['date_parsed'].max().date()}

Final performance:
- Train + validation accuracy: {train_accuracy:.4f}
- Test accuracy: {test_accuracy:.4f}
- Test balanced accuracy: {test_balanced_accuracy:.4f}
- Test ROC-AUC: {test_roc_auc:.4f}

Academic safety notes:
- Features are saved before the current match result is used.
- Chronological split is used instead of random split.
- The test set is not used to fit the model.
- Repeatedly trying many feature families against the same test period can still cause research overfitting. For final thesis reporting, keep this limitation clear.
- team1_bat_first is constant in the source dataset, so team ordering may reflect innings order. This code excludes team1_bat_first and toss_winner, but strict before-toss prediction should ideally randomize or standardize team ordering independent of innings.
"""

with open(OUTPUT_DIR / "run_summary.txt", "w") as f:
    f.write(run_summary)

print(f"All result files saved to: {OUTPUT_DIR}")
print(run_summary)

# %%
# ======================================
# Cell 10: Zip the result folder
# ======================================

zip_path = OUTPUT_DIR.with_suffix(".zip")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
    for file_path in OUTPUT_DIR.rglob("*"):
        zipf.write(file_path, arcname=file_path.relative_to(OUTPUT_DIR.parent))

print(f"Results zipped at: {zip_path}")
