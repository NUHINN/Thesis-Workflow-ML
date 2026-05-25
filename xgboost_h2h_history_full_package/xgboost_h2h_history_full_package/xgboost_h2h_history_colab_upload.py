# %%
# ===============================================================
# Thesis-safe Cricket Match Winner Prediction
# Plain XGBoost + ONE feature engineering family:
# Pre-match Head-to-Head Historical Features
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
from collections import defaultdict, deque

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

# Head-to-head smoothing candidates.
# These are selected using validation data only, not test data.
H2H_PRIOR_MATCHES_CANDIDATES = [8, 10, 12, 16, 20]
H2H_RECENT_WINDOW = 5
H2H_RECENT_PRIOR_MATCHES = 2.0

# Conservative XGBoost configurations.
# This is model selection, not extra cricket feature engineering.
MODEL_CONFIGS = [
    {
        "model_config_name": "regularized_depth1",
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
            tree_method="hist",
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
            tree_method="hist",
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
OUTPUT_DIR = Path(DATA_PATH).resolve().parent / "xgb_h2h_history_outputs"
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
# ==================================================================
# Cell 5: ONE feature engineering only: Head-to-head history features
# ==================================================================

# This creates one feature-engineering family only:
# pre-match head-to-head historical features between the two teams.
#
# Anti-leakage logic:
# 1. Read the teams' previous head-to-head record BEFORE the current match.
# 2. Save those pre-match values as features.
# 3. Update head-to-head history only AFTER the current match result.
#
# Main created columns:
# - team1_h2h_win_rate_before_match
# - team2_h2h_win_rate_before_match
# - h2h_win_rate_diff_before_match
# - h2h_matches_played_before
# - h2h_recent_win_rate_last_5
# - h2h_recent_win_rate_diff_last_5
# - h2h_pair_key
#
# No Elo, no recent-form EWMA, no Bayesian historical strength,
# no opponent-adjusted strength, no Colley Matrix, and no Massey Matrix features are used.


def add_h2h_history_features(
    df: pd.DataFrame,
    prior_matches: float,
    recent_window: int = 5,
    recent_prior_matches: float = 2.0,
) -> pd.DataFrame:
    """
    Adds pre-match head-to-head features using only previous matches between the two teams.

    The pair key is unordered, so Bangladesh vs India and India vs Bangladesh share
    the same historical record. The saved win rate is then oriented back toward
    current team1 and current team2.
    """
    df = df.copy().sort_values(["date_parsed", "match_code_num"]).reset_index(drop=True)

    # Stores wins for alphabetically/sorted first team in the pair key.
    pair_wins_for_key0 = defaultdict(float)
    pair_matches = defaultdict(float)
    recent_pair_results = defaultdict(lambda: deque(maxlen=recent_window))

    output = {
        "team1_h2h_win_rate_before_match": [],
        "team2_h2h_win_rate_before_match": [],
        "h2h_win_rate_diff_before_match": [],
        "h2h_matches_played_before": [],
        "h2h_matches_played_log_before": [],
        "h2h_confidence_before": [],
        "h2h_no_history_flag": [],
        "h2h_recent_win_rate_last_5": [],
        "team2_h2h_recent_win_rate_last_5": [],
        "h2h_recent_win_rate_diff_last_5": [],
        "h2h_recent_matches_played_before": [],
        "h2h_pair_key": [],
    }

    for _, row in df.iterrows():
        team1 = row["team1"]
        team2 = row["team2"]
        winner = row["match_winner"]

        pair_key_tuple = tuple(sorted([team1, team2]))
        key0 = pair_key_tuple[0]
        total_previous_pair_matches = pair_matches[pair_key_tuple]
        key0_previous_wins = pair_wins_for_key0[pair_key_tuple]

        if team1 == key0:
            team1_previous_wins = key0_previous_wins
        else:
            team1_previous_wins = total_previous_pair_matches - key0_previous_wins

        # Bayesian smoothing keeps early no-history matches neutral instead of extreme.
        team1_h2h_rate = (
            team1_previous_wins + 0.5 * prior_matches
        ) / (total_previous_pair_matches + prior_matches)
        team2_h2h_rate = 1.0 - team1_h2h_rate

        recent_results = list(recent_pair_results[pair_key_tuple])
        recent_count = len(recent_results)
        key0_recent_wins = sum(recent_results) if recent_count > 0 else 0.0

        if team1 == key0:
            team1_recent_wins = key0_recent_wins
        else:
            team1_recent_wins = recent_count - key0_recent_wins

        team1_recent_rate = (
            team1_recent_wins + 0.5 * recent_prior_matches
        ) / (recent_count + recent_prior_matches)
        team2_recent_rate = 1.0 - team1_recent_rate

        # Save pre-match features first. This is the anti-leakage step.
        output["team1_h2h_win_rate_before_match"].append(float(team1_h2h_rate))
        output["team2_h2h_win_rate_before_match"].append(float(team2_h2h_rate))
        output["h2h_win_rate_diff_before_match"].append(float(team1_h2h_rate - team2_h2h_rate))
        output["h2h_matches_played_before"].append(float(total_previous_pair_matches))
        output["h2h_matches_played_log_before"].append(float(np.log1p(total_previous_pair_matches)))
        output["h2h_confidence_before"].append(float(total_previous_pair_matches / (total_previous_pair_matches + prior_matches)))
        output["h2h_no_history_flag"].append(int(total_previous_pair_matches == 0))
        output["h2h_recent_win_rate_last_5"].append(float(team1_recent_rate))
        output["team2_h2h_recent_win_rate_last_5"].append(float(team2_recent_rate))
        output["h2h_recent_win_rate_diff_last_5"].append(float(team1_recent_rate - team2_recent_rate))
        output["h2h_recent_matches_played_before"].append(float(recent_count))
        output["h2h_pair_key"].append(f"{pair_key_tuple[0]}__vs__{pair_key_tuple[1]}")

        # Update only AFTER saving features for this current row.
        pair_matches[pair_key_tuple] += 1.0
        key0_won_current_match = 1.0 if winner == key0 else 0.0
        pair_wins_for_key0[pair_key_tuple] += key0_won_current_match
        recent_pair_results[pair_key_tuple].append(int(key0_won_current_match))

    for col, values in output.items():
        df[col] = values

    return df

example_h2h_df = add_h2h_history_features(
    clean_df,
    prior_matches=10,
    recent_window=H2H_RECENT_WINDOW,
    recent_prior_matches=H2H_RECENT_PRIOR_MATCHES,
)

example_h2h_df[[
    "date", "team1", "team2", "match_winner",
    "team1_h2h_win_rate_before_match", "team2_h2h_win_rate_before_match",
    "h2h_matches_played_before", "h2h_recent_win_rate_last_5", "team1_won"
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

train_df, val_df, train_val_df, test_df = chronological_split(example_h2h_df)

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
# ===========================================================
# Cell 7: Build XGBoost pipeline and tune H2H settings/config
# ===========================================================

BASE_CATEGORICAL_FEATURES = ["team1", "team2", "venue"]
TARGET_COLUMN = "team1_won"

FEATURE_SET_CANDIDATES = [
    {
        "feature_set_name": "h2h_screenshot_core",
        "numeric_features": [
            "team1_h2h_win_rate_before_match",
            "team2_h2h_win_rate_before_match",
            "h2h_matches_played_before",
            "h2h_recent_win_rate_last_5",
        ],
    },
    {
        "feature_set_name": "h2h_core_with_diff",
        "numeric_features": [
            "team1_h2h_win_rate_before_match",
            "team2_h2h_win_rate_before_match",
            "h2h_win_rate_diff_before_match",
            "h2h_matches_played_before",
            "h2h_recent_win_rate_last_5",
            "h2h_recent_win_rate_diff_last_5",
        ],
    },
    {
        "feature_set_name": "h2h_full_context",
        "numeric_features": [
            "team1_h2h_win_rate_before_match",
            "team2_h2h_win_rate_before_match",
            "h2h_win_rate_diff_before_match",
            "h2h_matches_played_before",
            "h2h_matches_played_log_before",
            "h2h_confidence_before",
            "h2h_no_history_flag",
            "h2h_recent_win_rate_last_5",
            "team2_h2h_recent_win_rate_last_5",
            "h2h_recent_win_rate_diff_last_5",
            "h2h_recent_matches_played_before",
        ],
    },
]

CATEGORICAL_SET_CANDIDATES = [
    {
        "categorical_set_name": "team_venue_only",
        "categorical_features": ["team1", "team2", "venue"],
    },
    {
        "categorical_set_name": "team_venue_plus_h2h_pair_key",
        "categorical_features": ["team1", "team2", "venue", "h2h_pair_key"],
    },
]


def build_xgb_pipeline(model_params: dict, categorical_features: list, numeric_features: list) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", make_one_hot_encoder(), categorical_features),
            ("num", "passthrough", numeric_features),
        ],
        remainder="drop",
    )

    model = XGBClassifier(**model_params)
    pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
    return pipeline


def find_best_threshold(y_true, probabilities):
    """Select a probability threshold using validation data only."""
    candidates = np.unique(np.concatenate([np.round(np.arange(0.35, 0.651, 0.005), 3), np.array([0.50])]))
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


def evaluate_candidate(prior_matches: float, feature_set: dict, categorical_set: dict, model_config: dict) -> dict:
    h2h_df = add_h2h_history_features(
        clean_df,
        prior_matches=prior_matches,
        recent_window=H2H_RECENT_WINDOW,
        recent_prior_matches=H2H_RECENT_PRIOR_MATCHES,
    )
    train_df, val_df, _, _ = chronological_split(h2h_df)

    categorical_features = categorical_set["categorical_features"]
    numeric_features = feature_set["numeric_features"]
    feature_columns = categorical_features + numeric_features

    pipeline = build_xgb_pipeline(model_config["params"], categorical_features, numeric_features)
    pipeline.fit(train_df[feature_columns], train_df[TARGET_COLUMN])

    val_prob = pipeline.predict_proba(val_df[feature_columns])[:, 1]
    val_pred_fixed = (val_prob >= 0.50).astype(int)
    best_threshold, val_acc_at_threshold, val_bal_at_threshold = find_best_threshold(
        val_df[TARGET_COLUMN].values,
        val_prob,
    )

    return {
        "prior_matches": prior_matches,
        "recent_window": H2H_RECENT_WINDOW,
        "feature_set_name": feature_set["feature_set_name"],
        "categorical_set_name": categorical_set["categorical_set_name"],
        "model_config_name": model_config["model_config_name"],
        "validation_accuracy_fixed_0_50": accuracy_score(val_df[TARGET_COLUMN], val_pred_fixed),
        "validation_balanced_accuracy_fixed_0_50": balanced_accuracy_score(val_df[TARGET_COLUMN], val_pred_fixed),
        "validation_roc_auc": roc_auc_score(val_df[TARGET_COLUMN], val_prob),
        "selected_probability_threshold_from_validation": best_threshold,
        "validation_accuracy_at_selected_threshold": val_acc_at_threshold,
        "validation_balanced_accuracy_at_selected_threshold": val_bal_at_threshold,
    }

validation_rows = []
for prior_matches in H2H_PRIOR_MATCHES_CANDIDATES:
    for feature_set in FEATURE_SET_CANDIDATES:
        for categorical_set in CATEGORICAL_SET_CANDIDATES:
            for model_config in MODEL_CONFIGS:
                validation_rows.append(evaluate_candidate(prior_matches, feature_set, categorical_set, model_config))

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

best_prior_matches = float(validation_results.loc[0, "prior_matches"])
best_feature_set_name = validation_results.loc[0, "feature_set_name"]
best_categorical_set_name = validation_results.loc[0, "categorical_set_name"]
best_model_config_name = validation_results.loc[0, "model_config_name"]
best_threshold = float(validation_results.loc[0, "selected_probability_threshold_from_validation"])

best_feature_set = next(item for item in FEATURE_SET_CANDIDATES if item["feature_set_name"] == best_feature_set_name)
best_categorical_set = next(item for item in CATEGORICAL_SET_CANDIDATES if item["categorical_set_name"] == best_categorical_set_name)
best_model_config = next(item for item in MODEL_CONFIGS if item["model_config_name"] == best_model_config_name)

print(f"Best H2H prior_matches selected from validation only: {best_prior_matches}")
print(f"Best H2H feature set selected from validation only: {best_feature_set_name}")
print(f"Best categorical set selected from validation only: {best_categorical_set_name}")
print(f"Best XGBoost config selected from validation only: {best_model_config_name}")
print(f"Best probability threshold selected from validation only: {best_threshold:.6f}")
validation_results.head(10)

# %%
# ================================================
# Cell 8: Train final model and evaluate on test set
# ================================================

final_df = add_h2h_history_features(
    clean_df,
    prior_matches=best_prior_matches,
    recent_window=H2H_RECENT_WINDOW,
    recent_prior_matches=H2H_RECENT_PRIOR_MATCHES,
)
train_df, val_df, train_val_df, test_df = chronological_split(final_df)

CATEGORICAL_FEATURES = best_categorical_set["categorical_features"]
NUMERIC_FEATURES = best_feature_set["numeric_features"]
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES

final_pipeline = build_xgb_pipeline(best_model_config["params"], CATEGORICAL_FEATURES, NUMERIC_FEATURES)
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
print(f"Best H2H prior_matches: {best_prior_matches}")
print(f"Best H2H feature set: {best_feature_set_name}")
print(f"Best categorical set: {best_categorical_set_name}")
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

# Rebuild split summary with final_df, because final H2H prior/settings may differ from preview.
train_df, val_df, train_val_df, test_df = chronological_split(final_df)
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

split_summary.to_csv(OUTPUT_DIR / "split_summary.csv", index=False)
validation_results.to_csv(OUTPUT_DIR / "validation_h2h_model_selection_results.csv", index=False)

metrics_summary = {
    "feature_engineering_used": "Pre-match Head-to-head historical features only",
    "elo_used": False,
    "recent_form_ewma_used": False,
    "bayesian_history_win_rate_features_used": False,
    "opponent_adjusted_historical_strength_used": False,
    "colley_matrix_features_used": False,
    "massey_matrix_features_used": False,
    "best_h2h_prior_matches_selected_from_validation_only": best_prior_matches,
    "h2h_recent_window": H2H_RECENT_WINDOW,
    "h2h_recent_prior_matches": H2H_RECENT_PRIOR_MATCHES,
    "best_h2h_feature_set_selected_from_validation_only": best_feature_set_name,
    "best_categorical_set_selected_from_validation_only": best_categorical_set_name,
    "best_xgboost_config_selected_from_validation_only": best_model_config_name,
    "probability_threshold_selected_from_validation_only": best_threshold,
    "model_selection_metric": "validation_accuracy_at_selected_threshold_then_validation_roc_auc_then_validation_balanced_accuracy_at_selected_threshold",
    "categorical_features_used": CATEGORICAL_FEATURES,
    "numeric_h2h_features_used": NUMERIC_FEATURES,
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
    "team1_h2h_win_rate_before_match", "team2_h2h_win_rate_before_match",
    "h2h_win_rate_diff_before_match", "h2h_matches_played_before",
    "h2h_matches_played_log_before", "h2h_confidence_before", "h2h_no_history_flag",
    "h2h_recent_win_rate_last_5", "team2_h2h_recent_win_rate_last_5",
    "h2h_recent_win_rate_diff_last_5", "h2h_recent_matches_played_before", "h2h_pair_key",
]
test_predictions = test_df[prediction_cols].copy()
test_predictions["predicted_probability_team1_win"] = test_prob
test_predictions["predicted_team1_won"] = test_pred
test_predictions["predicted_winner"] = np.where(test_predictions["predicted_team1_won"] == 1, test_predictions["team1"], test_predictions["team2"])
test_predictions["correct_prediction"] = (test_predictions["predicted_team1_won"] == test_predictions["team1_won"]).astype(int)
test_predictions.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)

final_df.to_csv(OUTPUT_DIR / "cleaned_dataset_with_h2h_history_features.csv", index=False)

# Save ROC curve points and plot.
fpr, tpr, thresholds = roc_curve(test_df[TARGET_COLUMN], test_prob)
roc_points = pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thresholds})
roc_points.to_csv(OUTPUT_DIR / "roc_curve_points.csv", index=False)

plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f"ROC-AUC = {test_roc_auc:.4f}")
plt.plot([0, 1], [0, 1], linestyle="--", label="Random baseline")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - XGBoost Head-to-Head History Model")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "roc_curve.png", dpi=200)
plt.close()

plt.figure(figsize=(5, 4))
plt.imshow(cm)
plt.title("Confusion Matrix - XGBoost Head-to-Head History Model")
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
    plt.title("Top Feature Importances - Head-to-Head History Model")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_importance_top20.png", dpi=200)
    plt.close()
except Exception as e:
    print(f"Feature importance saving skipped due to: {e}")

joblib.dump(final_pipeline, OUTPUT_DIR / "xgboost_h2h_history_pipeline.joblib")

run_summary = f"""
XGBoost Head-to-Head Historical Features Model Summary
======================================================

Feature engineering used:
- Pre-match head-to-head historical features only

Not used:
- Elo ratings
- Recent-form EWMA ratings
- Bayesian historical strength features
- Opponent-adjusted historical strength features
- Colley Matrix features
- Regularized Massey Matrix features
- Toss winner
- team1_bat_first
- Current-match batting/bowling/scorecard columns
- match_winner/match_result as features

Best settings selected from validation only:
- H2H prior_matches: {best_prior_matches}
- H2H recent window: {H2H_RECENT_WINDOW}
- H2H feature set: {best_feature_set_name}
- Categorical feature set: {best_categorical_set_name}
- XGBoost config: {best_model_config_name}
- Probability threshold: {best_threshold:.6f}

Numeric H2H features used:
{NUMERIC_FEATURES}

Categorical features used:
{CATEGORICAL_FEATURES}

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
- Head-to-head values are saved before the current match result is used.
- Chronological split is used instead of random split.
- The test set is not used to fit the model or choose the probability threshold.
- Head-to-head history can be weak for teams that rarely played each other, so this feature family is academically useful but may not outperform formal rating systems like Massey or Colley.
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
