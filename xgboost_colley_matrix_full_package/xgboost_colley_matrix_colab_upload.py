# %%
# ===============================================================
# Thesis-safe Cricket Match Winner Prediction
# Plain XGBoost + ONE feature engineering family:
# Pre-match Colley Matrix Team Strength
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

# Candidate Colley prior values. These are selected using validation data only.
# The test set is not used to pick the final value.
COLLEY_PRIOR_SCALE_CANDIDATES = [0.5, 1, 2, 4, 6, 8, 10, 16, 20, 30, 40, 60]

# Small conservative XGBoost configuration list.
# This is not extra feature engineering. It is normal validation-based model selection.
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
OUTPUT_DIR = Path(DATA_PATH).resolve().parent / "xgb_colley_matrix_outputs"
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
        "West": "West Indies",   # dataset scraping inconsistency
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

    # Keep only resolved matches where winner is exactly one of the two teams.
    resolved_mask = (df["match_winner"] == df["team1"]) | (df["match_winner"] == df["team2"])
    df = df[resolved_mask].copy()

    if "match_code" in df.columns:
        df["match_code_num"] = pd.to_numeric(df["match_code"], errors="coerce")
        df["match_code_num"] = df["match_code_num"].fillna(pd.Series(np.arange(len(df)), index=df.index))
    else:
        df["match_code"] = np.arange(len(df))
        df["match_code_num"] = np.arange(len(df))

    df = df.sort_values(["date_parsed", "match_code_num"]).reset_index(drop=True)

    # Binary target: 1 = team1 won, 0 = team2 won.
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
# Cell 5: ONE feature engineering only: Colley Matrix strength rating
# ==================================================================

# This creates one feature-engineering family only:
# pre-match Colley Matrix team-strength features.
#
# Colley rating is a formal sports ranking method. It estimates team strength
# from previous wins/losses and strength of schedule through a linear system:
#
#     C r = b
#
# For each match:
# 1. Read both teams' Colley ratings BEFORE the current match.
# 2. Save those pre-match values as features.
# 3. Update the win/loss/opponent-count history AFTER the current match.
# 4. Recompute ratings for the next row only.
#
# Created columns:
# - team1_colley_strength_pre
# - team2_colley_strength_pre
# - colley_strength_diff
# - team1_colley_matches_pre
# - team2_colley_matches_pre
# - colley_matches_diff
#
# No Elo, no recent-form EWMA, no Bayesian historical win-rate features,
# and no opponent-adjusted handcrafted quality features are used.
# No current-match scorecard/post-match performance column is used.


def compute_colley_ratings(active_teams, wins, losses, pair_counts, prior_scale: float) -> dict:
    """
    Computes Colley Matrix ratings for teams using history available so far.

    Standard Colley idea:
        C_ii = prior_scale + games_i
        C_ij = - games_between_i_and_j
        b_i  = prior_scale/2 + (wins_i - losses_i)/2

    A neutral team with no history receives 0.5.
    """
    teams = list(active_teams)
    n_teams = len(teams)
    if n_teams == 0:
        return {}

    index = {team: i for i, team in enumerate(teams)}
    C = np.eye(n_teams) * prior_scale
    b = np.ones(n_teams) * (prior_scale / 2.0)

    for team in teams:
        i = index[team]
        games = wins[team] + losses[team]
        C[i, i] += games
        b[i] += (wins[team] - losses[team]) / 2.0

    for (team_a, team_b), count in pair_counts.items():
        if team_a in index and team_b in index:
            i = index[team_a]
            j = index[team_b]
            C[i, j] -= count

    try:
        ratings = np.linalg.solve(C, b)
    except np.linalg.LinAlgError:
        ratings = np.linalg.lstsq(C, b, rcond=None)[0]

    return {team: float(ratings[index[team]]) for team in teams}


def add_colley_strength_features(df: pd.DataFrame, prior_scale: float) -> pd.DataFrame:
    """
    Adds pre-match Colley Matrix strength features using only past results.

    Anti-leakage rule:
    Features for each row are saved BEFORE updating history with that row's result.
    """
    df = df.copy().sort_values(["date_parsed", "match_code_num"]).reset_index(drop=True)

    wins = defaultdict(float)
    losses = defaultdict(float)
    matches = defaultdict(float)
    pair_counts = defaultdict(float)
    active_teams = set()
    current_ratings = {}

    team1_strength_values = []
    team2_strength_values = []
    strength_diff_values = []
    team1_match_count_values = []
    team2_match_count_values = []
    match_count_diff_values = []

    for _, row in df.iterrows():
        team1 = row["team1"]
        team2 = row["team2"]
        winner = row["match_winner"]

        # New teams are neutral before they have any historical information.
        team1_strength_pre = current_ratings.get(team1, 0.50)
        team2_strength_pre = current_ratings.get(team2, 0.50)
        team1_matches_pre = float(matches[team1])
        team2_matches_pre = float(matches[team2])

        # Save pre-match values first. This is the anti-leakage step.
        team1_strength_values.append(float(team1_strength_pre))
        team2_strength_values.append(float(team2_strength_pre))
        strength_diff_values.append(float(team1_strength_pre - team2_strength_pre))
        team1_match_count_values.append(team1_matches_pre)
        team2_match_count_values.append(team2_matches_pre)
        match_count_diff_values.append(float(team1_matches_pre - team2_matches_pre))

        # Update only AFTER saving the features for this row.
        active_teams.update([team1, team2])

        if winner == team1:
            wins[team1] += 1.0
            losses[team2] += 1.0
        else:
            wins[team2] += 1.0
            losses[team1] += 1.0

        matches[team1] += 1.0
        matches[team2] += 1.0
        pair_counts[(team1, team2)] += 1.0
        pair_counts[(team2, team1)] += 1.0

        current_ratings = compute_colley_ratings(
            active_teams=active_teams,
            wins=wins,
            losses=losses,
            pair_counts=pair_counts,
            prior_scale=prior_scale,
        )

    df["team1_colley_strength_pre"] = team1_strength_values
    df["team2_colley_strength_pre"] = team2_strength_values
    df["colley_strength_diff"] = strength_diff_values
    df["team1_colley_matches_pre"] = team1_match_count_values
    df["team2_colley_matches_pre"] = team2_match_count_values
    df["colley_matches_diff"] = match_count_diff_values

    return df

example_colley_df = add_colley_strength_features(clean_df, prior_scale=4)
example_colley_df[[
    "date", "team1", "team2", "match_winner",
    "team1_colley_strength_pre", "team2_colley_strength_pre",
    "colley_strength_diff", "team1_colley_matches_pre", "team2_colley_matches_pre",
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

train_df, val_df, train_val_df, test_df = chronological_split(example_colley_df)

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
# ==========================================================
# Cell 7: Build XGBoost pipeline and tune Colley prior/config
# ==========================================================

CATEGORICAL_FEATURES = ["team1", "team2", "venue"]
NUMERIC_FEATURES = [
    "team1_colley_strength_pre",
    "team2_colley_strength_pre",
    "colley_strength_diff",
    "team1_colley_matches_pre",
    "team2_colley_matches_pre",
    "colley_matches_diff",
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

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    return pipeline


def evaluate_candidate(prior_scale: float, model_config: dict) -> dict:
    colley_df = add_colley_strength_features(clean_df, prior_scale=prior_scale)
    train_df, val_df, _, _ = chronological_split(colley_df)

    pipeline = build_xgb_pipeline(model_config["params"])
    pipeline.fit(train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN])

    val_prob = pipeline.predict_proba(val_df[FEATURE_COLUMNS])[:, 1]
    val_pred = (val_prob >= 0.50).astype(int)

    return {
        "prior_scale": prior_scale,
        "model_config_name": model_config["model_config_name"],
        "validation_accuracy": accuracy_score(val_df[TARGET_COLUMN], val_pred),
        "validation_balanced_accuracy": balanced_accuracy_score(val_df[TARGET_COLUMN], val_pred),
        "validation_roc_auc": roc_auc_score(val_df[TARGET_COLUMN], val_prob),
    }

validation_rows = []
for prior_scale in COLLEY_PRIOR_SCALE_CANDIDATES:
    for model_config in MODEL_CONFIGS:
        validation_rows.append(evaluate_candidate(prior_scale, model_config))

validation_results = pd.DataFrame(validation_rows)

# Since the user goal is highest honest winner-prediction accuracy, selection uses
# validation accuracy first. Test data is untouched here.
validation_results = validation_results.sort_values(
    ["validation_accuracy", "validation_balanced_accuracy", "validation_roc_auc"],
    ascending=False,
).reset_index(drop=True)

best_prior_scale = float(validation_results.loc[0, "prior_scale"])
best_model_config_name = validation_results.loc[0, "model_config_name"]
best_model_config = next(config for config in MODEL_CONFIGS if config["model_config_name"] == best_model_config_name)

print(f"Best Colley prior_scale selected from validation only: {best_prior_scale}")
print(f"Best XGBoost config selected from validation only: {best_model_config_name}")
validation_results

# %%
# ================================================
# Cell 8: Train final model and evaluate on test set
# ================================================

final_df = add_colley_strength_features(clean_df, prior_scale=best_prior_scale)
train_df, val_df, train_val_df, test_df = chronological_split(final_df)

final_pipeline = build_xgb_pipeline(best_model_config["params"])
final_pipeline.fit(train_val_df[FEATURE_COLUMNS], train_val_df[TARGET_COLUMN])

train_val_prob = final_pipeline.predict_proba(train_val_df[FEATURE_COLUMNS])[:, 1]
train_val_pred = (train_val_prob >= 0.50).astype(int)

test_prob = final_pipeline.predict_proba(test_df[FEATURE_COLUMNS])[:, 1]
test_pred = (test_prob >= 0.50).astype(int)

train_accuracy = accuracy_score(train_val_df[TARGET_COLUMN], train_val_pred)
test_accuracy = accuracy_score(test_df[TARGET_COLUMN], test_pred)
test_balanced_accuracy = balanced_accuracy_score(test_df[TARGET_COLUMN], test_pred)
test_roc_auc = roc_auc_score(test_df[TARGET_COLUMN], test_prob)

print("Final Results")
print("-------------")
print(f"Best Colley prior_scale: {best_prior_scale}")
print(f"Best XGBoost config: {best_model_config_name}")
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

# Save split summary and validation results.
split_summary.to_csv(OUTPUT_DIR / "split_summary.csv", index=False)
validation_results.to_csv(OUTPUT_DIR / "validation_colley_model_selection_results.csv", index=False)

metrics_summary = {
    "feature_engineering_used": "Pre-match Colley Matrix team-strength only",
    "elo_used": False,
    "recent_form_used": False,
    "bayesian_history_win_rate_features_used": False,
    "opponent_adjusted_historical_strength_used": False,
    "best_colley_prior_scale_selected_from_validation_only": best_prior_scale,
    "best_xgboost_config_selected_from_validation_only": best_model_config_name,
    "model_selection_metric": "validation_accuracy_then_validation_balanced_accuracy_then_validation_roc_auc",
    "rows_used_after_cleaning": int(len(final_df)),
    "train_validation_rows": int(len(train_val_df)),
    "test_rows": int(len(test_df)),
    "train_validation_accuracy": float(train_accuracy),
    "test_accuracy": float(test_accuracy),
    "test_balanced_accuracy": float(test_balanced_accuracy),
    "test_roc_auc": float(test_roc_auc),
    "features_used": FEATURE_COLUMNS,
    "excluded_for_leakage_reason": (
        "All current-match scorecard/player performance columns, toss_winner, team1_bat_first, "
        "match_result, match_outcome_type, and match_winner as feature"
    ),
    "cleaning_notes": [
        "West is standardised to West Indies",
        "broken scraped venue menu text is replaced with Unknown",
        "only resolved matches where winner equals team1 or team2 are kept",
        "chronological train/validation/test split is used",
    ],
    "anti_leakage_notes": [
        "Colley Matrix strength features are written before updating with the current match result",
        "No current-match scorecard field is used as a model feature",
        "The Colley prior and XGBoost config are selected using validation data only, not the test set",
        "Later test matches may use earlier test-period results because those results would be known before the later match date in a chronological pre-match simulation",
    ],
}
with open(OUTPUT_DIR / "metrics_summary.json", "w") as f:
    json.dump(metrics_summary, f, indent=4)

report_dict = classification_report(
    test_df[TARGET_COLUMN],
    test_pred,
    target_names=["team2_won", "team1_won"],
    output_dict=True,
)
pd.DataFrame(report_dict).transpose().to_csv(OUTPUT_DIR / "classification_report.csv")

cm = confusion_matrix(test_df[TARGET_COLUMN], test_pred)
pd.DataFrame(
    cm,
    index=["actual_team2_won", "actual_team1_won"],
    columns=["predicted_team2_won", "predicted_team1_won"],
).to_csv(OUTPUT_DIR / "confusion_matrix.csv")

predictions_df = test_df[[
    "match_code", "date", "team1", "team2", "venue", "match_winner",
    "team1_colley_strength_pre", "team2_colley_strength_pre",
    "colley_strength_diff", "team1_colley_matches_pre", "team2_colley_matches_pre",
    "colley_matches_diff", "team1_won"
]].copy()
predictions_df["predicted_probability_team1_win"] = test_prob
predictions_df["predicted_team1_won"] = test_pred
predictions_df["predicted_winner"] = np.where(test_pred == 1, predictions_df["team1"], predictions_df["team2"])
predictions_df["correct_prediction"] = predictions_df["predicted_winner"] == predictions_df["match_winner"]
predictions_df.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)

final_df.to_csv(OUTPUT_DIR / "cleaned_dataset_with_colley_matrix_features.csv", index=False)

joblib.dump(final_pipeline, OUTPUT_DIR / "xgboost_colley_matrix_pipeline.joblib")

print(f"Saved result files to: {OUTPUT_DIR}")
print(sorted([p.name for p in OUTPUT_DIR.iterdir()]))

# %%
# ======================================
# Cell 10: Visual outputs
# ======================================

# Confusion matrix plot
fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(cm)
ax.set_title("Confusion Matrix - XGBoost Colley Matrix Model")
ax.set_xlabel("Predicted label")
ax.set_ylabel("True label")
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(["team2_won", "team1_won"], rotation=25, ha="right")
ax.set_yticklabels(["team2_won", "team1_won"])
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=200, bbox_inches="tight")
plt.show()

# ROC curve
fpr, tpr, thresholds = roc_curve(test_df[TARGET_COLUMN], test_prob)
roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thresholds})
roc_df.to_csv(OUTPUT_DIR / "roc_curve_points.csv", index=False)

fig, ax = plt.subplots(figsize=(5, 4))
ax.plot(fpr, tpr, label=f"ROC-AUC = {test_roc_auc:.3f}")
ax.plot([0, 1], [0, 1], linestyle="--")
ax.set_title("ROC Curve - XGBoost Colley Matrix Model")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "roc_curve.png", dpi=200, bbox_inches="tight")
plt.show()

# Feature importance
preprocessor = final_pipeline.named_steps["preprocess"]
model = final_pipeline.named_steps["model"]

try:
    cat_feature_names = list(
        preprocessor.named_transformers_["cat"].get_feature_names_out(CATEGORICAL_FEATURES)
    )
except Exception:
    cat_feature_names = [f"encoded_cat_{i}" for i in range(model.feature_importances_.shape[0] - len(NUMERIC_FEATURES))]

all_feature_names = cat_feature_names + NUMERIC_FEATURES
importances = model.feature_importances_

feature_importance_df = pd.DataFrame({
    "feature": all_feature_names,
    "importance": importances,
}).sort_values("importance", ascending=False)
feature_importance_df.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)

top_features = feature_importance_df.head(20).sort_values("importance", ascending=True)
fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(top_features["feature"], top_features["importance"])
ax.set_title("Top 20 Feature Importances")
ax.set_xlabel("Importance")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "feature_importance_top20.png", dpi=200, bbox_inches="tight")
plt.show()

print("Top 20 feature importances:")
feature_importance_df.head(20)

# %%
# ======================================
# Cell 11: Write final run summary and zip outputs
# ======================================

summary_text = f"""
XGBoost Colley Matrix Team-Strength Model Summary
=================================================

Model type:
- Plain existing XGBoost binary classifier
- Target: team1_won, where 1 means team1 won and 0 means team2 won

Single feature-engineering family used:
- Pre-match Colley Matrix team-strength only

Engineered features:
- team1_colley_strength_pre
- team2_colley_strength_pre
- colley_strength_diff
- team1_colley_matches_pre
- team2_colley_matches_pre
- colley_matches_diff

Not used:
- Elo ratings
- Recent-form EWMA ratings
- Bayesian historical win-rate features
- Opponent-adjusted historical quality-strength features from the previous version
- batting scorecard columns from the current match
- bowling scorecard columns from the current match
- player scorecard columns from the current match
- match_result
- match_outcome_type
- match_winner as a feature
- toss_winner
- team1_bat_first

Cleaning performed:
- Team names cleaned
- 'West' standardised to 'West Indies'
- Broken scraped venue menu text converted to 'Unknown'
- Date parsed with day-first format
- Only resolved matches where winner is team1 or team2 are kept
- Same-team errors removed

Anti-leakage logic:
- Features for each match are saved before updating that match's result
- Chronological train/validation/test split is used
- Colley prior and XGBoost config are selected using validation data only
- The test set is not used to select settings
- Later test-period rows can use earlier test-period match results because this is chronological pre-match simulation

Dataset:
- Clean/resolved rows used: {len(final_df)}
- Train + validation rows: {len(train_val_df)}
- Test rows: {len(test_df)}
- Test period: {test_df['date_parsed'].min().date()} to {test_df['date_parsed'].max().date()}

Selected settings:
- best_colley_prior_scale: {best_prior_scale}
- best_xgboost_config: {best_model_config_name}
- selection metric order: validation accuracy, then validation balanced accuracy, then validation ROC-AUC

Final results:
- Train + validation accuracy: {train_accuracy:.4f} ({train_accuracy * 100:.2f}%)
- Test accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)
- Test balanced accuracy: {test_balanced_accuracy:.4f} ({test_balanced_accuracy * 100:.2f}%)
- Test ROC-AUC: {test_roc_auc:.4f}

Important thesis warning:
- Your dataset has team1_bat_first = 1 for every row, so team1/team2 may still reflect batting order.
- This code excludes team1_bat_first and toss_winner.
- For strict before-toss prediction, team ordering should ideally be randomized or made independent of innings order.
""".strip()

with open(OUTPUT_DIR / "run_summary.txt", "w") as f:
    f.write(summary_text)

zip_path = OUTPUT_DIR.with_suffix(".zip")
if zip_path.exists():
    zip_path.unlink()
with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for file_path in OUTPUT_DIR.rglob("*"):
        if file_path.is_file():
            zf.write(file_path, arcname=file_path.relative_to(OUTPUT_DIR.parent))

print(summary_text)
print(f"\nZipped outputs saved to: {zip_path}")

# %%
# ======================================
# Cell 12: Small prediction helper for manual checking
# ======================================

# This helper predicts one existing row from final_df.
# It is only for checking how the trained pipeline behaves on an already-featured row.
# For future real matches, you must calculate the Colley features using matches before that future date.

def predict_existing_row(row_index: int):
    sample = final_df.iloc[[row_index]][FEATURE_COLUMNS]
    prob_team1 = final_pipeline.predict_proba(sample)[:, 1][0]
    pred_team1_won = int(prob_team1 >= 0.50)
    row = final_df.iloc[row_index]
    predicted_winner = row["team1"] if pred_team1_won == 1 else row["team2"]
    return {
        "date": row["date"],
        "team1": row["team1"],
        "team2": row["team2"],
        "actual_winner": row["match_winner"],
        "predicted_probability_team1_win": float(prob_team1),
        "predicted_winner": predicted_winner,
    }

predict_existing_row(len(final_df) - 1)
