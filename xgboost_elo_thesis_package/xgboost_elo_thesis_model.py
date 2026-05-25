"""
Thesis-safe XGBoost + Elo model for ODI cricket match winner prediction.

Goal:
    Predict whether team1 wins against team2 using only pre-match/raw matchup fields
    plus one feature-engineering group: historical Elo ratings.

Important honesty note:
    This script intentionally does NOT use scorecard columns such as batting runs,
    balls, wickets, overs, economy, dismissals, match_result, or match_outcome_type.
    Those are post-match variables and would create data leakage.

Default prediction setting:
    Strict pre-match mode: team1, team2, venue, and Elo only.
    Toss features are excluded by default because toss is not known before the match.
"""

# =========================
# 1. IMPORTS
# =========================

import os
import re
import json
import warnings
from typing import Dict, Tuple, List

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


# =========================
# 2. CONFIGURATION
# =========================

DATA_PATH = "Dataset-1.csv"       # Change this if your CSV has another name/path
OUTPUT_DIR = "xgb_elo_outputs"

RANDOM_STATE = 42
TEST_SIZE = 0.20                  # Final 20% chronological matches are used as unseen test set

INITIAL_ELO = 1500.0
ELO_K_FACTOR = 24.0

# Keep this False for honest "pre-match before toss" thesis work.
# Set True ONLY if your thesis clearly says prediction is made after toss.
ALLOW_TOSS_FEATURES = False

# This is optional. False = fast and reproducible fixed model.
# True = slower training-only time-series grid search, still no test leakage.
DO_HYPERPARAMETER_SEARCH = False

# Optional threshold tuning using a validation block from the training period only.
# This does not use the test set.
TUNE_DECISION_THRESHOLD = True

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================
# 3. SMALL VERSION-SAFE HELPERS
# =========================

def make_one_hot_encoder() -> OneHotEncoder:
    """
    Creates OneHotEncoder while staying compatible with different sklearn versions.
    New sklearn uses sparse_output; older sklearn uses sparse.
    """
    try:
        return OneHotEncoder(
            handle_unknown="ignore",
            min_frequency=2,
            sparse_output=True,
        )
    except TypeError:
        try:
            return OneHotEncoder(
                handle_unknown="ignore",
                min_frequency=2,
                sparse=True,
            )
        except TypeError:
            return OneHotEncoder(
                handle_unknown="ignore",
                sparse=True,
            )


def clean_text(value) -> str:
    """Basic text cleaning without changing real meaning."""
    if pd.isna(value):
        return np.nan
    value = str(value).replace("\xa0", " ")
    value = re.sub(r"\s+", " ", value).strip()
    return value


def clean_team_name(value) -> str:
    """
    Cleans team names and fixes obvious scraping inconsistency.
    In your dataset, 'West' is used for West Indies in many rows.
    """
    value = clean_text(value)
    if pd.isna(value):
        return np.nan

    team_fixes = {
        "West": "West Indies",
    }
    return team_fixes.get(value, value)


def clean_venue(value) -> str:
    """
    Cleans venue text.
    Your dataset has a repeated broken scraped menu value in the venue column.
    This is converted to 'Unknown' instead of being used as a fake venue.
    """
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


def require_columns(df: pd.DataFrame, required_cols: List[str]) -> None:
    """Stops early if important columns are missing."""
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


# =========================
# 4. LOAD AND CLEAN DATA
# =========================

def load_and_clean_data(path: str) -> pd.DataFrame:
    """
    Loads the dataset and keeps only rows usable for honest winner prediction.

    Removed:
        - missing essential fields
        - ties
        - no results
        - rows where match_winner is not team1 or team2
        - same-team errors
    """
    df = pd.read_csv(path)

    required_cols = ["match_code", "date", "team1", "team2", "venue", "match_winner"]
    require_columns(df, required_cols)

    original_rows = len(df)

    for col in ["team1", "team2", "match_winner", "toss_winner"]:
        if col in df.columns:
            df[col] = df[col].apply(clean_team_name)

    df["venue"] = df["venue"].apply(clean_venue)

    # Dataset date format is day/month/year, e.g., 27/03/2009
    df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y", errors="coerce")

    df = df.dropna(subset=["date", "team1", "team2", "match_winner"])
    df = df[df["team1"] != df["team2"]].copy()

    # Keep only matches with a real winner from the two playing teams.
    valid_winner_mask = (df["match_winner"] == df["team1"]) | (df["match_winner"] == df["team2"])
    df = df[valid_winner_mask].copy()

    # Binary target: 1 means team1 won, 0 means team2 won.
    # This is safer than multiclass winner prediction because each match has only two possible winners.
    df["target_team1_win"] = (df["match_winner"] == df["team1"]).astype(int)

    # Sort chronologically so Elo and train-test split respect time.
    df = df.sort_values(["date", "match_code"]).reset_index(drop=True)

    removed_rows = original_rows - len(df)
    print(f"Original rows: {original_rows}")
    print(f"Usable resolved matches: {len(df)}")
    print(f"Removed rows: {removed_rows}  (ties, no results, missing/invalid winner rows)")

    return df


# =========================
# 5. ELO FEATURE ENGINEERING
# =========================

def add_pre_match_elo_features(
    df: pd.DataFrame,
    initial_elo: float = INITIAL_ELO,
    k_factor: float = ELO_K_FACTOR,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Adds Elo ratings using only information available BEFORE each match.

    For every row:
        1. Read current ratings of team1 and team2.
        2. Store those as pre-match features.
        3. After storing features, update ratings using the actual result.

    This avoids leakage because the current match result is never used to create
    that same row's Elo features.
    """
    ratings: Dict[str, float] = {}

    team1_elo_pre = []
    team2_elo_pre = []
    elo_diff = []

    for _, row in df.iterrows():
        team1 = row["team1"]
        team2 = row["team2"]

        r1 = ratings.get(team1, initial_elo)
        r2 = ratings.get(team2, initial_elo)

        # Save pre-match ratings first.
        team1_elo_pre.append(r1)
        team2_elo_pre.append(r2)
        elo_diff.append(r1 - r2)

        # Now update ratings after the match.
        expected_team1 = 1.0 / (1.0 + 10.0 ** ((r2 - r1) / 400.0))
        actual_team1 = float(row["target_team1_win"])

        ratings[team1] = r1 + k_factor * (actual_team1 - expected_team1)
        ratings[team2] = r2 + k_factor * ((1.0 - actual_team1) - (1.0 - expected_team1))

    df = df.copy()
    df["team1_elo_pre"] = team1_elo_pre
    df["team2_elo_pre"] = team2_elo_pre
    df["elo_diff"] = elo_diff

    return df, ratings


# =========================
# 6. FEATURE SELECTION
# =========================

def get_feature_columns(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    """
    Uses only safe pre-match columns plus Elo features.

    Used by default:
        - team1
        - team2
        - venue
        - team1_elo_pre
        - team2_elo_pre
        - elo_diff

    Excluded deliberately:
        - match_code: unique-ish ID, not a real cricket predictor
        - date: used for ordering/Elo only, not used as a model feature
        - series: too high-cardinality in this dataset and can behave like a match identifier
        - toss_winner: excluded unless ALLOW_TOSS_FEATURES=True
        - all batting/bowling scorecard columns: post-match leakage
    """
    categorical_features = ["team1", "team2", "venue"]

    if "match_type" in df.columns and df["match_type"].nunique(dropna=True) > 1:
        categorical_features.append("match_type")

    if ALLOW_TOSS_FEATURES and "toss_winner" in df.columns:
        categorical_features.append("toss_winner")

    numeric_features = ["team1_elo_pre", "team2_elo_pre", "elo_diff"]

    feature_cols = categorical_features + numeric_features
    return feature_cols, categorical_features, numeric_features


# =========================
# 7. CHRONOLOGICAL SPLIT
# =========================

def chronological_train_test_split(df: pd.DataFrame, test_size: float = TEST_SIZE):
    """
    Splits by time, not random shuffle.

    Sports data is time-dependent. Random train-test split allows future matches
    to help train a model that predicts older matches, which is not honest.
    """
    split_index = int(len(df) * (1.0 - test_size))

    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()

    print("\nChronological split:")
    print(f"Train rows: {len(train_df)} | {train_df['date'].min().date()} to {train_df['date'].max().date()}")
    print(f"Test rows : {len(test_df)} | {test_df['date'].min().date()} to {test_df['date'].max().date()}")

    return train_df, test_df


# =========================
# 8. MODEL BUILDING
# =========================

def build_preprocessor(categorical_features: List[str], numeric_features: List[str]) -> ColumnTransformer:
    """Preprocessing only: encoding and missing-value handling."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", make_one_hot_encoder(), categorical_features),
            ("numeric", SimpleImputer(strategy="median"), numeric_features),
        ],
        remainder="drop",
    )
    return preprocessor


def build_fixed_xgb_model() -> XGBClassifier:
    """
    Conservative XGBoost model for small tabular sports dataset.

    This is plain existing XGBoost, not a custom/deep model.
    """
    return XGBClassifier(
        n_estimators=180,
        max_depth=1,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=6,
        reg_lambda=15,
        reg_alpha=1,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=1,
        tree_method="hist",
    )


def build_pipeline(categorical_features: List[str], numeric_features: List[str]) -> Pipeline:
    """Full sklearn pipeline: preprocessing + XGBoost."""
    preprocessor = build_preprocessor(categorical_features, numeric_features)
    model = build_fixed_xgb_model()

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )
    return pipeline


def run_training_only_grid_search(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Pipeline:
    """
    Optional time-series hyperparameter search using training data only.

    The final test set is not touched during tuning.
    """
    param_grid = {
        "model__n_estimators": [80, 120, 180],
        "model__max_depth": [1, 2],
        "model__learning_rate": [0.04, 0.08],
        "model__subsample": [0.85, 0.95],
        "model__colsample_bytree": [0.85, 0.95],
        "model__min_child_weight": [4, 6, 8],
        "model__reg_lambda": [10, 15, 20],
        "model__reg_alpha": [0, 1, 2],
    }

    tscv = TimeSeriesSplit(n_splits=4)

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="accuracy",
        cv=tscv,
        n_jobs=-1,
        verbose=1,
    )

    search.fit(X_train, y_train)

    print("\nBest CV accuracy:", round(search.best_score_ * 100, 2), "%")
    print("Best parameters:")
    print(json.dumps(search.best_params_, indent=4))

    return search.best_estimator_


# =========================
# 9. THRESHOLD TUNING
# =========================

def find_best_threshold_from_training_validation(
    pipeline: Pipeline,
    train_df: pd.DataFrame,
    feature_cols: List[str],
) -> float:
    """
    Finds a probability threshold using only the training period.

    Default 0.50 is standard, but sometimes slightly changing it improves accuracy.
    This function uses the last 20% of the training period as validation.
    The final test set remains unseen.
    """
    validation_start = int(len(train_df) * 0.80)

    inner_train = train_df.iloc[:validation_start].copy()
    validation = train_df.iloc[validation_start:].copy()

    X_inner = inner_train[feature_cols]
    y_inner = inner_train["target_team1_win"]

    X_val = validation[feature_cols]
    y_val = validation["target_team1_win"]

    pipeline.fit(X_inner, y_inner)
    val_proba = pipeline.predict_proba(X_val)[:, 1]

    thresholds = np.arange(0.35, 0.66, 0.01)

    best_threshold = 0.50
    best_accuracy = -1.0

    for threshold in thresholds:
        val_pred = (val_proba >= threshold).astype(int)
        val_accuracy = accuracy_score(y_val, val_pred)

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_threshold = float(threshold)

    print(f"\nBest validation threshold from training period: {best_threshold:.2f}")
    print(f"Validation accuracy at this threshold: {best_accuracy * 100:.2f}%")

    return best_threshold


# =========================
# 10. EVALUATION
# =========================

def evaluate_model(
    pipeline: Pipeline,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    threshold: float = 0.50,
) -> pd.DataFrame:
    """Evaluates model honestly on chronological unseen test set."""
    X_train = train_df[feature_cols]
    y_train = train_df["target_team1_win"]

    X_test = test_df[feature_cols]
    y_test = test_df["target_team1_win"]

    train_proba = pipeline.predict_proba(X_train)[:, 1]
    test_proba = pipeline.predict_proba(X_test)[:, 1]

    train_pred = (train_proba >= threshold).astype(int)
    test_pred = (test_proba >= threshold).astype(int)

    train_accuracy = accuracy_score(y_train, train_pred)
    test_accuracy = accuracy_score(y_test, test_pred)
    test_balanced_accuracy = balanced_accuracy_score(y_test, test_pred)
    test_auc = roc_auc_score(y_test, test_proba)

    print("\n================ FINAL RESULTS ================")
    print(f"Decision threshold        : {threshold:.2f}")
    print(f"Training accuracy         : {train_accuracy * 100:.2f}%")
    print(f"Testing accuracy          : {test_accuracy * 100:.2f}%")
    print(f"Testing balanced accuracy : {test_balanced_accuracy * 100:.2f}%")
    print(f"Testing ROC-AUC           : {test_auc:.4f}")

    print("\nConfusion Matrix [rows=true, columns=predicted]")
    print("Labels: 0 = team2_win, 1 = team1_win")
    cm = confusion_matrix(y_test, test_pred)
    print(cm)

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            test_pred,
            target_names=["team2_win", "team1_win"],
            digits=4,
        )
    )

    # Save metrics
    metrics = {
        "threshold": threshold,
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "test_balanced_accuracy": test_balanced_accuracy,
        "test_roc_auc": test_auc,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "test_start_date": str(test_df["date"].min().date()),
        "test_end_date": str(test_df["date"].max().date()),
    }

    with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    # Save confusion matrix
    cm_df = pd.DataFrame(
        cm,
        index=["actual_team2_win", "actual_team1_win"],
        columns=["pred_team2_win", "pred_team1_win"],
    )
    cm_df.to_csv(os.path.join(OUTPUT_DIR, "confusion_matrix.csv"), index=True)

    plt.figure(figsize=(5, 4))
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.xticks([0, 1], ["team2_win", "team1_win"])
    plt.yticks([0, 1], ["team2_win", "team1_win"])

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=300)
    plt.close()

    # Save per-match predictions
    prediction_df = test_df[
        ["match_code", "date", "team1", "team2", "venue", "match_winner", "target_team1_win"]
    ].copy()

    prediction_df["predicted_team1_win_probability"] = test_proba
    prediction_df["predicted_team1_win"] = test_pred
    prediction_df["predicted_winner"] = np.where(
        prediction_df["predicted_team1_win"] == 1,
        prediction_df["team1"],
        prediction_df["team2"],
    )
    prediction_df["correct_prediction"] = prediction_df["predicted_winner"] == prediction_df["match_winner"]

    prediction_df.to_csv(os.path.join(OUTPUT_DIR, "test_predictions.csv"), index=False)

    return prediction_df


# =========================
# 11. FEATURE IMPORTANCE
# =========================

def save_feature_importance(pipeline: Pipeline) -> None:
    """Saves XGBoost feature importance after preprocessing."""
    preprocessor = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]

    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        feature_names = [f"feature_{i}" for i in range(model.feature_importances_.shape[0])]

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    importance_df.to_csv(os.path.join(OUTPUT_DIR, "feature_importance.csv"), index=False)

    top_n = min(20, len(importance_df))
    top_features = importance_df.head(top_n).iloc[::-1]

    plt.figure(figsize=(9, 6))
    plt.barh(top_features["feature"], top_features["importance"])
    plt.title("Top Feature Importances")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance_top20.png"), dpi=300)
    plt.close()


# =========================
# 12. SAVE FINAL MODEL
# =========================

def save_artifacts(
    pipeline: Pipeline,
    model_df: pd.DataFrame,
    final_elo_ratings: Dict[str, float],
    feature_cols: List[str],
    threshold: float,
) -> None:
    """Saves model, cleaned Elo dataset, ratings, and config."""
    model_path = os.path.join(OUTPUT_DIR, "xgboost_elo_pipeline.joblib")
    joblib.dump(pipeline, model_path)

    model_df.to_csv(os.path.join(OUTPUT_DIR, "cleaned_dataset_with_elo.csv"), index=False)

    ratings_df = (
        pd.DataFrame(
            [{"team": team, "latest_elo": rating} for team, rating in final_elo_ratings.items()]
        )
        .sort_values("latest_elo", ascending=False)
        .reset_index(drop=True)
    )
    ratings_df.to_csv(os.path.join(OUTPUT_DIR, "latest_elo_ratings.csv"), index=False)

    config = {
        "feature_columns": feature_cols,
        "categorical_features_used": [col for col in feature_cols if model_df[col].dtype == "object"],
        "elo_features_used": ["team1_elo_pre", "team2_elo_pre", "elo_diff"],
        "target": "target_team1_win",
        "initial_elo": INITIAL_ELO,
        "elo_k_factor": ELO_K_FACTOR,
        "allow_toss_features": ALLOW_TOSS_FEATURES,
        "decision_threshold": threshold,
        "note": "Only Elo ratings are engineered. Scorecard/post-match columns are not used.",
    }

    with open(os.path.join(OUTPUT_DIR, "model_config.json"), "w") as f:
        json.dump(config, f, indent=4)

    print(f"\nSaved model and outputs inside: {OUTPUT_DIR}/")


# =========================
# 13. OPTIONAL SINGLE MATCH PREDICTION
# =========================

def predict_single_match(
    pipeline: Pipeline,
    latest_elo_ratings: Dict[str, float],
    team1: str,
    team2: str,
    venue: str,
    threshold: float,
) -> Dict[str, object]:
    """
    Predicts a new match using latest Elo ratings.

    Use this only after the model has been trained.
    """
    team1 = clean_team_name(team1)
    team2 = clean_team_name(team2)
    venue = clean_venue(venue)

    r1 = latest_elo_ratings.get(team1, INITIAL_ELO)
    r2 = latest_elo_ratings.get(team2, INITIAL_ELO)

    row = pd.DataFrame(
        [
            {
                "team1": team1,
                "team2": team2,
                "venue": venue,
                "team1_elo_pre": r1,
                "team2_elo_pre": r2,
                "elo_diff": r1 - r2,
            }
        ]
    )

    if ALLOW_TOSS_FEATURES:
        raise ValueError("predict_single_match needs toss_winner added if ALLOW_TOSS_FEATURES=True.")

    probability_team1_win = float(pipeline.predict_proba(row)[:, 1][0])
    predicted_team1_win = int(probability_team1_win >= threshold)
    predicted_winner = team1 if predicted_team1_win == 1 else team2

    return {
        "team1": team1,
        "team2": team2,
        "venue": venue,
        "team1_elo_pre": r1,
        "team2_elo_pre": r2,
        "elo_diff": r1 - r2,
        "probability_team1_win": probability_team1_win,
        "decision_threshold": threshold,
        "predicted_winner": predicted_winner,
    }


# =========================
# 14. MAIN SCRIPT
# =========================

def main():
    # Load, clean, and create honest binary target
    clean_df = load_and_clean_data(DATA_PATH)

    # Add the only engineered feature group: Elo ratings
    model_df, final_elo_ratings = add_pre_match_elo_features(
        clean_df,
        initial_elo=INITIAL_ELO,
        k_factor=ELO_K_FACTOR,
    )

    # Feature columns
    feature_cols, categorical_features, numeric_features = get_feature_columns(model_df)

    print("\nFeatures used:")
    for col in feature_cols:
        print(f"  - {col}")

    print("\nTarget used:")
    print("  - target_team1_win (1 = team1 won, 0 = team2 won)")

    # Save the prepared dataset before splitting
    model_df.to_csv(os.path.join(OUTPUT_DIR, "prepared_model_dataset.csv"), index=False)

    # Chronological train-test split
    train_df, test_df = chronological_train_test_split(model_df, test_size=TEST_SIZE)

    X_train = train_df[feature_cols]
    y_train = train_df["target_team1_win"]

    # Build model
    pipeline = build_pipeline(categorical_features, numeric_features)

    # Optional tuning on training data only
    if DO_HYPERPARAMETER_SEARCH:
        pipeline = run_training_only_grid_search(pipeline, X_train, y_train)

    # Optional validation threshold search using training period only
    if TUNE_DECISION_THRESHOLD:
        threshold = find_best_threshold_from_training_validation(
            pipeline=build_pipeline(categorical_features, numeric_features),
            train_df=train_df,
            feature_cols=feature_cols,
        )
    else:
        threshold = 0.50

    # Fit final model on the full training period only
    pipeline.fit(X_train, y_train)

    # Evaluate on unseen chronological test set
    prediction_df = evaluate_model(
        pipeline=pipeline,
        train_df=train_df,
        test_df=test_df,
        feature_cols=feature_cols,
        threshold=threshold,
    )

    # Save importance and artifacts
    save_feature_importance(pipeline)
    save_artifacts(
        pipeline=pipeline,
        model_df=model_df,
        final_elo_ratings=final_elo_ratings,
        feature_cols=feature_cols,
        threshold=threshold,
    )

    print("\nSample predictions:")
    print(prediction_df.head(10)[
        ["date", "team1", "team2", "match_winner", "predicted_winner", "correct_prediction"]
    ])

    # Example of predicting a future/single match:
    # example = predict_single_match(
    #     pipeline=pipeline,
    #     latest_elo_ratings=final_elo_ratings,
    #     team1="India",
    #     team2="Australia",
    #     venue="Dubai International Cricket Stadium",
    #     threshold=threshold,
    # )
    # print(example)


if __name__ == "__main__":
    main()
