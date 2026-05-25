# %%
# ================================
# Cell 1: Imports and basic setup
# ================================

import os
import json
import zipfile
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

try:
    from xgboost import XGBClassifier
except ImportError as exc:
    raise ImportError(
        "xgboost is not installed. In Colab, run: !pip install xgboost"
    ) from exc

try:
    import joblib
except ImportError:
    joblib = None

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 200)

RANDOM_STATE = 42
TEST_SIZE = 0.20
VALIDATION_SIZE_WITHIN_TRAIN = 0.20

# Only one feature-engineering family is used: EWMA recent form.
# Higher alpha means the model gives more importance to very recent matches.
FORM_ALPHA_CANDIDATES = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60]

# Fixed XGBoost settings. We tune only the recent-form alpha on validation data.
XGB_PARAMS = {
    "n_estimators": 120,
    "max_depth": 2,
    "learning_rate": 0.06,
    "subsample": 0.90,
    "colsample_bytree": 0.90,
    "min_child_weight": 5,
    "reg_lambda": 3.0,
    "reg_alpha": 0.10,
    "eval_metric": "logloss",
    "random_state": RANDOM_STATE,
    "n_jobs": 2,
    "tree_method": "hist",
}

print("Setup complete.")


# %%
# =============================================
# Cell 2: Upload dataset easily in Google Colab
# =============================================

def get_dataset_path():
    """Return the dataset path. In Colab, this opens an upload button."""
    try:
        from google.colab import files  # type: ignore
        print("Upload your dataset CSV file now, for example: Dataset-1.csv")
        uploaded = files.upload()
        if not uploaded:
            raise ValueError("No file was uploaded.")
        uploaded_filename = list(uploaded.keys())[0]
        print(f"Uploaded file detected: {uploaded_filename}")
        return uploaded_filename
    except ModuleNotFoundError:
        # Local/Jupyter fallback. This lets the same code run outside Colab.
        possible_paths = [
            "/mnt/data/Dataset-1.csv",
            "Dataset-1.csv",
            "./Dataset-1.csv",
        ]
        for path in possible_paths:
            if Path(path).exists():
                print(f"Using local dataset file: {path}")
                return path
        raise FileNotFoundError(
            "Dataset-1.csv was not found locally. Put it in the working folder or run this in Colab and upload it."
        )

DATA_PATH = get_dataset_path()
OUTPUT_DIR = Path(DATA_PATH).resolve().parent / "xgb_recent_form_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"DATA_PATH  = {DATA_PATH}")
print(f"OUTPUT_DIR = {OUTPUT_DIR}")


# %%
# ======================================
# Cell 3: Load, clean, and inspect data
# ======================================

def clean_text_column(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .replace({"nan": "Unknown", "None": "Unknown", "": "Unknown"})
    )


def load_and_clean_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    original_rows = len(df)

    required_columns = ["team1", "team2", "venue", "date", "match_winner"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    text_columns_to_clean = [
        "team1", "team2", "venue", "date", "match_winner", "match_code", "match_type", "series", "toss_winner"
    ]
    for col in text_columns_to_clean:
        if col in df.columns:
            df[col] = clean_text_column(df[col])

    df["date_parsed"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["date_parsed", "team1", "team2", "venue", "match_winner"]).copy()

    # Keep only resolved matches where winner is exactly team1 or team2.
    team1_lower = df["team1"].str.lower()
    team2_lower = df["team2"].str.lower()
    winner_lower = df["match_winner"].str.lower()
    resolved_mask = (winner_lower == team1_lower) | (winner_lower == team2_lower)
    df = df[resolved_mask].copy()

    if "match_code" in df.columns:
        df["match_code_num"] = pd.to_numeric(df["match_code"], errors="coerce")
    else:
        df["match_code_num"] = np.arange(len(df))

    df = df.sort_values(["date_parsed", "match_code_num"]).reset_index(drop=True)

    # Binary target: 1 = team1 won, 0 = team2 won.
    df["team1_won"] = (df["match_winner"].str.lower() == df["team1"].str.lower()).astype(int)

    removed_rows = original_rows - len(df)
    print(f"Original rows: {original_rows}")
    print(f"Rows after cleaning/resolved-match filtering: {len(df)}")
    print(f"Rows removed: {removed_rows}")
    print(f"Date range: {df['date_parsed'].min().date()} to {df['date_parsed'].max().date()}")
    print("Target distribution:")
    print(df["team1_won"].value_counts(normalize=False).rename(index={1: "team1_won", 0: "team2_won"}))
    print("Target distribution percentage:")
    print((df["team1_won"].value_counts(normalize=True) * 100).round(2).rename(index={1: "team1_won", 0: "team2_won"}))

    return df

clean_df = load_and_clean_dataset(DATA_PATH)
clean_df.head()


# %%
# =====================================================
# Cell 4: One feature engineering only: Recent Form EWMA
# =====================================================

# This function creates only one feature-engineering family:
# pre-match exponentially weighted recent form rating.
#
# For each match, the form values are recorded BEFORE updating with that match result.
# So the current match's result is never used to create its own features.
#
# Created columns:
# - team1_recent_form_pre
# - team2_recent_form_pre
# - recent_form_diff
#
# No Elo rating is used anywhere.

def add_recent_form_features(df: pd.DataFrame, alpha: float) -> pd.DataFrame:
    df = df.copy().sort_values(["date_parsed", "match_code_num"]).reset_index(drop=True)

    team_form = defaultdict(lambda: 0.50)  # neutral starting value for teams with no history

    team1_form_values = []
    team2_form_values = []
    form_diff_values = []

    for _, row in df.iterrows():
        team1 = row["team1"]
        team2 = row["team2"]
        winner = row["match_winner"]

        team1_form_pre = float(team_form[team1])
        team2_form_pre = float(team_form[team2])

        team1_form_values.append(team1_form_pre)
        team2_form_values.append(team2_form_pre)
        form_diff_values.append(team1_form_pre - team2_form_pre)

        # Update only AFTER storing pre-match form.
        team1_result = 1.0 if winner == team1 else 0.0
        team2_result = 1.0 if winner == team2 else 0.0

        team_form[team1] = (1.0 - alpha) * team_form[team1] + alpha * team1_result
        team_form[team2] = (1.0 - alpha) * team_form[team2] + alpha * team2_result

    df["team1_recent_form_pre"] = team1_form_values
    df["team2_recent_form_pre"] = team2_form_values
    df["recent_form_diff"] = form_diff_values

    return df

example_form_df = add_recent_form_features(clean_df, alpha=0.20)
example_form_df[["date", "team1", "team2", "match_winner", "team1_recent_form_pre", "team2_recent_form_pre", "recent_form_diff", "team1_won"]].head(10)


# %%
# ===============================================
# Cell 5: Chronological train/validation/test split
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

train_df, val_df, train_val_df, test_df = chronological_split(example_form_df)

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
# ========================================================
# Cell 6: Build XGBoost pipeline and tune only form alpha
# ========================================================

CATEGORICAL_FEATURES = ["team1", "team2", "venue"]
NUMERIC_FEATURES = ["team1_recent_form_pre", "team2_recent_form_pre", "recent_form_diff"]
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES
TARGET_COLUMN = "team1_won"


def build_xgb_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("num", "passthrough", NUMERIC_FEATURES),
        ]
    )

    model = XGBClassifier(**XGB_PARAMS)

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    return pipeline


def evaluate_alpha(alpha: float) -> dict:
    form_df = add_recent_form_features(clean_df, alpha=alpha)
    train_df, val_df, _, _ = chronological_split(form_df)

    pipeline = build_xgb_pipeline()
    pipeline.fit(train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN])

    val_prob = pipeline.predict_proba(val_df[FEATURE_COLUMNS])[:, 1]
    val_pred = (val_prob >= 0.50).astype(int)

    return {
        "alpha": alpha,
        "validation_accuracy": accuracy_score(val_df[TARGET_COLUMN], val_pred),
        "validation_balanced_accuracy": balanced_accuracy_score(val_df[TARGET_COLUMN], val_pred),
        "validation_roc_auc": roc_auc_score(val_df[TARGET_COLUMN], val_prob),
    }

validation_results = pd.DataFrame([evaluate_alpha(alpha) for alpha in FORM_ALPHA_CANDIDATES])
validation_results = validation_results.sort_values(
    ["validation_accuracy", "validation_roc_auc"], ascending=False
).reset_index(drop=True)

best_alpha = float(validation_results.loc[0, "alpha"])
print(f"Best recent-form alpha selected from validation only: {best_alpha}")
validation_results


# %%
# ================================================
# Cell 7: Train final model and evaluate on test set
# ================================================

final_df = add_recent_form_features(clean_df, alpha=best_alpha)
train_df, val_df, train_val_df, test_df = chronological_split(final_df)

final_pipeline = build_xgb_pipeline()
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
print(f"Best alpha: {best_alpha}")
print(f"Train + validation accuracy: {train_accuracy:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")
print(f"Test balanced accuracy: {test_balanced_accuracy:.4f}")
print(f"Test ROC-AUC: {test_roc_auc:.4f}")
print()
print("Classification report:")
print(classification_report(test_df[TARGET_COLUMN], test_pred, target_names=["team2_won", "team1_won"]))

confusion_matrix(test_df[TARGET_COLUMN], test_pred)


# %%
# ====================================
# Cell 8: Save all result files/folder
# ====================================

# Save split summary
split_summary.to_csv(OUTPUT_DIR / "split_summary.csv", index=False)
validation_results.to_csv(OUTPUT_DIR / "validation_alpha_results.csv", index=False)

# Save cleaned feature dataset
final_df.to_csv(OUTPUT_DIR / "cleaned_dataset_with_recent_form_features.csv", index=False)

# Save classification report
report_dict = classification_report(
    test_df[TARGET_COLUMN],
    test_pred,
    target_names=["team2_won", "team1_won"],
    output_dict=True,
)
report_df = pd.DataFrame(report_dict).transpose()
report_df.to_csv(OUTPUT_DIR / "classification_report.csv")

# Save confusion matrix
cm = confusion_matrix(test_df[TARGET_COLUMN], test_pred)
cm_df = pd.DataFrame(
    cm,
    index=["actual_team2_won", "actual_team1_won"],
    columns=["predicted_team2_won", "predicted_team1_won"],
)
cm_df.to_csv(OUTPUT_DIR / "confusion_matrix.csv")

# Save prediction-level file
predictions_df = test_df[[
    "match_code", "date", "team1", "team2", "venue", "match_winner",
    "team1_recent_form_pre", "team2_recent_form_pre", "recent_form_diff", "team1_won"
]].copy()
predictions_df["predicted_probability_team1_won"] = test_prob
predictions_df["predicted_team1_won"] = test_pred
predictions_df["predicted_winner"] = np.where(test_pred == 1, predictions_df["team1"], predictions_df["team2"])
predictions_df["correct_prediction"] = predictions_df["predicted_winner"] == predictions_df["match_winner"]
predictions_df.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)

# Save metrics summary
metrics_summary = {
    "model": "XGBoostClassifier",
    "feature_engineering": "Pre-match EWMA recent form rating only; Elo removed",
    "best_recent_form_alpha": best_alpha,
    "threshold": 0.50,
    "original_rows": int(pd.read_csv(DATA_PATH).shape[0]),
    "clean_resolved_rows_used": int(len(final_df)),
    "train_plus_validation_rows": int(len(train_val_df)),
    "test_rows": int(len(test_df)),
    "train_plus_validation_accuracy": float(train_accuracy),
    "test_accuracy": float(test_accuracy),
    "test_balanced_accuracy": float(test_balanced_accuracy),
    "test_roc_auc": float(test_roc_auc),
    "test_start_date": str(test_df["date_parsed"].min().date()),
    "test_end_date": str(test_df["date_parsed"].max().date()),
    "features_used": FEATURE_COLUMNS,
    "excluded_for_leakage_reason": "All scorecard/player performance columns, toss_winner, team1_bat_first, match_result, match_outcome_type, match_winner as feature",
    "xgboost_params": XGB_PARAMS,
}

with open(OUTPUT_DIR / "metrics_summary.json", "w", encoding="utf-8") as f:
    json.dump(metrics_summary, f, indent=4)

# Save model pipeline
if joblib is not None:
    joblib.dump(final_pipeline, OUTPUT_DIR / "xgboost_recent_form_pipeline.joblib")

print(f"Saved result files to: {OUTPUT_DIR}")
print(sorted([p.name for p in OUTPUT_DIR.iterdir()]))


# %%
# ==========================================
# Cell 9: Feature importance and result plots
# ==========================================

# Feature names after one-hot encoding
preprocessor = final_pipeline.named_steps["preprocess"]
model = final_pipeline.named_steps["model"]

cat_encoder = preprocessor.named_transformers_["cat"]
cat_feature_names = list(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
all_feature_names = cat_feature_names + NUMERIC_FEATURES

feature_importance_df = pd.DataFrame({
    "feature": all_feature_names,
    "importance": model.feature_importances_,
}).sort_values("importance", ascending=False)

feature_importance_df.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)

# Plot confusion matrix
plt.figure(figsize=(6, 5))
plt.imshow(cm)
plt.title("Confusion Matrix - XGBoost Recent Form Model")
plt.xticks([0, 1], ["Pred team2", "Pred team1"])
plt.yticks([0, 1], ["Actual team2", "Actual team1"])
plt.xlabel("Predicted label")
plt.ylabel("Actual label")
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=160)
plt.show()

# Plot ROC curve
fpr, tpr, _ = roc_curve(test_df[TARGET_COLUMN], test_prob)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"ROC-AUC = {test_roc_auc:.4f}")
plt.plot([0, 1], [0, 1], linestyle="--", label="Random baseline")
plt.title("ROC Curve - XGBoost Recent Form Model")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "roc_curve.png", dpi=160)
plt.show()

# Plot top feature importances
top_n = 20
top_features = feature_importance_df.head(top_n).iloc[::-1]
plt.figure(figsize=(8, 7))
plt.barh(top_features["feature"], top_features["importance"])
plt.title(f"Top {top_n} Feature Importances")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "feature_importance_top20.png", dpi=160)
plt.show()

feature_importance_df.head(20)


# %%
# =================================
# Cell 10: Write run summary and zip
# =================================

summary_text = f"""
XGBoost Recent Form Model Summary
=================================

Task:
Predict whether team1 wins a cricket ODI match using pre-match data.

Model:
Plain XGBoostClassifier.

Only feature engineering used:
Pre-match EWMA recent form rating.
- team1_recent_form_pre
- team2_recent_form_pre
- recent_form_diff

Elo status:
Elo ratings are completely removed and not used anywhere.

Leakage control:
The model excludes post-match scorecard columns, match_result, match_outcome_type,
toss_winner, team1_bat_first, and match_winner as a feature.
The recent-form feature is calculated before each match is updated with that match result.

Selected alpha from validation only:
{best_alpha}

Rows:
Clean/resolved rows used: {len(final_df)}
Train + validation rows: {len(train_val_df)}
Test rows: {len(test_df)}

Date ranges:
Train + validation: {train_val_df['date_parsed'].min().date()} to {train_val_df['date_parsed'].max().date()}
Test: {test_df['date_parsed'].min().date()} to {test_df['date_parsed'].max().date()}

Final metrics:
Train + validation accuracy: {train_accuracy:.4f}
Test accuracy: {test_accuracy:.4f}
Test balanced accuracy: {test_balanced_accuracy:.4f}
Test ROC-AUC: {test_roc_auc:.4f}

Important thesis note:
Your dataset has team1_bat_first = 1 for all rows, which suggests team1 may be the batting-first team.
This code excludes team1_bat_first and toss_winner, but if team1/team2 ordering itself was created after toss,
then this is safer as a post-toss/team-order prediction setup than as a strict before-toss setup.
For strict before-toss prediction, team ordering should be made independent of innings order.
""".strip()

with open(OUTPUT_DIR / "run_summary.txt", "w", encoding="utf-8") as f:
    f.write(summary_text)

zip_path = OUTPUT_DIR.parent / "xgb_recent_form_outputs.zip"
with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zipf:
    for file_path in sorted(OUTPUT_DIR.iterdir()):
        zipf.write(file_path, arcname=f"xgb_recent_form_outputs/{file_path.name}")

print(summary_text)
print(f"\nZipped results saved to: {zip_path}")
