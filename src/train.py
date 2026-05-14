"""Model training and evaluation module."""

from pathlib import Path
import argparse
import json
import pickle
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.base import clone
from xgboost import XGBClassifier


RANDOM_STATE = 42


def _manual_cv_f1(model, x_data: pd.DataFrame, y_data: pd.Series, folds: int = 3) -> np.ndarray:
    """Compute CV F1 manually to avoid estimator tag compatibility issues."""
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    scores: list[float] = []
    for train_idx, valid_idx in splitter.split(x_data, y_data):
        x_fold_train = x_data.iloc[train_idx]
        x_fold_valid = x_data.iloc[valid_idx]
        y_fold_train = y_data.iloc[train_idx]
        y_fold_valid = y_data.iloc[valid_idx]

        fold_model = clone(model)
        fold_model.fit(x_fold_train, y_fold_train)
        y_fold_pred = fold_model.predict(x_fold_valid)
        scores.append(f1_score(y_fold_valid, y_fold_pred, zero_division=0))
    return np.array(scores)


def train_models(input_path: str, models_dir: str, reports_dir: str) -> tuple[Path, Path]:
    """Train three classifiers, evaluate, and persist metrics plus artifacts."""
    features_file = Path(input_path)
    if not features_file.exists():
        raise FileNotFoundError(f"Features file not found at '{features_file}'.")

    df = pd.read_csv(features_file)
    if "is_account_takeover" not in df.columns:
        raise ValueError("Target column 'is_account_takeover' missing from features file.")

    x_data = df.drop(columns=["is_account_takeover"])
    y_data = df["is_account_takeover"].astype(int)

    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=0.2, stratify=y_data, random_state=RANDOM_STATE
    )

    distribution_before = y_train.value_counts().sort_index().to_dict()
    print(f"Train distribution before SMOTE: {distribution_before}")

    smote = SMOTE(random_state=RANDOM_STATE)
    x_train_smote, y_train_smote = smote.fit_resample(x_train, y_train)
    distribution_after = y_train_smote.value_counts().sort_index().to_dict()
    print(f"Train distribution after SMOTE: {distribution_after}")

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1_000, solver="lbfgs", random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=250, random_state=RANDOM_STATE, n_jobs=-1),
        "XGBoost": XGBClassifier(
            n_estimators=250,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    models_path = Path(models_dir)
    reports_path = Path(reports_dir)
    models_path.mkdir(parents=True, exist_ok=True)
    reports_path.mkdir(parents=True, exist_ok=True)

    metrics: dict = {
        "data_distribution": {
            "before_smote": {str(k): int(v) for k, v in distribution_before.items()},
            "after_smote": {str(k): int(v) for k, v in distribution_after.items()},
        },
        "models": {},
        "best_model": "",
    }

    best_model_name = ""
    best_auc = -1.0

    for name, model in models.items():
        print(f"\nTraining {name} ...")
        cv_scores = _manual_cv_f1(model, x_train_smote, y_train_smote, folds=3)
        model.fit(x_train_smote, y_train_smote)

        y_pred = model.predict(x_test)
        y_prob = model.predict_proba(x_test)[:, 1]

        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1_val = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_prob)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        conf = confusion_matrix(y_test, y_pred)

        print(f"{name} classification report:")
        print(classification_report(y_test, y_pred, digits=4, zero_division=0))
        print(f"{name} ROC-AUC: {roc_auc:.4f}")

        model_file_name = f"{name.lower().replace(' ', '_')}.pkl"
        with (models_path / model_file_name).open("wb") as handle:
            pickle.dump(model, handle)

        metrics["models"][name] = {
            "model_file": model_file_name,
            "cv_f1_mean": float(np.mean(cv_scores)),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1_val),
            "roc_auc": float(roc_auc),
            "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
            "confusion_matrix": conf.tolist(),
        }

        if name == "Random Forest":
            metrics["models"][name]["feature_importances"] = {
                col: float(score) for col, score in zip(x_data.columns, model.feature_importances_)
            }

        if roc_auc > best_auc:
            best_auc = roc_auc
            best_model_name = name

    metrics["best_model"] = best_model_name
    with (reports_path / "model_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    with (models_path / "feature_columns.json").open("w", encoding="utf-8") as handle:
        json.dump({"columns": x_data.columns.tolist()}, handle, indent=2)

    print(f"Best model by ROC-AUC: {best_model_name} ({best_auc:.4f})")
    return models_path, reports_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ML models and save evaluation metrics.")
    parser.add_argument("--input", default="data/processed/features.csv")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--reports-dir", default="reports")
    args = parser.parse_args()
    train_models(args.input, args.models_dir, args.reports_dir)
