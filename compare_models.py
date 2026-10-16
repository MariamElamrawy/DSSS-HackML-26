"""
Compare a logistic regression baseline against tree ensembles on ROC-AUC.

Uses the processed files from src/preprocess.py. By default it reproduces the
setup from the competition notebooks: a stratified 80/20 split with seed 42.
Pass --cv to get a 5-fold stratified cross-validation estimate instead, which
is less sensitive to how one split happens to fall.

Usage:
    python src/compare_models.py
    python src/compare_models.py --cv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42


def build_models() -> dict:
    """Model configurations taken from the competition notebooks."""
    models = {
        # Baseline: everything else has to beat this to be worth the complexity.
        "Logistic Regression (baseline)": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000)
        ),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=94, learning_rate=0.1, max_depth=3, random_state=SEED
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.01, random_state=SEED
        ),
    }
    try:
        from xgboost import XGBClassifier

        models["XGBoost"] = XGBClassifier(
            n_estimators=200, learning_rate=0.1, max_depth=5, random_state=SEED
        )
    except ImportError:
        print("xgboost not installed, skipping")
    try:
        from catboost import CatBoostClassifier

        models["CatBoost"] = CatBoostClassifier(
            iterations=200, learning_rate=0.05, random_state=SEED, verbose=0
        )
    except ImportError:
        print("catboost not installed, skipping")
    return models


def holdout(X, y, models):
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    rows, preds = [], {}
    for name, model in models.items():
        model.fit(X_tr, y_tr)
        preds[name] = model.predict_proba(X_val)[:, 1]
        rows.append({"model": name, "val_auc": roc_auc_score(y_val, preds[name])})
        print(f"{name:32s} {rows[-1]['val_auc']:.4f}")

    # Simple average of the three strongest individual models.
    # Weights are NOT tuned on this validation set, so the score stays honest.
    top3 = sorted(rows, key=lambda r: r["val_auc"], reverse=True)[:3]
    names = [r["model"] for r in top3]
    avg = np.mean([preds[n] for n in names], axis=0)
    rows.append({"model": "Average of top 3: " + ", ".join(names),
                 "val_auc": roc_auc_score(y_val, avg)})
    print(f"{'Average of top 3':32s} {rows[-1]['val_auc']:.4f}")
    return pd.DataFrame(rows)


def cross_validate(X, y, models, folds=5):
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    rows = []
    for name, model in models.items():
        scores = []
        for tr, va in skf.split(X, y):
            model.fit(X.iloc[tr], y.iloc[tr])
            scores.append(roc_auc_score(y.iloc[va], model.predict_proba(X.iloc[va])[:, 1]))
        rows.append({"model": name, "cv_auc_mean": np.mean(scores), "cv_auc_std": np.std(scores)})
        print(f"{name:32s} {np.mean(scores):.4f} +/- {np.std(scores):.4f}")
    return pd.DataFrame(rows)


def main(data_dir: Path, use_cv: bool):
    # The row id is an identifier, not information about the graduate.
    X = pd.read_csv(data_dir / "processed" / "X_train_clean.csv").drop(columns=["id"])
    y =pd.read_csv(data_dir / "processed" / "y_train.csv")["overqualified"]
    print(f"{len(y)} rows, {X.shape[1]} features, {y.mean():.1%} overqualified\n")

    models = build_models()
    results = cross_validate(X, y, models) if use_cv else holdout(X, y, models)

    Path("results").mkdir(exist_ok=True)
    out = Path("results") / ("model_comparison_cv.csv" if use_cv else "model_comparison.csv")
    results.round(4).to_csv(out, index=False)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("Data"))
    parser.add_argument("--cv", action="store_true", help="5-fold stratified CV instead of one split")
    args = parser.parse_args()
    main(args.data_dir, args.cv)
