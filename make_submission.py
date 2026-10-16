"""
Train the final model on all training data and write the Kaggle submission.

The submitted model was sklearn's GradientBoostingClassifier (94 trees),
the strongest single model in validation. Output is the predicted
probability of being overqualified, since the competition is scored on ROC-AUC.

Usage:
    python src/make_submission.py
"""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier


def main(data_dir: Path, out_path: Path):
    processed = data_dir / "processed"
    X_train = pd.read_csv(processed / "X_train_clean.csv")
    y_train = pd.read_csv(processed / "y_train.csv")["overqualified"]
    X_test = pd.read_csv(processed / "X_test_clean.csv")

    ids = X_test["id"]
    # The row id is an identifier, not a feature.
    X_train = X_train.drop(columns=["id"])
    X_test = X_test.drop(columns=["id"])

    model = GradientBoostingClassifier(n_estimators=94, random_state=42)
    model.fit(X_train, y_train)

    submission = pd.DataFrame({
        "id": ids,
        "overqualified": model.predict_proba(X_test)[:, 1],
    })
    submission.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(submission)} rows)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("Data"))
    parser.add_argument("--out", type=Path, default=Path("submission.csv"))
    args = parser.parse_args()
    main(args.data_dir, args.out)
