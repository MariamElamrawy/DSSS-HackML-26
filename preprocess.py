"""
Clean and encode the National Graduates Survey (NGS 2020) data for modelling.

Reads Data/train.csv and Data/test.csv, writes encoded feature matrices to
Data/processed/. Every statistic used for imputation is learned on the
training set only and then applied to the test set.

Usage:
    python src/preprocess.py
    python src/preprocess.py --data-dir Data
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

TARGET = "overqualified"
ID_COL = "id"

# Survey codes that mean "valid skip", "don't know", or "not stated".
# NOTE: these are replaced in every numeric column. In a few columns
# (e.g. PGMCIPAP, field of study) 6 or 9 may be a real category, so this
# rule is worth checking against the NGS codebook. See README, "Limitations".
MISSING_CODES = [6, 9, 99]

CATEGORICAL_COLS = [
    "CERTLEVP", "PGMCIPAP", "PGM_P034", "PGM_P036", "PGM_280A", "PGM_280B",
    "PGM_280C", "PGM_280F", "PGM_P401", "STULOANS", "GENDER2", "CTZSHIPP",
    "VISBMINP", "DDIS_FL", "PAR1GRD", "PAR2GRD", "BEF_P140",
]

# Ordered categories (education level, debt band, age group, ...).
# Treated as categorical so "Missing" can be its own level.
ORDINAL_COLS = ["HLOSGRDP", "PREVLEVP", "DBTOTGRD", "SCHOLARP", "GRADAGEP"]

NUMERIC_COLS = ["BEF_160"]


def replace_missing_codes(df: pd.DataFrame) -> pd.DataFrame:
    """Turn survey non-response codes into NaN in numeric columns."""
    df = df.copy()
    for col in df.columns:
        if col == ID_COL:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].replace(MISSING_CODES, np.nan)
    return df


def fill_missing(train: pd.DataFrame, test: pd.DataFrame):
    """Median-fill numeric columns (median from train); 'Missing' level for categoricals."""
    train, test = train.copy(), test.copy()

    for col in NUMERIC_COLS:
        med = train[col].median()
        train[col] = train[col].fillna(med)
        test[col] = test[col].fillna(med)

    for col in CATEGORICAL_COLS + ORDINAL_COLS:
        train[col] = train[col].fillna("Missing").astype(str)
        test[col] = test[col].fillna("Missing").astype(str)

    return train, test


def encode(train: pd.DataFrame, test: pd.DataFrame):
    """One-hot encode and align test columns to train."""
    cols = CATEGORICAL_COLS + ORDINAL_COLS
    X_train = pd.get_dummies(train, columns=cols, drop_first=True)
    X_test = pd.get_dummies(test, columns=cols, drop_first=True)
    # Categories that only appear in test are dropped; ones missing from test are 0.
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
    return X_train, X_test


def main(data_dir: Path):
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    print(f"train: {train.shape}  test: {test.shape}")

    y_train = train[TARGET]
    X_train = train.drop(columns=[TARGET])
    X_test = test.copy()

    X_train = replace_missing_codes(X_train)
    X_test = replace_missing_codes(X_test)

    missing = X_train.isna().mean().sort_values(ascending=False).head(10)
    print("\nMost-missing columns after recoding:\n" + missing.round(3).to_string())

    X_train, X_test = fill_missing(X_train, X_test)
    assert X_train.isna().sum().sum() == 0 and X_test.isna().sum().sum() == 0

    X_train, X_test = encode(X_train, X_test)
    print(f"\nEncoded: train {X_train.shape}  test {X_test.shape}")

    out = data_dir / "processed"
    out.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(out / "X_train_clean.csv", index=False)
    y_train.to_frame(TARGET).to_csv(out / "y_train.csv", index=False)
    X_test.to_csv(out / "X_test_clean.csv", index=False)
    print(f"Saved to {out}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--data-dir", type=Path, default=Path("Data"))
    main(parser.parse_args().data_dir)
