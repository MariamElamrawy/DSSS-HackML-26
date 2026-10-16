# Predicting Graduate Overqualification · HackML 2026

Can you tell, from a graduate's survey answers, whether they ended up in a job that doesn't need their degree?

This was my entry to **DSSS HackML 2026**, a Kaggle competition run by SFU's Data Science Student Society. The data comes from Statistics Canada's **National Graduates Survey (2020)**. The task is binary classification: predict whether a graduate is `overqualified` for their current job, scored on **ROC-AUC**.

**Best result:** gradient boosting at **0.705 validation AUC**, against **0.692** for a logistic regression baseline.

<!-- Add your final leaderboard placing here if you have it, e.g. "Placed 12th of 80 teams." -->

## The data

| | |
|---|---|
| Training rows | 7,709 graduates |
| Test rows | 2,508 |
| Features | 23 survey variables (credential level, field of study, student loans, parents' education, age at graduation, and more) |
| Target | `overqualified`: 2,964 yes (38%), 4,745 no |

Almost every variable is a coded category, and the survey marks non-responses with codes like `6` (valid skip), `9` (not stated) and `99`. Some columns are mostly missing once those codes are recoded: `PGM_P036` is 83% missing, `DBTOTGRD` (student debt) is 79%.

The competition data isn't included here. Download it from the Kaggle competition page and place `train.csv` and `test.csv` in `Data/`.

## Approach

**1. Cleaning** (`src/preprocess.py`)
- Recode survey non-response codes (`6`, `9`, `99`) as missing.
- Give each categorical variable its own `Missing` level instead of dropping rows. In a survey, *not answering* can carry signal.
- Median-fill the one numeric column, using the training median only.
- One-hot encode, then align test columns to the training columns (train has 114 dummy columns, test has only 84 before alignment).

**2. Baseline first.** Logistic regression sets the bar every other model has to beat.

**3. Compare a small set of tree ensembles** on the same split: random forest, sklearn gradient boosting, HistGradientBoosting, XGBoost and CatBoost, plus simple averaged ensembles.

**4. Submit** the strongest single model, retrained on all the training data.

## Results

Validation ROC-AUC from the competition notebooks, on an 80/20 hold-out split.

| Model | AUC | Notebook |
|---|---|---|
| Logistic regression (baseline) | 0.6915 | 03 |
| Random forest | 0.6801 | 03 |
| XGBoost | 0.6916 | 04 |
| **Gradient boosting (submitted)** | **0.7052** | 03 |
| Weighted ensemble: GB / RF / XGB (0.6 / 0.2 / 0.2) | 0.7065* | 05 |

On a second, stratified split (notebook 06):

| Model | AUC |
|---|---|
| HistGradientBoosting | 0.6932 |
| Gradient boosting | 0.6999 |
| CatBoost | 0.7011 |
| Simple average of the three | 0.7036 |
| Ensemble with searched weights | 0.7042* |

\* Ensemble weights were chosen by looking at the validation score, so these two numbers are optimistic.

### What the numbers say

- **The baseline is almost as good as the best model.** Gradient boosting beats logistic regression by about 0.014 AUC. Every model lands between 0.68 and 0.71. That pattern says the ceiling is set by the information in the survey answers, not by the choice of algorithm.
- **Ensembling added very little.** Averaging three models gained about 0.002 to 0.004 AUC, which is within the noise of a single split. I submitted the single gradient boosting model because it is simpler and almost as accurate.
- **More complex wasn't better.** XGBoost and HistGradientBoosting did not beat sklearn's plain gradient boosting here.

## Limitations and what I'd do next

Going back through the code after the competition, I found four things worth fixing. Each one changes the inputs, so the tables above are from the original run. Run `python src/compare_models.py` with the data in place to regenerate them.

1. **The row `id` was used as a feature.** It is just an identifier, so it can only add noise. The scripts in `src/` now drop it before training.
2. **Six hand-made flag features were always zero.** The columns had already been converted to strings like `"1.0"`, so the check `== '1'` never matched. They are removed in `src/preprocess.py`.
3. **Missing-value codes were applied to every column.** In some variables (field of study, for example) `6` or `9` may be a real category rather than a non-response. The fix is to recode per column, following the NGS codebook.
4. **One hold-out split is a noisy way to rank models** this close together. `python src/compare_models.py --cv` gives a 5-fold cross-validated estimate with a spread, which is the fairer comparison.

Beyond that: check which survey variables drive the predictions (permutation importance), and look at where the model is wrong by subgroup, such as field of study and credential level. A model that is mostly wrong for one group of graduates would be a problem in any real use.

## How to run

```bash
pip install -r requirements.txt

# put train.csv and test.csv in Data/
python src/preprocess.py           # writes Data/processed/
python src/compare_models.py       # results/model_comparison.csv
python src/compare_models.py --cv  # 5-fold CV version
python src/make_submission.py      # writes submission.csv
```

## Repo structure

```
├── src/
│   ├── preprocess.py        cleaning and encoding
│   ├── compare_models.py    baseline vs. ensembles, hold-out or 5-fold CV
│   └── make_submission.py   final model and Kaggle submission
├── notebooks/               the working notebooks from the competition, in order
│   ├── 01_class_balance.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_baseline_comparison.ipynb
│   ├── 04_xgboost.ipynb
│   ├── 05_ensemble_gb_rf_xgb.ipynb
│   ├── 06_final_comparison.ipynb
│   └── 07_submission.ipynb
├── requirements.txt
└── Data/                    not committed (see above)
```

The notebooks are the record of how the work actually went, including the dead ends. The scripts in `src/` are the clean, re-runnable version.

## Tools

Python · pandas · NumPy · scikit-learn · XGBoost · CatBoost · Jupyter
