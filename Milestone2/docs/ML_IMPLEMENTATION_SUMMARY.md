# ML Implementation Summary

The ML stage compares baseline market-only models with macro-enriched models.

## Target

The main target is next-day BIST 100 direction:

- `1`: next-day return is positive
- `0`: next-day return is zero or negative

## Baseline Features

- Lagged BIST 100 returns
- Lagged GARAN returns
- Rolling mean
- Rolling volatility

## Macro-Enriched Features

- Lagged USD/TRY returns
- Lagged CPI changes
- Lagged policy rate changes

## Models

Classification:

- Dummy Classifier
- Logistic Regression
- Random Forest Classifier
- Gradient Boosting Classifier

Secondary regression:

- Ridge Regression
- Random Forest Regressor

## Evaluation

Classification metrics:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

Regression metrics:

- MAE
- RMSE
- R-squared

The key comparison is baseline vs macro-enriched performance.
