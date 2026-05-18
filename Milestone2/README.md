# DSA 210 Term Project: BIST 100 and Macroeconomic Enrichment

**Student:** Gökberk Marmara  
**Course:** DSA 210  

## Project summary

This project studies whether Turkish macroeconomic indicators add useful information when analyzing and predicting BIST 100 daily movements. Instead of treating the stock market as a single isolated price series, the project combines market data with inflation, interest-rate, and exchange-rate indicators.

The main question is:

> Do CPI, policy rate, and USD/TRY movements provide additional explanatory or predictive information for BIST 100 daily returns?

The project follows this pipeline:

```text
Data collection -> EDA -> Hypothesis testing -> Machine learning -> Baseline vs macro-enriched comparison
```

## Data

The project uses two sources:

1. **Financial market data** from Yahoo Finance via `yfinance`
   - `XU100`: BIST 100 close price
   - `GARAN`: GARAN close price

2. **Macroeconomic data** from TCMB EVDS via `evdspy`
   - `CPI`: Consumer Price Index
   - `PolicyRate`: policy interest rate
   - `USD_TRY`: USD/TRY exchange rate

The expected dataset is:

```text
data/enriched_bist_data.csv
```

Required columns:

```text
Date, XU100, GARAN, CPI, PolicyRate, USD_TRY
```

If `data/clean_enriched_bist_data.csv` exists, the ML notebook uses it. Otherwise, it falls back to `data/enriched_bist_data.csv`.

## Main files

```text
Proposal/DSA210_Project_Proposal_Final.md
notebooks/EDA_Hypothesis_Testing.ipynb
notebooks/ML_Modeling_BIST_EVDS.ipynb
scripts/EDA_Hypothesis_Testing.py
scripts/ML_Modeling_BIST_EVDS.py
requirements.txt
```

## Methodology

### EDA and hypothesis testing

The EDA stage checks missing values, summary statistics, market and macro time-series behavior, normalized comparisons, return distributions, rolling volatility, transformed-variable correlations, scatter plots, and yearly return distributions.

The transformed variables are:

- `XU100_Return`
- `GARAN_Return`
- `USD_TRY_Return`
- `CPI_Diff`
- `PolicyRate_Diff`

The hypothesis tests are:

- ADF stationarity test
- Pearson correlation test
- Granger causality test

Granger causality is interpreted as predictive information, not proof of true causality.

### Machine learning

The ML notebook compares:

1. Baseline models using only historical market features.
2. Macro-enriched models using market features plus TCMB EVDS macro variables.

The main prediction target is next-day BIST 100 market direction:

- `1`: next-day BIST 100 return is positive
- `0`: next-day BIST 100 return is zero or negative

Models used:

- Dummy Classifier
- Logistic Regression
- Random Forest Classifier
- Gradient Boosting Classifier
- Ridge Regression as a secondary regression model
- Random Forest Regressor as a secondary regression model

A chronological 80/20 train-test split is used because the dataset is time-series data.

## How to run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run EDA and hypothesis testing first:

```bash
jupyter notebook notebooks/EDA_Hypothesis_Testing.ipynb
```

Then run ML implementation:

```bash
jupyter notebook notebooks/ML_Modeling_BIST_EVDS.ipynb
```

Before final review, run all cells and save the notebooks so that outputs are visible on GitHub.

## Interpretation note

The goal is not to claim perfect stock-market prediction. The project tests a narrower question: whether adding CPI, policy rate, and USD/TRY variables improves the analysis and prediction of BIST 100 daily direction compared with market-only features.
