# DSA 210 Term Project: BIST 100 and Macroeconomic Enrichment

**Student:** Gökberk Marmara  
**Course:** DSA 210  

## Project summary

This project studies whether Turkish macroeconomic indicators add useful information when analyzing and predicting BIST 100 daily movements. Instead of treating the stock market as a single isolated price series, the project combines market data with inflation, interest-rate, and exchange-rate indicators.

The main question is:

> Do CPI, policy rate, and USD/TRY movements provide additional explanatory or predictive information for BIST 100 daily returns?

The project follows one consistent pipeline:

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

## Current repository note

Some earlier milestone files are under the `Milestone1/` and `Milestone2/` folders. For final review, the important files are:

```text
README.md
requirements.txt
Milestone2/Proposal/DSA210_Project_Proposal_Final.md
Milestone2/notebooks/EDA_Hypothesis_Testing.ipynb
Milestone2/notebooks/ML_Modeling_BIST_EVDS.ipynb
Milestone2/scripts/EDA_Hypothesis_Testing.py
Milestone2/scripts/ML_Modeling_BIST_EVDS.py
Milestone1/AI_USAGE.md
```

## Methodology

### EDA and hypothesis testing

The EDA stage checks the structure of the dataset, missing values, and basic statistics. It then creates several visualizations, including time-series plots, normalized comparisons, return distributions, rolling volatility, transformed-variable correlations, scatter plots, and yearly return distributions.

Raw price and macroeconomic level variables are transformed before statistical testing:

- `XU100_Return`: BIST 100 daily return
- `GARAN_Return`: GARAN daily return
- `USD_TRY_Return`: daily USD/TRY return
- `CPI_Diff`: first difference of CPI
- `PolicyRate_Diff`: first difference of policy rate

The hypothesis tests are:

- **ADF test** for stationarity
- **Pearson correlation** for linear association
- **Granger causality** for predictive information

Granger causality is interpreted carefully. It does not prove true economic causality; it only tests whether past values of one series help predict another series.

### Machine learning

The ML implementation compares two setups:

1. **Baseline models:** use only historical market features.
2. **Macro-enriched models:** use historical market features plus lagged macroeconomic variables.

The main prediction target is next-day BIST 100 direction:

- `1`: next-day BIST 100 return is positive
- `0`: next-day BIST 100 return is zero or negative

Classification models:

- Dummy Classifier
- Logistic Regression
- Random Forest Classifier
- Gradient Boosting Classifier

Secondary regression models:

- Ridge Regression
- Random Forest Regressor

Because the data is time-series data, the notebooks use a chronological 80/20 train-test split rather than a random split. This avoids leaking future observations into training.

## How to run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run EDA and hypothesis testing first:

```bash
jupyter notebook Milestone2/notebooks/EDA_Hypothesis_Testing.ipynb
```

Then run the ML notebook:

```bash
jupyter notebook Milestone2/notebooks/ML_Modeling_BIST_EVDS.ipynb
```

For final review, run all cells and save the notebooks so that outputs are visible on GitHub.

## Outputs

The notebooks save generated files into:

```text
images/
results/
data/
```

Typical outputs include EDA plots, ADF/Pearson/Granger result tables, model performance tables, baseline vs macro-enriched comparisons, and feature importance plots.

## Notes on interpretation

The goal is not to claim that the stock market can be predicted perfectly. Daily market movements are noisy and affected by many variables that are not included here. The practical question is narrower: whether the macro-enriched setup performs better than a market-only setup and whether the statistical tests support using macro variables in the modeling stage.

## AI usage

AI tools were used for coding support, structure, and debugging. The final project direction, interpretation, and responsibility for explaining the work belong to me. Details are documented in `Milestone1/AI_USAGE.md`.
