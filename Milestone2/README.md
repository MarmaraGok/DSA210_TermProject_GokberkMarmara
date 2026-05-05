# DSA 210 Term Project — BIST 100 and Macroeconomic Enrichment

**Student:** Gökberk Marmara  
**Course:** DSA 210  

## Project Overview

This project analyzes whether Turkish macroeconomic indicators improve the analysis and prediction of BIST 100 market movements. The project combines historical market data with macroeconomic indicators from TCMB EVDS.

The central idea is not to make a generic stock-price prediction model. Instead, the project compares a market-only approach with a macroeconomically enriched approach.

## Research Question

Do Turkish macroeconomic indicators such as CPI, policy rate, and USD/TRY exchange-rate movements provide useful explanatory or predictive information for BIST 100 daily returns?

## Data Sources

### Financial Data

- Source: Yahoo Finance through `yfinance`
- Variables:
  - `XU100`: BIST 100 close price
  - `GARAN`: GARAN close price

### Macroeconomic Enrichment Data

- Source: TCMB EVDS through `evdspy`
- Variables:
  - `CPI`: Consumer Price Index
  - `PolicyRate`: policy interest rate
  - `USD_TRY`: USD/TRY exchange rate

## Required Dataset

Place the enriched dataset here:

```text
data/enriched_bist_data.csv
```

Required columns:

```text
Date, XU100, GARAN, CPI, PolicyRate, USD_TRY
```

## Repository Structure

```text
DSA210_TermProject_GokberkMarmara/
|
├── README.md
├── requirements.txt
|
├── Proposal/
│   ├── DSA210_Project_Proposal_Updated.docx
│   └── DSA210_Project_Proposal_Updated.md
|
├── data/
│   └── enriched_bist_data.csv
|
├── notebooks/
│   ├── EDA_Hypothesis_Testing.ipynb
│   └── ML_Modeling_BIST_EVDS.ipynb
|
├── scripts/
│   ├── EDA_Hypothesis_Testing.py
│   └── ML_Modeling_BIST_EVDS.py
|
├── images/
├── results/
|
├── docs/
│   ├── FINAL_ALIGNMENT_CHECK.md
│   ├── CODE_EXPLANATION_GUIDE.md
│   └── ML_IMPLEMENTATION_SUMMARY.md
|
└── Milestone1/
    └── AI_USAGE.md
```

## Methodology

### EDA and Hypothesis Testing

The EDA notebook includes:

- Summary statistics
- Missing value analysis
- Time-series plots
- Normalized comparison plots
- Return distribution
- Rolling volatility
- Correlation heatmap using transformed variables
- Scatter plots between macroeconomic changes and returns
- Yearly return distribution

The hypothesis testing notebook includes:

- ADF stationarity tests
- Pearson correlation tests
- Granger causality tests

### Machine Learning

The ML notebook compares two setups:

1. **Baseline models:** use only historical market features.
2. **Macro-enriched models:** add macroeconomic indicators from TCMB EVDS.

The main prediction target is next-day BIST 100 market direction:

- `1`: next-day BIST 100 return is positive
- `0`: next-day BIST 100 return is zero or negative

Models used:

- Dummy Classifier
- Logistic Regression
- Random Forest Classifier
- Gradient Boosting Classifier
- Ridge Regression as secondary regression model
- Random Forest Regressor as secondary regression model

## How to Run

Install requirements:

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

For final submission, run all cells and save the notebooks so that outputs are visible on GitHub.

## Important Note

This package contains final notebook templates and scripts. Since the actual dataset is not included here, the notebooks must be executed after placing `enriched_bist_data.csv` inside the `data/` folder.
