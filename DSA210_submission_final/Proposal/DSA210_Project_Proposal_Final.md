# DSA 210 Project Proposal

**Student:** Gökberk Marmara  
**Course:** DSA 210  
**Project title:** Macroeconomic Enrichment for BIST 100 Market Movement Analysis and Prediction

## 1. Motivation

Stock-price prediction is a common project topic, so I wanted to avoid building a project that only takes a price series and tries to predict the next value. The more useful angle for this project is to ask whether macroeconomic context adds information to market analysis.

The Turkish stock market is a good setting for this question. Inflation, interest-rate decisions, and exchange-rate movements are frequently discussed together with market behavior. For that reason, this project combines BIST 100 market data with macroeconomic indicators from TCMB EVDS.

The project is not designed to prove that the stock market can be predicted perfectly. A more realistic goal is to test whether macro-enriched features provide additional explanatory or predictive value compared with market-only features.

## 2. Research question

The main research question is:

> Do Turkish macroeconomic indicators such as CPI, policy rate, and USD/TRY movements provide useful explanatory or predictive information for BIST 100 daily returns?

The machine-learning stage focuses on a related comparison:

> Do macro-enriched models perform better than baseline models that use only historical market variables?

## 3. Data sources

The project uses two data sources.

### 3.1 Financial market data

Financial market data is collected from Yahoo Finance through `yfinance`.

Main variables:

- `XU100`: BIST 100 close price
- `GARAN`: GARAN close price

BIST 100 is the main index of the project. GARAN is included as an additional market-related variable because it is a major Turkish bank and can reflect domestic financial-sector behavior.

### 3.2 Macroeconomic enrichment data

Macroeconomic variables are collected from TCMB EVDS through `evdspy`.

Main variables:

- `CPI`: Consumer Price Index
- `PolicyRate`: policy interest rate
- `USD_TRY`: USD/TRY exchange rate

These variables represent inflation, monetary policy, and exchange-rate pressure. They are used to test whether macroeconomic context improves the analysis of BIST 100 returns.

## 4. Dataset construction

The expected dataset is:

```text
data/enriched_bist_data.csv
```

Required columns:

```text
Date, XU100, GARAN, CPI, PolicyRate, USD_TRY
```

Before statistical testing and modeling, the raw variables are transformed into returns or first differences:

- `XU100_Return`: daily percentage return of BIST 100
- `GARAN_Return`: daily percentage return of GARAN
- `USD_TRY_Return`: daily percentage change in USD/TRY
- `CPI_Diff`: first difference of CPI
- `PolicyRate_Diff`: first difference of policy rate

This transformation is important because raw market prices and macroeconomic level variables often contain trends. Working with changes helps reduce misleading relationships caused only by long-term upward or downward movement.

## 5. Exploratory data analysis

The EDA stage will examine both the raw variables and the transformed variables. The purpose is to understand the dataset before moving into formal tests and ML models.

The EDA includes:

- Summary statistics
- Missing value analysis
- Time-series plots of market and macro variables
- Normalized comparison of market and macro series
- BIST 100 return distribution
- Rolling volatility of BIST 100 returns
- Correlation heatmap using transformed variables
- Scatter plots between macroeconomic changes and BIST 100 returns
- Yearly distribution of BIST 100 daily returns

Each visualization is accompanied by written commentary so that the notebook shows not only the output but also the reasoning behind it.

## 6. Hypothesis testing

The project uses three statistical tests.

### 6.1 ADF stationarity test

The Augmented Dickey-Fuller test is used to check whether the transformed variables are stationary.

- Null hypothesis: the series is non-stationary.
- Alternative hypothesis: the series is stationary.

Stationarity matters because several time-series methods can be misleading when variables contain strong trends.

### 6.2 Pearson correlation test

Pearson correlation is used to test whether transformed macroeconomic variables have statistically significant linear relationships with BIST 100 daily returns.

- Null hypothesis: there is no statistically significant linear relationship.
- Alternative hypothesis: there is a statistically significant linear relationship.

The variables are aligned by date before testing.

### 6.3 Granger causality test

Granger causality is used to test whether past macroeconomic values contain predictive information for BIST 100 returns.

- Null hypothesis: the macro variable does not Granger-cause BIST 100 returns.
- Alternative hypothesis: the macro variable provides predictive information for BIST 100 returns.

This test is interpreted carefully. Granger causality does not prove true economic causality; it only tests whether one series helps predict another series.

## 7. Machine-learning methodology

The ML stage compares baseline market-only models with macro-enriched models.

### 7.1 Prediction target

The main task is classification. The target is next-day BIST 100 direction:

- `1`: next-day BIST 100 return is positive
- `0`: next-day BIST 100 return is zero or negative

This target is more appropriate than raw price prediction because raw stock prices are usually non-stationary. Direction prediction gives a clearer way to test whether macroeconomic variables add useful information.

A secondary regression task is also included to predict next-day return magnitude, but the classification task is the main focus.

### 7.2 Baseline feature set

The baseline feature set uses only historical market variables:

- Lagged BIST 100 returns
- Lagged GARAN returns
- BIST 100 rolling mean
- BIST 100 rolling volatility

### 7.3 Macro-enriched feature set

The macro-enriched feature set adds lagged macroeconomic variables:

- Lagged USD/TRY returns
- Lagged CPI changes
- Lagged policy-rate changes

This setup directly tests whether TCMB EVDS variables add predictive value beyond market history.

### 7.4 Models

Classification models:

- Dummy Classifier
- Logistic Regression
- Random Forest Classifier
- Gradient Boosting Classifier

Secondary regression models:

- Ridge Regression
- Random Forest Regressor

The Dummy Classifier is included as a simple benchmark. The other models are used to compare linear and nonlinear approaches.

### 7.5 Validation

Because the dataset is time-series data, the project uses a chronological train-test split. The first 80% of observations are used for training, and the last 20% are used for testing.

This avoids the main problem of random splitting in time-series work: future observations leaking into the training process.

### 7.6 Evaluation metrics

Classification metrics:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

Regression metrics:

- Mean Absolute Error
- Root Mean Squared Error
- R-squared

The most important comparison is baseline versus macro-enriched model performance, especially F1-score and ROC-AUC.

## 8. Expected contribution

The contribution of this project is the comparison itself. Instead of only applying ML to stock prices, the project checks whether macroeconomic enrichment changes the quality of the analysis and prediction.

If macro-enriched models perform better, this suggests that TCMB EVDS variables contain useful predictive information for BIST 100 daily direction. If they do not perform better, that result is still useful because it shows that these macro variables may not improve short-term prediction in this dataset.

## 9. Limitations

The project has several limitations:

- Financial markets are noisy and affected by many unobserved factors.
- Macroeconomic variables may be released at lower frequencies than daily stock prices.
- Granger causality does not prove true causality.
- Short-term market direction is difficult to predict.
- Results may depend on the selected period, variables, and model setup.

## 10. Final pipeline

The final project pipeline is:

```text
Data collection -> EDA -> Hypothesis testing -> ML modeling -> Baseline vs macro-enriched comparison
```

This keeps the proposal, EDA, hypothesis testing, and ML implementation connected to the same research question.
