# DSA 210 Project Proposal

**Student:** Gökberk Marmara  
**Course:** DSA 210  
**Project Title:** Macroeconomic Enrichment for BIST 100 Market Movement Analysis and Prediction

---

## 1. Project Motivation

Stock price prediction is a very common project topic. To make this project more specific, I focus on whether macroeconomic enrichment improves the analysis and prediction of BIST 100 market movements. Instead of treating the stock market as an isolated price series, I combine market data with Turkish macroeconomic indicators.

The motivation is that the Turkish stock market may be affected by inflation, monetary policy, and exchange-rate movements. Therefore, this project asks whether adding macroeconomic indicators from TCMB EVDS provides extra explanatory or predictive value beyond historical market prices alone.

This makes the project more than a generic stock prediction task. The main focus is the comparison between a market-only approach and a macroeconomically enriched approach.

---

## 2. Research Question

The main research question is:

> Do Turkish macroeconomic indicators such as CPI, policy rate, and USD/TRY exchange-rate movements provide useful explanatory or predictive information for BIST 100 daily returns?

A secondary machine-learning question is:

> Do macro-enriched models perform better than baseline models that use only historical market features?

---

## 3. Data Sources

The project uses two data sources.

### 3.1 Financial Market Data

Financial market data is collected from Yahoo Finance through `yfinance`.

Main variables:

- BIST 100 close price (`XU100`)
- GARAN close price (`GARAN`)

BIST 100 is the main market index of the project. GARAN is included as an additional market-related stock variable because it is a major Turkish bank and may reflect domestic financial-sector behavior.

### 3.2 Macroeconomic Enrichment Data

Macroeconomic indicators are collected from TCMB EVDS through `evdspy`.

Main variables:

- Consumer Price Index (`CPI`)
- Policy interest rate (`PolicyRate`)
- USD/TRY exchange rate (`USD_TRY`)

These variables represent inflation, monetary policy, and exchange-rate pressure. They are used to enrich the market dataset and test whether macroeconomic context improves the analysis.

---

## 4. Dataset Construction and Feature Enrichment

The enriched dataset combines financial market variables and macroeconomic variables by date. The expected main dataset is:

```text
data/enriched_bist_data.csv
```

Required columns:

```text
Date, XU100, GARAN, CPI, PolicyRate, USD_TRY
```

Before statistical testing and machine learning, raw level variables are transformed into returns or first differences:

- `XU100_Return`: daily percentage return of BIST 100
- `GARAN_Return`: daily percentage return of GARAN
- `USD_TRY_Return`: daily percentage change in USD/TRY
- `CPI_Diff`: first difference of CPI
- `PolicyRate_Diff`: first difference of policy rate

These transformations are important because financial and macroeconomic level series often contain trends. Working with changes instead of raw levels reduces the risk of misleading correlations.

---

## 5. Exploratory Data Analysis Plan

The EDA stage will examine both raw and transformed variables. The goal is to understand the dataset before formal statistical testing and ML modeling.

Planned EDA includes:

- Summary statistics
- Missing value analysis
- Time-series plots of BIST 100, GARAN, CPI, policy rate, and USD/TRY
- Normalized comparison of market and macro variables
- BIST 100 return distribution
- Rolling volatility of BIST 100 returns
- Correlation heatmap using transformed variables
- Scatter plots between macroeconomic changes and BIST 100 returns
- Yearly distribution of BIST 100 returns

The EDA will include written commentary explaining what each visualization shows and how it connects to the research question.

---

## 6. Hypothesis Testing Plan

The project uses three types of statistical tests.

### 6.1 ADF Stationarity Test

The Augmented Dickey-Fuller test checks whether the transformed time-series variables are stationary.

- Null hypothesis: the series is non-stationary.
- Alternative hypothesis: the series is stationary.

Stationarity matters because several time-series methods can be misleading when variables have strong trends.

### 6.2 Pearson Correlation Test

Pearson correlation tests whether transformed macroeconomic variables have statistically significant linear relationships with BIST 100 daily returns.

- Null hypothesis: there is no statistically significant linear relationship.
- Alternative hypothesis: there is a statistically significant linear relationship.

The variables will be aligned by date before testing.

### 6.3 Granger Causality Test

Granger causality tests whether past values of macroeconomic variables contain predictive information for BIST 100 returns.

- Null hypothesis: the macro variable does not Granger-cause BIST 100 returns.
- Alternative hypothesis: the macro variable provides predictive information for BIST 100 returns.

Granger causality does not prove true economic causality. It only tests whether past values of one variable help predict another variable.

---

## 7. Machine Learning Methodology

The machine-learning stage compares baseline market-only models with macro-enriched models.

### 7.1 Prediction Target

The main ML task is classification. The target variable is next-day BIST 100 market direction:

- `1`: next-day BIST 100 return is positive
- `0`: next-day BIST 100 return is zero or negative

This target is preferred over raw price prediction because raw stock prices are often non-stationary. Direction prediction is more appropriate for testing whether macroeconomic variables add predictive information.

A secondary regression task will also be explored to predict next-day return magnitude.

### 7.2 Baseline Feature Set

The baseline feature set uses only market-based historical variables:

- Lagged BIST 100 returns
- Lagged GARAN returns
- BIST 100 rolling mean
- BIST 100 rolling volatility

### 7.3 Macro-Enriched Feature Set

The macro-enriched feature set adds lagged macroeconomic variables:

- Lagged USD/TRY returns
- Lagged CPI changes
- Lagged policy rate changes

This setup directly tests whether TCMB EVDS macro indicators add predictive value beyond historical market data.

### 7.4 Models

Classification models:

- Dummy Classifier
- Logistic Regression
- Random Forest Classifier
- Gradient Boosting Classifier

Secondary regression models:

- Ridge Regression
- Random Forest Regressor

### 7.5 Validation Strategy

Because the dataset is time-series data, the project uses a chronological train-test split instead of random splitting. The first 80% of observations are used for training, and the last 20% are used for testing.

This prevents future observations from leaking into the training process.

### 7.6 Evaluation Metrics

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

The main comparison is whether macro-enriched models improve F1-score or ROC-AUC compared with baseline models.

---

## 8. Expected Contribution

The expected contribution of the project is not to claim perfect stock-market prediction. Financial markets are noisy and difficult to predict.

Instead, the project contributes by testing whether Turkish macroeconomic indicators provide additional explanatory or predictive information for BIST 100 market behavior. Even if the enriched models do not improve performance, that result is still meaningful because it shows that macro variables may not strongly improve short-term prediction in this dataset.

---

## 9. Limitations

The project has several limitations:

- Financial markets are noisy and affected by many unobserved factors.
- Macroeconomic variables may operate at different frequencies than daily stock prices.
- Granger causality does not prove true causality.
- Short-term market direction may be difficult to predict even with enriched features.
- Results may depend on the selected time period and variables.

---

## 10. Project Pipeline

The final project pipeline is:

```text
Data Collection -> EDA -> Hypothesis Testing -> ML Modeling -> Baseline vs Macro-Enriched Comparison
```

This structure keeps the proposal, EDA, hypothesis testing, and machine-learning implementation aligned around the same research question.
