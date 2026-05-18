# Code Explanation Guide

This guide summarizes the logic of the project so each code step can be explained clearly.

## Why use an enriched dataset?

The project is not only about BIST 100 historical prices. It tests whether macroeconomic indicators add useful information. That is why the dataset combines market data with CPI, policy rate, and USD/TRY.

## Why transform variables?

Raw price and macroeconomic level series often contain trends. Using trending levels directly can create misleading correlations. Returns and first differences focus on changes instead of levels.

## Why use ADF?

ADF tests stationarity. The null hypothesis is that the series is non-stationary. If p-value is below 0.05, the series is treated as stationary.

## Why use Pearson correlation?

Pearson tests linear association between BIST 100 returns and macroeconomic changes. It does not prove causality.

## Why use Granger causality?

Granger causality tests whether past macroeconomic values help predict BIST 100 returns. It does not prove true economic causality.

## Why use a time-based train-test split?

The dataset is time-series data. Random splitting can leak future information into training. A chronological split prevents this.

## Why compare baseline vs macro-enriched models?

This comparison directly answers the research question. If macro-enriched models perform better, that suggests TCMB EVDS variables provide additional predictive information.
