import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr
from statsmodels.tsa.stattools import adfuller, grangercausalitytests

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

PROJECT_ROOT = Path.cwd()
if PROJECT_ROOT.name == "notebooks":
    PROJECT_ROOT = PROJECT_ROOT.parent

DATA_DIR = PROJECT_ROOT / "data"
IMAGE_DIR = PROJECT_ROOT / "images"
RESULTS_DIR = PROJECT_ROOT / "results"

DATA_DIR.mkdir(exist_ok=True)
IMAGE_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

DATA_PATH = DATA_DIR / "enriched_bist_data.csv"
CLEAN_OUTPUT_PATH = DATA_DIR / "clean_enriched_bist_data.csv"

print("Project root:", PROJECT_ROOT)
print("Data path:", DATA_PATH)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        "data/enriched_bist_data.csv was not found. Place the enriched dataset in the data folder before running this notebook."
    )

df = pd.read_csv(DATA_PATH, index_col="Date", parse_dates=True)
df = df.sort_index()

required_columns = ["XU100", "GARAN", "CPI", "PolicyRate", "USD_TRY"]
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

print("Dataset loaded successfully")
print("Shape:", df.shape)
print("Date range:", df.index.min().date(), "to", df.index.max().date())
display(df.head())
display(df.tail())

display(df[required_columns].describe().round(4))
missing_summary = pd.DataFrame({
    "missing_count": df[required_columns].isnull().sum(),
    "missing_percent": (df[required_columns].isnull().sum() / len(df) * 100).round(2)
}).sort_values("missing_count", ascending=False)
display(missing_summary)

plt.figure(figsize=(9, 5))
missing_summary["missing_count"].plot(kind="bar")
plt.title("Missing Values by Variable")
plt.ylabel("Missing Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(IMAGE_DIR / "06_missing_values.png", dpi=300, bbox_inches="tight")
plt.show()

df["XU100_Return"] = df["XU100"].pct_change()
df["GARAN_Return"] = df["GARAN"].pct_change()
df["USD_TRY_Return"] = df["USD_TRY"].pct_change()
df["CPI_Diff"] = df["CPI"].diff()
df["PolicyRate_Diff"] = df["PolicyRate"].diff()

transformed_columns = ["XU100_Return", "GARAN_Return", "USD_TRY_Return", "CPI_Diff", "PolicyRate_Diff"]
display(df[transformed_columns].describe().round(6))

# 1. Market price time series
plt.figure(figsize=(14, 6))
plt.plot(df.index, df["XU100"], label="BIST 100")
plt.plot(df.index, df["GARAN"], label="GARAN")
plt.title("BIST 100 and GARAN Close Prices")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.legend()
plt.tight_layout()
plt.savefig(IMAGE_DIR / "01_time_series_prices.png", dpi=300, bbox_inches="tight")
plt.show()

# 2. Macro time series
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
axes[0].plot(df.index, df["CPI"]); axes[0].set_title("Consumer Price Index")
axes[1].plot(df.index, df["PolicyRate"]); axes[1].set_title("Policy Rate")
axes[2].plot(df.index, df["USD_TRY"]); axes[2].set_title("USD/TRY Exchange Rate")
plt.tight_layout()
plt.savefig(IMAGE_DIR / "02_macro_time_series.png", dpi=300, bbox_inches="tight")
plt.show()

# 3. Normalized comparison
normalized = df[["XU100", "GARAN", "CPI", "USD_TRY"]].dropna()
normalized = normalized / normalized.iloc[0] * 100
plt.figure(figsize=(14, 7))
for col in normalized.columns:
    plt.plot(normalized.index, normalized[col], label=col)
plt.title("Normalized Market and Macro Variables, First Observation = 100")
plt.ylabel("Normalized Index")
plt.legend()
plt.tight_layout()
plt.savefig(IMAGE_DIR / "03_normalized_price_macro_comparison.png", dpi=300, bbox_inches="tight")
plt.show()

# 4. Return distribution
plt.figure(figsize=(10, 6))
sns.histplot(df["XU100_Return"].dropna(), bins=60, kde=True)
plt.title("Distribution of BIST 100 Daily Returns")
plt.xlabel("Daily Return")
plt.tight_layout()
plt.savefig(IMAGE_DIR / "04_return_distribution.png", dpi=300, bbox_inches="tight")
plt.show()

# 5. Rolling volatility
df["XU100_RollingVol_20"] = df["XU100_Return"].rolling(window=20).std()
plt.figure(figsize=(14, 6))
plt.plot(df.index, df["XU100_RollingVol_20"])
plt.title("20-Day Rolling Volatility of BIST 100 Returns")
plt.ylabel("Rolling Volatility")
plt.tight_layout()
plt.savefig(IMAGE_DIR / "05_rolling_volatility.png", dpi=300, bbox_inches="tight")
plt.show()

# 6. Correlation heatmap
corr_data = df[transformed_columns].dropna()
corr_matrix = corr_data.corr()
plt.figure(figsize=(9, 7))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, cmap="coolwarm", center=0, fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap of Returns and Macro Changes")
plt.tight_layout()
plt.savefig(IMAGE_DIR / "07_correlation_heatmap_transformed.png", dpi=300, bbox_inches="tight")
plt.show()
display(corr_matrix.round(4))

# 7. Scatter plots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, macro in zip(axes, ["USD_TRY_Return", "CPI_Diff", "PolicyRate_Diff"]):
    plot_data = df[["XU100_Return", macro]].dropna()
    sns.regplot(data=plot_data, x=macro, y="XU100_Return", ax=ax, scatter_kws={"alpha": 0.4}, line_kws={"color": "black"})
    ax.set_title(f"XU100 Return vs {macro}")
plt.tight_layout()
plt.savefig(IMAGE_DIR / "08_scatter_macro_vs_returns.png", dpi=300, bbox_inches="tight")
plt.show()

# 8. Yearly return boxplot
df["Year"] = df.index.year
plt.figure(figsize=(14, 6))
sns.boxplot(data=df.dropna(subset=["XU100_Return"]), x="Year", y="XU100_Return")
plt.title("BIST 100 Daily Return Distribution by Year")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(IMAGE_DIR / "09_yearly_return_boxplot.png", dpi=300, bbox_inches="tight")
plt.show()

def adf_test(series, name):
    clean_series = series.dropna()
    if len(clean_series) < 20:
        return {"series": name, "adf_statistic": np.nan, "p_value": np.nan, "stationary_at_5pct": False, "comment": "Not enough observations"}
    result = adfuller(clean_series)
    p_value = result[1]
    return {"series": name, "adf_statistic": result[0], "p_value": p_value, "stationary_at_5pct": p_value < 0.05, "comment": "Stationary" if p_value < 0.05 else "Non-stationary"}

adf_results_df = pd.DataFrame([adf_test(df[col], col) for col in transformed_columns])
display(adf_results_df.round(5))
adf_results_df.to_csv(DATA_DIR / "adf_test_results.csv", index=False)

macro_test_vars = ["USD_TRY_Return", "CPI_Diff", "PolicyRate_Diff"]
pearson_results = []
for macro in macro_test_vars:
    test_data = df[["XU100_Return", macro]].dropna()
    corr_stat, p_value = pearsonr(test_data["XU100_Return"], test_data[macro])
    pearson_results.append({"macro_variable": macro, "correlation": corr_stat, "p_value": p_value, "significant_at_5pct": p_value < 0.05, "interpretation": "Significant linear relationship" if p_value < 0.05 else "No significant linear relationship"})
pearson_results_df = pd.DataFrame(pearson_results)
display(pearson_results_df.round(5))
pearson_results_df.to_csv(DATA_DIR / "pearson_correlation_results.csv", index=False)

max_lag = 5
granger_results = []
for macro in macro_test_vars:
    test_data = df[["XU100_Return", macro]].dropna()
    if len(test_data) <= max_lag + 10:
        granger_results.append({"macro_variable": macro, "best_lag": np.nan, "min_p_value": np.nan, "significant_at_5pct": False, "interpretation": "Not enough observations"})
        continue
    try:
        result = grangercausalitytests(test_data[["XU100_Return", macro]], maxlag=max_lag, verbose=False)
        lag_p_values = {lag: result[lag][0]["ssr_ftest"][1] for lag in range(1, max_lag + 1)}
        best_lag = min(lag_p_values, key=lag_p_values.get)
        min_p_value = lag_p_values[best_lag]
        granger_results.append({"macro_variable": macro, "best_lag": best_lag, "min_p_value": min_p_value, "significant_at_5pct": min_p_value < 0.05, "interpretation": "Predictive information detected" if min_p_value < 0.05 else "No statistically significant predictive information"})
    except Exception as error:
        granger_results.append({"macro_variable": macro, "best_lag": np.nan, "min_p_value": np.nan, "significant_at_5pct": False, "interpretation": f"Test failed: {error}"})

granger_results_df = pd.DataFrame(granger_results)
display(granger_results_df.round(5))
granger_results_df.to_csv(DATA_DIR / "granger_test_results.csv", index=False)

df.to_csv(CLEAN_OUTPUT_PATH)
print("Clean dataset saved to:", CLEAN_OUTPUT_PATH)