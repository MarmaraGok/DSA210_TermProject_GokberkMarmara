import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, mean_absolute_error, mean_squared_error, precision_score, r2_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

PROJECT_ROOT = Path.cwd()
if PROJECT_ROOT.name == "notebooks":
    PROJECT_ROOT = PROJECT_ROOT.parent
DATA_DIR = PROJECT_ROOT / "data"
IMAGE_DIR = PROJECT_ROOT / "images"
RESULTS_DIR = PROJECT_ROOT / "results"
DATA_DIR.mkdir(exist_ok=True)
IMAGE_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

clean_path = DATA_DIR / "clean_enriched_bist_data.csv"
raw_path = DATA_DIR / "enriched_bist_data.csv"
if clean_path.exists():
    data_path = clean_path
elif raw_path.exists():
    data_path = raw_path
else:
    raise FileNotFoundError("No dataset found. Run EDA first or place enriched_bist_data.csv under data/.")

df = pd.read_csv(data_path, index_col="Date", parse_dates=True).sort_index()
required_columns = ["XU100", "GARAN", "CPI", "PolicyRate", "USD_TRY"]
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")
print("Dataset loaded from:", data_path)
print("Shape:", df.shape)
display(df.head())

df["XU100_Return"] = df["XU100"].pct_change()
df["GARAN_Return"] = df["GARAN"].pct_change()
df["USD_TRY_Return"] = df["USD_TRY"].pct_change()
df["CPI_Diff"] = df["CPI"].diff()
df["PolicyRate_Diff"] = df["PolicyRate"].diff()

for lag in [1, 2, 3, 5]:
    df[f"XU100_Return_Lag{lag}"] = df["XU100_Return"].shift(lag)
    df[f"GARAN_Return_Lag{lag}"] = df["GARAN_Return"].shift(lag)

for window in [5, 10]:
    df[f"XU100_RollingMean_{window}"] = df["XU100_Return"].rolling(window=window).mean()
    df[f"XU100_RollingStd_{window}"] = df["XU100_Return"].rolling(window=window).std()

macro_base_features = ["USD_TRY_Return", "CPI_Diff", "PolicyRate_Diff"]
for feature in macro_base_features:
    for lag in [1, 2, 3, 5]:
        df[f"{feature}_Lag{lag}"] = df[feature].shift(lag)

df["Target_Return_NextDay"] = df["XU100_Return"].shift(-1)
df["Target_Direction_NextDay"] = (df["Target_Return_NextDay"] > 0).astype(int)

display(df.tail())

baseline_features = [
    "XU100_Return_Lag1", "XU100_Return_Lag2", "XU100_Return_Lag3", "XU100_Return_Lag5",
    "GARAN_Return_Lag1", "GARAN_Return_Lag2", "GARAN_Return_Lag3", "GARAN_Return_Lag5",
    "XU100_RollingMean_5", "XU100_RollingStd_5", "XU100_RollingMean_10", "XU100_RollingStd_10"
]
macro_features = [
    "USD_TRY_Return_Lag1", "USD_TRY_Return_Lag2", "USD_TRY_Return_Lag3", "USD_TRY_Return_Lag5",
    "CPI_Diff_Lag1", "CPI_Diff_Lag2", "CPI_Diff_Lag3", "CPI_Diff_Lag5",
    "PolicyRate_Diff_Lag1", "PolicyRate_Diff_Lag2", "PolicyRate_Diff_Lag3", "PolicyRate_Diff_Lag5"
]
enriched_features = baseline_features + macro_features
model_df = df[baseline_features + macro_features + ["Target_Return_NextDay", "Target_Direction_NextDay"]].dropna()

print("Modeling dataset shape:", model_df.shape)
display(model_df["Target_Direction_NextDay"].value_counts(normalize=True).rename("proportion").to_frame())

split_index = int(len(model_df) * 0.8)
train_df = model_df.iloc[:split_index].copy()
test_df = model_df.iloc[split_index:].copy()

X_train_baseline = train_df[baseline_features]
X_test_baseline = test_df[baseline_features]
X_train_enriched = train_df[enriched_features]
X_test_enriched = test_df[enriched_features]
y_train_class = train_df["Target_Direction_NextDay"]
y_test_class = test_df["Target_Direction_NextDay"]
y_train_reg = train_df["Target_Return_NextDay"]
y_test_reg = test_df["Target_Return_NextDay"]

print("Train period:", train_df.index.min().date(), "to", train_df.index.max().date())
print("Test period:", test_df.index.min().date(), "to", test_df.index.max().date())
print("Train rows:", len(train_df), "Test rows:", len(test_df))

classification_models = {
    "Dummy Classifier": DummyClassifier(strategy="most_frequent"),
    "Logistic Regression": Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))]),
    "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_leaf=5, random_state=42, class_weight="balanced"),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, learning_rate=0.03, max_depth=3, random_state=42)
}

def evaluate_classifier(model_name, feature_set_name, model, X_train, X_test):
    model.fit(X_train, y_train_class)
    predictions = model.predict(X_test)
    try:
        probabilities = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test_class, probabilities)
    except Exception:
        roc_auc = np.nan
    metrics = {
        "model": model_name,
        "feature_set": feature_set_name,
        "accuracy": accuracy_score(y_test_class, predictions),
        "precision": precision_score(y_test_class, predictions, zero_division=0),
        "recall": recall_score(y_test_class, predictions, zero_division=0),
        "f1_score": f1_score(y_test_class, predictions, zero_division=0),
        "roc_auc": roc_auc
    }
    print("\n" + "="*80)
    print(f"{model_name} | {feature_set_name}")
    display(pd.DataFrame([metrics]).round(4))
    print(classification_report(y_test_class, predictions, zero_division=0))
    cm = confusion_matrix(y_test_class, predictions)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Down/Flat","Up"], yticklabels=["Down/Flat","Up"])
    plt.title(f"Confusion Matrix - {model_name} ({feature_set_name})")
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.tight_layout()
    safe_name = f"{model_name}_{feature_set_name}".replace(" ", "_").replace("/", "_")
    plt.savefig(IMAGE_DIR / f"confusion_matrix_{safe_name}.png", dpi=300, bbox_inches="tight")
    plt.show()
    return metrics, model

classification_results = []
trained_models = {}
for model_name, model in classification_models.items():
    metrics, trained_model = evaluate_classifier(model_name, "Baseline", model, X_train_baseline, X_test_baseline)
    classification_results.append(metrics)
    trained_models[(model_name, "Baseline")] = trained_model
for model_name, model in classification_models.items():
    metrics, trained_model = evaluate_classifier(model_name, "Macro-Enriched", model, X_train_enriched, X_test_enriched)
    classification_results.append(metrics)
    trained_models[(model_name, "Macro-Enriched")] = trained_model

classification_results_df = pd.DataFrame(classification_results)
display(classification_results_df.round(4))
classification_results_df.to_csv(RESULTS_DIR / "classification_model_results.csv", index=False)

comparison = classification_results_df.pivot(index="model", columns="feature_set", values=["accuracy", "f1_score", "roc_auc"])
display(comparison.round(4))
comparison.to_csv(RESULTS_DIR / "baseline_vs_enriched_classification_comparison.csv")

for metric in ["f1_score", "roc_auc"]:
    table = classification_results_df.pivot(index="model", columns="feature_set", values=metric)
    table.plot(kind="bar", figsize=(10,6))
    plt.title(f"Baseline vs Macro-Enriched - {metric}")
    plt.ylabel(metric)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / f"baseline_vs_enriched_{metric}.png", dpi=300, bbox_inches="tight")
    plt.show()

rf_enriched = trained_models.get(("Random Forest", "Macro-Enriched"))
if rf_enriched is not None and hasattr(rf_enriched, "feature_importances_"):
    importances = pd.DataFrame({"feature": enriched_features, "importance": rf_enriched.feature_importances_}).sort_values("importance", ascending=False)
    display(importances.head(15).round(5))
    importances.to_csv(RESULTS_DIR / "feature_importance_random_forest_enriched.csv", index=False)
    plt.figure(figsize=(10, 8))
    top_features = importances.head(15).sort_values("importance", ascending=True)
    plt.barh(top_features["feature"], top_features["importance"])
    plt.title("Top 15 Feature Importances - Random Forest Macro-Enriched Model")
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "feature_importance_random_forest_enriched.png", dpi=300, bbox_inches="tight")
    plt.show()

regression_models = {
    "Ridge Regression": Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))]),
    "Random Forest Regressor": RandomForestRegressor(n_estimators=300, max_depth=5, min_samples_leaf=5, random_state=42)
}

def evaluate_regressor(model_name, feature_set_name, model, X_train, X_test):
    model.fit(X_train, y_train_reg)
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test_reg, predictions))
    metrics = {"model": model_name, "feature_set": feature_set_name, "mae": mean_absolute_error(y_test_reg, predictions), "rmse": rmse, "r2_score": r2_score(y_test_reg, predictions)}
    pred_df = pd.DataFrame({"Actual_Return": y_test_reg, "Predicted_Return": predictions}, index=y_test_reg.index)
    safe_name = f"{model_name}_{feature_set_name}".replace(" ", "_").replace("/", "_")
    pred_df.to_csv(RESULTS_DIR / f"regression_predictions_{safe_name}.csv")
    plt.figure(figsize=(12,5))
    plt.plot(pred_df.index, pred_df["Actual_Return"], label="Actual")
    plt.plot(pred_df.index, pred_df["Predicted_Return"], label="Predicted")
    plt.title(f"Actual vs Predicted Returns - {model_name} ({feature_set_name})")
    plt.legend(); plt.tight_layout()
    plt.savefig(IMAGE_DIR / f"actual_vs_predicted_{safe_name}.png", dpi=300, bbox_inches="tight")
    plt.show()
    return metrics

regression_results = []
for model_name, model in regression_models.items():
    regression_results.append(evaluate_regressor(model_name, "Baseline", model, X_train_baseline, X_test_baseline))
for model_name, model in regression_models.items():
    regression_results.append(evaluate_regressor(model_name, "Macro-Enriched", model, X_train_enriched, X_test_enriched))
regression_results_df = pd.DataFrame(regression_results)
display(regression_results_df.round(6))
regression_results_df.to_csv(RESULTS_DIR / "regression_model_results.csv", index=False)