# Data folder

Place the enriched dataset in this folder before running the notebooks.

Expected file:

```text
data/enriched_bist_data.csv
```

Required columns:

```text
Date, XU100, GARAN, CPI, PolicyRate, USD_TRY
```

The EDA notebook creates `clean_enriched_bist_data.csv`, which is then used by the ML notebook if available.
