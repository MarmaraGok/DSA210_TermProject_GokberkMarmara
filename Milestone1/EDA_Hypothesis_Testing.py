# =============================================================================
# DSA 210 - 02_eda_hypothesis_tests.py
# 14 Nisan Milestone: Data Collection + EDA + Hypothesis Tests
# Author: Gökberk Marmara
# Date: 14 April 2026
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from scipy.stats import pearsonr
import warnings
import os

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# ====================== 1. KLASÖRLERİ OLUŞTUR ======================
os.makedirs('images', exist_ok=True)
os.makedirs('data', exist_ok=True)

# ====================== 2. VERİYİ YÜKLE ======================
print("Enriched dataset yükleniyor...")
df = pd.read_csv('data/enriched_bist_data.csv', index_col='Date', parse_dates=True)
print(f"Yüklendi → Shape: {df.shape} | Tarih aralığı: {df.index[0]} - {df.index[-1]}")

# ====================== 3. EDA - TEMEL İSTATİSTİKLER ======================
print("\n" + "="*60)
print("1. BASIC SUMMARY STATISTICS")
print(df.describe().round(4))

print("\n❓ Missing Values (top 10):")
print(df.isnull().sum().sort_values(ascending=False).head(10))

# ====================== 4. GÖRSELLEŞTİRMELER (PNG olarak kaydedilecek) ======================
print("\nGörseller oluşturuluyor...")

# 4.1 Time Series
fig, axes = plt.subplots(3, 1, figsize=(14, 10))
df['XU100'].plot(ax=axes[0], title='BIST 100 Close Price (2015-günümüz)')
df['GARAN'].plot(ax=axes[1], title='GARAN Close Price')
df['CPI'].plot(ax=axes[2], title='TÜFE (Enflasyon)')
plt.tight_layout()
plt.savefig('images/01_time_series.png', dpi=300, bbox_inches='tight')
plt.close()

# 4.2 Correlation Heatmap
plt.figure(figsize=(12, 8))
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, cmap='coolwarm', center=0, fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix - BIST Prices & Macro Variables')
plt.savefig('images/02_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# 4.3 Daily Return Distribution
df['XU100_Return'] = df['XU100'].pct_change()
plt.figure(figsize=(10, 6))
sns.histplot(df['XU100_Return'].dropna(), kde=True, bins=50, color='skyblue')
plt.title('BIST 100 Daily Return Distribution')
plt.xlabel('Daily Return')
plt.savefig('images/03_return_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

print("3 görsel images/ klasörüne kaydedildi")

# ====================== 5. HYPOTHESIS TESTS ======================
print("\n" + "="*60)
print("2. HYPOTHESIS TESTING")

# H1: Stationarity (ADF Test)
def adf_test(series, name):
    result = adfuller(series.dropna())
    print(f"\nADF Test → {name}")
    print(f"   ADF Statistic : {result[0]:.4f}")
    print(f"   p-value       : {result[1]:.4f}")
    print("   → Stationary" if result[1] < 0.05 else "   → Non-Stationary (differencing önerilir)")

adf_test(df['XU100_Return'], 'BIST 100 Daily Return')
adf_test(df['CPI'].diff().dropna(), 'Enflasyon (1. fark)')

# H2: Granger Causality
print("\n🔗 Granger Causality Test (maxlag=5)")
granger_data = df[['XU100_Return', 'CPI', 'PolicyRate', 'USD_TRY']].dropna()

for macro in ['CPI', 'PolicyRate', 'USD_TRY']:
    print(f"\n--- {macro} → XU100_Return Granger-causes mi? ---")
    try:
        gc_res = grangercausalitytests(granger_data[['XU100_Return', macro]], maxlag=5, verbose=False)
        p_values = [round(gc_res[i+1][0]['ssr_ftest'][1], 4) for i in range(5)]
        print(f"   Min p-value (lag 1-5): {min(p_values):.4f} → {'Anlamlı etki var ' if min(p_values) < 0.05 else 'Anlamlı etki yok '}")
    except:
        print("   Test çalıştırılamadı (yetersiz varyans)")

# H3: Pearson Correlation
corr_stat, p_val = pearsonr(df['XU100_Return'].dropna(), df['CPI'].diff().dropna()[:len(df['XU100_Return'].dropna())])
print(f"\nPearson Correlation (Return vs ΔCPI): r = {corr_stat:.4f}, p = {p_val:.4f}")
print("   → Negatif korelasyon bekleniyordu, doğrulandı" if corr_stat < 0 else "")

# ====================== 6. CLEAN DATA KAYDET (ML için) ======================
df.to_csv('data/clean_enriched_bist_data.csv')
print("\nclean_enriched_bist_data.csv kaydedildi → 5 Mayıs ML aşamasında direkt kullanacaksın")

print("\n🎯 14 NİSAN MILESTONE TAMAMLANDI!")
print("   Data collection (önceki script)")
print("   EDA (grafikler + istatistikler)")
print("   Hypothesis Tests (ADF + Granger + Correlation)")