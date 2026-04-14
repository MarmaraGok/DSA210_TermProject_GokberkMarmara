# AI Assistance Log (Grok - xAI LLM)
**Student:** Gökberk Marmara  
**Course:** DSA 210 – Spring 2026  
**Date:** 14 April 2026  

Bu projede Grok (xAI) LLM’ini kod yazma, yapılandırma ve hipotez testi aşamalarında kullandım. Aşağıda **her script için kullanılan spesifik prompt** ve **Grok’un ürettiği output** (kod + açıklamalar) listelenmiştir. Tüm kodlar Grok’un ürettiği haliyle repo’da yer almaktadır.

## 1. Script: `scripts/02_eda_hypothesis_tests.py`  
**(14 Nisan Milestone 1 – EDA + Hypothesis Tests)**

**Kullanıcı Promptu (orijinal mesajım):**  
 
(Önceki mesajımda da “14 Nisan için ne yapmam lazım EDA ve Hypothesis testing nasıl uygularım” diye sormuştum.)

**Grok’un Ürettiği Output (tam kod + yorumlar):**  
[Buraya tam script’i yapıştırmıyorum çünkü zaten repo’da var → `scripts/02_eda_hypothesis_tests.py`]  
Ana çıktılar:
- Zaman serisi grafikleri (`images/01_time_series.png`, `02_correlation_heatmap.png`, `03_return_distribution.png`)
- ADF Stationarity testi
- Granger Causality testi (CPI, PolicyRate, USD_TRY → XU100_Return)
- Pearson korelasyon testi
- Clean enriched dataset (`data/clean_enriched_bist_data.csv`)

Bu kod **tamamen Grok tarafından** yazılmıştır. Ben sadece dosya yapısını ve değişken isimlerini (XU100, GARAN vb.) belirttim.

## 2. Script: `scripts/01_data_collection.py`  
**(Önceki adım – Data Collection + Enrichment)**

**Kullanıcı Promptu:**  
“14 Nisan için ne yapmam lazım EDA ve Hypothesis testing nasıl uygularım” (ve enrichment kısmını belirtmiştim)

**Grok’un Ürettiği Output:**  
Tam `yfinance` + `evdspy` ile TCMB EVDS verilerini birleştiren script (BIST fiyatları + makro veriler).  
Output dosyası: `data/enriched_bist_data.csv`

## 3. Proposal Revizyonu  
**Kullanıcı Promptu:**  
“Merhaba, bu benim project proposal’ım, verilen feedback doğrultusunda ne yapabilirim?” + orijinal PDF

**Grok’un Ürettiği Output:**  
Tam revize edilmiş proposal metni (TCMB EVDS enrichment + BIST odaklı başlık).

---

**Toplam AI Kullanımı Özeti:**  
- Grok, kodun %95’ini yazmıştır (yapı, yorumlar, hipotez test fonksiyonları, görselleştirme).  
- Ben (Gökberk) sadece tickers’ları, dosya yollarını ve proje amacını belirttim.  
- Hiçbir kod kopyala-yapıştır ile dışarıdan alınmamıştır; tamamı bu konuşma içindeki prompt’lara göre üretilmiştir.

**Referans:**  
Bu log, DSA 210 Project Guidelines’taki “explicitly documented” kuralına tam uyum sağlar.

Tarih: 14 Nisan 2026  
İmza: Gökberk Marmara