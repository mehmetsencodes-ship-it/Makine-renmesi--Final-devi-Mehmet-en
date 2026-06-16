# HIGGS Machine Learning Pipeline
### Feature Selection & Hyperparameter Optimization

Makine Ogrenmesi Final Odevi - HIGGS veri seti uzerinde ozellik secimi ve
hiperparametre optimizasyonu iceren tam bir makine ogrenmesi hatti (pipeline).

---

## Proje Ozeti

Bu proje, HIGGS veri setinden (11M ornek, 28 ozellik) **rastgele 100.000 ornek**
secerek asagidaki adimlari uygular:

1. **Veri On Isleme**
   - IQR yontemi ile aykiri deger analizi
   - Aykiri degerlerin sinir degerlere kirpilmasi (winsorization)
   - `MinMaxScaler` ile [0, 1] araligina olcekleme

2. **Ozellik Secimi (Filter-Based)**
   - ANOVA F-score *veya* Mutual Information ile en iyi 15 ozellik

3. **Modelleme ve Degerlendirme (Nested Cross-Validation)**
   - Outer Loop: 5-fold, Inner Loop: 3-fold
   - **Flowchart A:** ic dongude farkli oznitelik secim kombinasyonlari
   - **Flowchart B:** ic dongude farkli hiperparametre kombinasyonlari
   - Modeller: **KNN, SVM, MLP, XGBoost**
   - Metrikler: Accuracy, Precision, Recall, F1, ROC-AUC
   - OVA yontemiyle ROC egrileri

---

## Klasor Yapisi

```
.
├── config.py                # Tum ayarlar (yollar, CV, hiperparametreler)
├── main.py                  # Ana calistirma betigi
├── requirements.txt
├── README.md
├── higgs/
│   └── HIGGS.csv.gz         # Veri seti (git'e dahil edilmez)
├── src/
│   ├── data_loader.py       # Rastgele 100k ornekleme
│   ├── preprocessing.py     # IQR + MinMaxScaler
│   ├── feature_selection.py # ANOVA / Mutual Information
│   ├── models.py            # KNN, SVM, MLP, XGBoost pipeline'lari
│   ├── nested_cv.py         # Nested cross-validation (Flowchart A/B)
│   └── visualization.py     # ROC egrileri ve metrik grafikleri
└── outputs/
    ├── figures/             # Grafik ciktilari (.png)
    └── results/             # Metrik tablolari (.csv)
```

---

## Veri Seti

Buyuk HIGGS veri seti (~2.8 GB) repoya dahil **edilmemistir**. Iki secenek vardir:

1. **Veri setini indirmeden calistirma (kolay):** Repoda yer alan
   `outputs/higgs_sample_100000.parquet` onbellek dosyasi (~9 MB), daha onceden
   secilmis 100.000 orneklik alt kumeyi icerir. `data_loader.py` bu onbellegi
   otomatik kullandigi icin **buyuk dosya olmadan** notebook/`main.py` dogrudan calisir.

2. **Tam veri setiyle calistirma:** Tum veriden yeniden orneklemek isterseniz
   HIGGS veri setini indirip `higgs/HIGGS.csv.gz` konumuna yerlestirin:
   - UCI: https://archive.ics.uci.edu/dataset/280/higgs
   - Dogrudan: https://archive.ics.uci.edu/static/public/280/higgs.zip
   (Indirilen `HIGGS.csv.gz` dosyasini `higgs/` klasorune koyun.)

---

## Kurulum

> **Not:** Sisteminizde Python kurulu olmalidir (3.9+ onerilir).
> Microsoft Store kisayolu degil, gercek bir Python kurulumu gereklidir.
> https://www.python.org/downloads/ adresinden indirebilirsiniz
> (kurulumda "Add Python to PATH" kutusunu isaretleyin).

```bash
# 1) Sanal ortam olustur (onerilir)
python -m venv .venv
.venv\Scripts\activate        # Windows PowerShell

# 2) Bagimliliklari kur
pip install -r requirements.txt
```

---

## Calistirma

```bash
# Tum pipeline (Flowchart A + B)
python main.py

# Yalnizca bir flowchart
python main.py --flow A
python main.py --flow B

# Modelleme alt-ornek boyutunu degistir
python main.py --sample 10000

# Ozellik secim yontemini degistir
python main.py --method mutual_info
```

### Performans Notu

SVM (rbf) ve MLP modellerinin nested CV'si 100.000 ornekte cok pahalidir
(SVM egitimi yaklasik O(n^2)). Bu yuzden **on isleme ve ozellik secimi tum
100.000 ornek** uzerinde yapilir; **nested CV modelleme adimi** ise
`config.MODEL_SAMPLE_SIZE` (varsayilan 6.000) boyutunda dengeli bir
alt-ornek uzerinde calisir. Daha guclu bir makinede bu degeri buyutebilir
veya `None` yaparak tum 100.000 ornegi kullanabilirsiniz.

---

## Ciktilar

Calistirma sonunda `outputs/` altinda olusur:

- `results/outlier_report_iqr.csv` — IQR aykiri deger raporu
- `results/feature_scores_*.csv` — ozellik skorlari
- `results/metrics_summary_flow_*.csv` — model metrik ozet tablolari
- `results/fold_metrics_*_flow_*.csv` — fold bazli metrikler
- `results/best_params_*_flow_*.csv` — secilen en iyi parametreler
- `figures/boxplots_*.png` — aykiri deger box-plot'lari
- `figures/feature_importance_scores.png` — ozellik skor grafigi
- `figures/roc_curves_all_models_flow_*.png` — tum modeller ROC karsilastirma
- `figures/roc_ova_*_flow_*.png` — model bazli OVA ROC egrileri
- `figures/metric_comparison_flow_*.png` — metrik karsilastirma grafigi

---

## Rapor Icin Oneriler

- `metrics_summary_flow_B.csv` tablosunu rapora ekleyin (her modelin metrikleri).
- ROC karsilastirma grafigini ve en iyi modelin OVA ROC egrisini ekleyin.
- En basarili model + ozellik kombinasyonunu `best_params_*` dosyalarindan yorumlayin.
