"""
Merkezi yapilandirma dosyasi.

Tum proje boyunca kullanilan yollar, ornekleme boyutlari, cross-validation
ayarlari ve hiperparametre araliklari burada tanimlanir. Boylece deneyleri
tek bir yerden yonetebilirsiniz.
"""

import os

# ---------------------------------------------------------------------------
# Yollar
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# HIGGS veri seti (gzip'li CSV). Etiket ilk sutunda, ardindan 28 ozellik gelir.
DATA_PATH = os.path.join(PROJECT_ROOT, "higgs", "HIGGS.csv.gz")

# Cikti klasorleri
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")

# ---------------------------------------------------------------------------
# Ornekleme
# ---------------------------------------------------------------------------
# Odev geregi tum veri setinden RASTGELE 100.000 ornek kullanilir.
SAMPLE_SIZE = 100_000

# Cok buyuk dosyayi parca parca okurken kullanilacak chunk boyutu.
CHUNK_SIZE = 500_000

# Tekrar uretilebilirlik icin sabit tohum.
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Modelleme alt-ornegi
# ---------------------------------------------------------------------------
# SVM (rbf) ve MLP gibi modellerin nested CV'si 100.000 ornekte cok pahalidir
# (ozellikle SVM egitim suresi O(n^2)). Bu nedenle on isleme ve ozellik secimi
# tum 100.000 ornek uzerinde yapilir; nested CV modelleme adimi ise asagidaki
# boyutta dengeli (stratified) bir alt-ornek uzerinde calistirilir.
# Daha guclu bir makineniz varsa bu degeri buyutebilir veya None yapıp
# tum 100.000 ornegi kullanabilirsiniz.
MODEL_SAMPLE_SIZE = 6_000

# ---------------------------------------------------------------------------
# Ozellik secimi
# ---------------------------------------------------------------------------
# Filter-based feature selection ile secilecek ozellik sayisi.
N_FEATURES_TO_SELECT = 15

# "anova" veya "mutual_info"
FEATURE_SELECTION_METHOD = "anova"

# ---------------------------------------------------------------------------
# Nested Cross-Validation
# ---------------------------------------------------------------------------
OUTER_FOLDS = 5
INNER_FOLDS = 3

# Flowchart A icin ic dongude denenecek ozellik sayisi kombinasyonlari.
FEATURE_K_GRID = [10, 15, 20, 28]

# ---------------------------------------------------------------------------
# Hiperparametre araliklari (Flowchart B - Inner CV)
# ---------------------------------------------------------------------------
PARAM_GRIDS = {
    "KNN": {
        "model__n_neighbors": [3, 5, 7, 9, 11],
    },
    "SVM": {
        "model__C": [0.1, 1, 10],
        "model__kernel": ["linear", "rbf"],
    },
    "MLP": {
        "model__hidden_layer_sizes": [(50,), (100,)],
        "model__activation": ["relu", "tanh"],
    },
    "XGBoost": {
        "model__n_estimators": [100, 200],
        "model__max_depth": [3, 6],
        "model__learning_rate": [0.1, 0.3],
    },
}

# IQR aykiri deger katsayisi (genellikle 1.5).
IQR_MULTIPLIER = 1.5
