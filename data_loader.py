"""
HIGGS veri setini yukleme ve rastgele ornekleme.

HIGGS veri seti (~11 milyon satir, 2.8GB gzip) basliksiz bir CSV'dir:
    - Sutun 0  : sinif etiketi (1 = sinyal, 0 = arka plan)
    - Sutun 1-28: 28 ozellik (ilk 21 dusuk-seviye kinematik, son 7 yuksek-seviye)

Tum dosyayi bellege almak yerine, dosya parca parca (chunk) okunur ve her
parcadan rastgele bir oran secilerek toplam ~SAMPLE_SIZE ornek toplanir.
Bu, bellek dostu ve gercek anlamda "rastgele" bir alt ornek saglar.
"""

import os
import numpy as np
import pandas as pd

import config


# Sinif etiketi + 28 ozellik icin anlamli isimler.
LOW_LEVEL_FEATURES = [
    "lepton_pT", "lepton_eta", "lepton_phi",
    "missing_energy_magnitude", "missing_energy_phi",
    "jet1_pt", "jet1_eta", "jet1_phi", "jet1_b_tag",
    "jet2_pt", "jet2_eta", "jet2_phi", "jet2_b_tag",
    "jet3_pt", "jet3_eta", "jet3_phi", "jet3_b_tag",
    "jet4_pt", "jet4_eta", "jet4_phi", "jet4_b_tag",
]
HIGH_LEVEL_FEATURES = [
    "m_jj", "m_jjj", "m_lv", "m_jlv", "m_bb", "m_wbb", "m_wwbb",
]
FEATURE_NAMES = LOW_LEVEL_FEATURES + HIGH_LEVEL_FEATURES
LABEL_NAME = "label"
COLUMN_NAMES = [LABEL_NAME] + FEATURE_NAMES


def _cache_path(sample_size):
    """Ornekleme sonucunu hizli tekrar yukleme icin onbellek yolu."""
    return os.path.join(config.OUTPUT_DIR, f"higgs_sample_{sample_size}.parquet")


def load_higgs_sample(sample_size=None, use_cache=True):
    """
    HIGGS veri setinden rastgele `sample_size` ornek dondurur.

    Parametreler
    ------------
    sample_size : int
        Toplanacak ornek sayisi (varsayilan config.SAMPLE_SIZE).
    use_cache : bool
        True ise daha onceden olusturulmus parquet onbellegi kullanilir.

    Donus
    -----
    X : pd.DataFrame  (sample_size, 28)
    y : pd.Series     (sample_size,)
    """
    sample_size = sample_size or config.SAMPLE_SIZE
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    cache_file = _cache_path(sample_size)
    if use_cache and os.path.exists(cache_file):
        print(f"[data_loader] Onbellekten yukleniyor: {cache_file}")
        df = pd.read_parquet(cache_file)
        X = df[FEATURE_NAMES]
        y = df[LABEL_NAME].astype(int)
        return X, y

    if not os.path.exists(config.DATA_PATH):
        raise FileNotFoundError(
            f"HIGGS veri seti bulunamadi: {config.DATA_PATH}\n"
            "Lutfen 'higgs/HIGGS.csv.gz' dosyasinin mevcut oldugundan emin olun."
        )

    print(f"[data_loader] {config.DATA_PATH} parca parca okunuyor ve "
          f"rastgele {sample_size} ornek toplaniyor...")

    # HIGGS toplam ~11.000.000 satir. Her parcadan secilecek oran:
    total_rows = 11_000_000
    frac = min(1.0, (sample_size * 1.5) / total_rows)  # biraz fazla topla
    rng = np.random.RandomState(config.RANDOM_STATE)

    collected = []
    n_collected = 0
    reader = pd.read_csv(
        config.DATA_PATH,
        header=None,
        names=COLUMN_NAMES,
        chunksize=config.CHUNK_SIZE,
        compression="gzip",
    )
    for i, chunk in enumerate(reader):
        # Her parcadan rastgele bir alt-ornek sec.
        seed = config.RANDOM_STATE + i
        part = chunk.sample(frac=frac, random_state=seed)
        collected.append(part)
        n_collected += len(part)
        print(f"  - chunk {i + 1}: toplam toplanan ornek = {n_collected}")
        if n_collected >= sample_size:
            break

    df = pd.concat(collected, ignore_index=True)
    # Tam olarak sample_size ornege indir (stratify olmadan rastgele).
    df = df.sample(n=sample_size, random_state=config.RANDOM_STATE).reset_index(drop=True)

    if use_cache:
        df.to_parquet(cache_file, index=False)
        print(f"[data_loader] Ornek onbellege kaydedildi: {cache_file}")

    X = df[FEATURE_NAMES]
    y = df[LABEL_NAME].astype(int)
    print(f"[data_loader] Yuklendi: X={X.shape}, y dagilimi={dict(y.value_counts())}")
    return X, y


if __name__ == "__main__":
    X, y = load_higgs_sample()
    print(X.head())
    print(y.head())
