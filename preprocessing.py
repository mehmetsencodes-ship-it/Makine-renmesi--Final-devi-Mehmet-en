"""
Bolum 1: Veri On Isleme (Preprocessing).

1) IQR yontemi ile aykiri deger analizi
   - Her ozellik icin Q1, Q3, IQR hesaplanir.
   - 1.5*IQR sinirlarinin disindaki degerler raporlanir.
   - Bu projede aykiri degerler SILINMEZ; bunun yerine sinir degerlere
     "clip" (winsorize) edilir. Boylece 100.000 orneklik denge korunur ve
     asiri uc degerlerin modeli bozmasi engellenir.

2) Ozellik olcekleme
   - Tum sayisal degiskenler MinMaxScaler ile [0, 1] araligina donusturulur.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

import config


def analyze_outliers_iqr(X, multiplier=None, save_report=True):
    """
    IQR yontemi ile her ozellikteki aykiri deger sayisini hesaplar.

    Donus: aykiri deger ozetini iceren DataFrame.
    """
    multiplier = multiplier or config.IQR_MULTIPLIER
    rows = []
    for col in X.columns:
        q1 = X[col].quantile(0.25)
        q3 = X[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - multiplier * iqr
        upper = q3 + multiplier * iqr
        mask = (X[col] < lower) | (X[col] > upper)
        n_out = int(mask.sum())
        rows.append({
            "feature": col,
            "Q1": q1, "Q3": q3, "IQR": iqr,
            "lower_bound": lower, "upper_bound": upper,
            "n_outliers": n_out,
            "outlier_pct": 100.0 * n_out / len(X),
        })
    report = pd.DataFrame(rows).sort_values("n_outliers", ascending=False)

    if save_report:
        os.makedirs(config.RESULTS_DIR, exist_ok=True)
        path = os.path.join(config.RESULTS_DIR, "outlier_report_iqr.csv")
        report.to_csv(path, index=False)
        print(f"[preprocessing] Aykiri deger raporu kaydedildi: {path}")
    return report


def clip_outliers_iqr(X, multiplier=None):
    """
    Aykiri degerleri IQR sinirlarina kirpar (winsorization).

    Silme yerine kirpma tercih edilir cunku:
      - 100.000 orneklik sabit boyut korunur,
      - bilgi kaybi minimumda tutulur,
      - uc degerlerin model egitimine olan olumsuz etkisi azaltilir.
    """
    multiplier = multiplier or config.IQR_MULTIPLIER
    X_clipped = X.copy()
    for col in X.columns:
        q1 = X[col].quantile(0.25)
        q3 = X[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - multiplier * iqr
        upper = q3 + multiplier * iqr
        X_clipped[col] = X_clipped[col].clip(lower=lower, upper=upper)
    return X_clipped


def plot_boxplots(X, fname="boxplots_before_scaling.png", title="Ozellik Box-Plot'lari"):
    """Tum ozellikler icin box-plot cizer (aykiri degerleri gorsel inceleme)."""
    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    n = len(X.columns)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = axes.flatten()
    for i, col in enumerate(X.columns):
        sns.boxplot(y=X[col], ax=axes[i], color="#4C72B0")
        axes[i].set_title(col, fontsize=9)
        axes[i].set_ylabel("")
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    path = os.path.join(config.FIGURES_DIR, fname)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[preprocessing] Box-plot kaydedildi: {path}")
    return path


def scale_minmax(X_train, X_test=None):
    """
    MinMaxScaler ile [0, 1] araligina olcekler.

    Scaler yalnizca egitim verisine fit edilir (veri sizintisini onlemek icin)
    ve hem egitim hem test verisine uygulanir.
    """
    scaler = MinMaxScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns, index=X_train.index,
    )
    if X_test is not None:
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns, index=X_test.index,
        )
        return X_train_scaled, X_test_scaled, scaler
    return X_train_scaled, scaler
