"""
Bolum 2: Ozellik Secimi (Filter-Based Feature Selection).

ANOVA F-score (f_classif) veya Mutual Information (mutual_info_classif)
yontemlerinden biri ile en iyi `k` ozellik secilir.

Bu modul hem tek seferlik analiz/gorsellestirme (en iyi 15 ozellik) hem de
nested CV icinde scikit-learn Pipeline adimi olarak kullanilir.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_selection import (
    SelectKBest, f_classif, mutual_info_classif,
)

import config


def get_score_func(method=None):
    """Yontem adina gore skor fonksiyonunu dondurur."""
    method = method or config.FEATURE_SELECTION_METHOD
    if method == "anova":
        return f_classif
    if method == "mutual_info":
        return lambda X, y: mutual_info_classif(X, y, random_state=config.RANDOM_STATE)
    raise ValueError(f"Bilinmeyen yontem: {method}")


def rank_features(X, y, method=None):
    """
    Tum ozellikleri skorlarina gore siralar.

    Donus: feature, score sutunlarini iceren DataFrame (azalan skor).
    """
    method = method or config.FEATURE_SELECTION_METHOD
    score_func = get_score_func(method)
    selector = SelectKBest(score_func=score_func, k="all")
    selector.fit(X, y)
    scores = selector.scores_
    ranked = pd.DataFrame({
        "feature": X.columns,
        "score": scores,
    }).sort_values("score", ascending=False).reset_index(drop=True)
    return ranked


def select_top_k(X, y, k=None, method=None, save=True):
    """
    En iyi `k` ozelligi secer ve secilen ozellik isimlerini dondurur.
    """
    k = k or config.N_FEATURES_TO_SELECT
    method = method or config.FEATURE_SELECTION_METHOD
    ranked = rank_features(X, y, method)
    selected = ranked.head(k)["feature"].tolist()

    if save:
        os.makedirs(config.RESULTS_DIR, exist_ok=True)
        path = os.path.join(config.RESULTS_DIR, f"feature_scores_{method}.csv")
        ranked.to_csv(path, index=False)
        print(f"[feature_selection] Ozellik skorlari kaydedildi: {path}")
    return selected, ranked


def plot_feature_scores(ranked, k=None, method=None,
                        fname="feature_importance_scores.png"):
    """Ozellik skorlarini cubuk grafikte gosterir; secilen k tanesi vurgulanir."""
    k = k or config.N_FEATURES_TO_SELECT
    method = method or config.FEATURE_SELECTION_METHOD
    os.makedirs(config.FIGURES_DIR, exist_ok=True)

    colors = ["#C44E52" if i < k else "#BBBBBB" for i in range(len(ranked))]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(ranked["feature"][::-1], ranked["score"][::-1],
            color=colors[::-1])
    method_label = "ANOVA F-score" if method == "anova" else "Mutual Information"
    ax.set_xlabel(f"{method_label} skoru")
    ax.set_title(f"Filter-Based Ozellik Secimi ({method_label}) - En iyi {k} (kirmizi)")
    fig.tight_layout()
    path = os.path.join(config.FIGURES_DIR, fname)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[feature_selection] Ozellik skor grafigi kaydedildi: {path}")
    return path
