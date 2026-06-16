"""
Gorsellestirme: ROC egrileri ve metrik karsilastirma grafikleri.

ROC egrileri OVA (One-vs-All) mantigiyla cizilir. HIGGS ikili (binary) bir
problem oldugundan pozitif sinif (1 = sinyal) icin ROC egrisi, negatif sinifin
(0 = arka plan) tamamlayicisidir; bu nedenle her iki sinif da gosterilir.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

import config


def plot_roc_all_models(results, fname="roc_curves_all_models.png"):
    """
    Tum modellerin OOF ROC egrilerini tek grafikte karsilastirir.

    results : run_nested_cv ciktilarinin listesi.
    """
    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    for res in results:
        fpr, tpr, _ = roc_curve(res["oof_y_true"], res["oof_y_proba"])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2,
                label=f"{res['model']} (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Rastgele (AUC = 0.500)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Egrileri - Tum Modeller (Out-of-Fold)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(config.FIGURES_DIR, fname)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[visualization] ROC karsilastirma grafigi kaydedildi: {path}")
    return path


def plot_roc_ova(y_true, y_proba, model_name, fname=None):
    """
    Tek bir model icin OVA (One-vs-All) ROC egrileri.

    Ikili problemde her iki sinif (0 ve 1) icin ayri ROC egrisi cizilir.
    """
    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    fname = fname or f"roc_ova_{model_name}.png"

    classes = [0, 1]
    # Her sinif icin pozitif olasilik: sinif 1 -> y_proba, sinif 0 -> 1 - y_proba
    proba_matrix = np.column_stack([1 - y_proba, y_proba])
    y_bin = label_binarize(y_true, classes=classes)
    # ikili durumda label_binarize tek sutun dondurur; iki sutuna genislet.
    if y_bin.shape[1] == 1:
        y_bin = np.column_stack([1 - y_bin[:, 0], y_bin[:, 0]])

    fig, ax = plt.subplots(figsize=(7, 6))
    class_labels = {0: "Arka plan (0)", 1: "Sinyal (1)"}
    for i, cls in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], proba_matrix[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2,
                label=f"{class_labels[cls]} (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"OVA ROC Egrileri - {model_name}")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(config.FIGURES_DIR, fname)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[visualization] OVA ROC grafigi kaydedildi: {path}")
    return path


def plot_metric_comparison(summary_df, fname="metric_comparison.png"):
    """Modellerin ortalama metriklerini gruplu cubuk grafikte karsilastirir."""
    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    metrics = ["accuracy_mean", "precision_mean", "recall_mean", "f1_mean", "roc_auc_mean"]
    labels = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]

    models = summary_df["model"].tolist()
    x = np.arange(len(metrics))
    width = 0.8 / len(models)

    fig, ax = plt.subplots(figsize=(11, 6))
    for i, model in enumerate(models):
        row = summary_df[summary_df["model"] == model].iloc[0]
        vals = [row[m] for m in metrics]
        ax.bar(x + i * width, vals, width, label=model)
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Skor")
    ax.set_title("Model Performans Karsilastirmasi (Nested CV Ortalamalari)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(config.FIGURES_DIR, fname)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[visualization] Metrik karsilastirma grafigi kaydedildi: {path}")
    return path
