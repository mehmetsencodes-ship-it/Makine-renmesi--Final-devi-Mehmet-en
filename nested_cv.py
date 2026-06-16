"""
Bolum 3: Nested Cross-Validation.

Outer Loop : 5-fold (dis dongu) -> test performansi degerlendirilir.
Inner Loop : 3-fold (ic dongu)  -> GridSearchCV ile en iyi konfigurasyon secilir.

Iki senaryo desteklenir:
    Flowchart A) Ic dongude farkli OZNITELIK SECIM kombinasyonlari denenir
                 (param_grid = {"select__k": [...]}).
    Flowchart B) Ic dongude farkli HIPERPARAMETRE kombinasyonlari denenir
                 (param_grid = PARAM_GRIDS[model]).

Her iki durumda da dis dongude bagimsiz test fold'u uzerinde metrikler
hesaplanir ve out-of-fold (OOF) olasilik tahminleri ROC egrileri icin toplanir.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score,
)

import config
from src.models import build_pipeline


def _compute_metrics(y_true, y_pred, y_proba):
    """Tum metrikleri tek bir sozlukte dondurur."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }


def run_nested_cv(X, y, model_name, param_grid, scoring="roc_auc"):
    """
    Tek bir model icin nested cross-validation calistirir.

    Parametreler
    ------------
    X, y        : ozellikler ve etiket
    model_name  : "KNN" | "SVM" | "MLP" | "XGBoost"
    param_grid  : ic dongu GridSearchCV icin parametre izgarasi
    scoring     : ic dongu secim metrigi (varsayilan roc_auc)

    Donus
    -----
    dict:
        fold_metrics : her dis fold icin metrik listesi
        best_params  : her dis fold icin secilen en iyi parametreler
        oof_y_true   : OOF gercek etiketler (np.array)
        oof_y_proba  : OOF pozitif sinif olasiliklari (np.array)
        summary      : metriklerin ortalama +/- std ozeti
    """
    X = np.asarray(X)
    y = np.asarray(y)

    outer_cv = StratifiedKFold(
        n_splits=config.OUTER_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    inner_cv = StratifiedKFold(
        n_splits=config.INNER_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)

    fold_metrics = []
    best_params = []
    oof_y_true = np.zeros(len(y))
    oof_y_proba = np.zeros(len(y))

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), start=1):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        pipe = build_pipeline(model_name)
        grid = GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            scoring=scoring,
            cv=inner_cv,
            n_jobs=-1,
            refit=True,
        )
        grid.fit(X_tr, y_tr)

        best = grid.best_estimator_
        y_pred = best.predict(X_te)
        y_proba = best.predict_proba(X_te)[:, 1]

        m = _compute_metrics(y_te, y_pred, y_proba)
        m["fold"] = fold_idx
        fold_metrics.append(m)
        best_params.append(grid.best_params_)

        oof_y_true[test_idx] = y_te
        oof_y_proba[test_idx] = y_proba

        print(f"  [{model_name}] Fold {fold_idx}/{config.OUTER_FOLDS} "
              f"-> AUC={m['roc_auc']:.4f}, F1={m['f1']:.4f}, "
              f"params={grid.best_params_}")

    metrics_df = pd.DataFrame(fold_metrics)
    summary = {
        "model": model_name,
        "accuracy_mean": metrics_df["accuracy"].mean(),
        "accuracy_std": metrics_df["accuracy"].std(),
        "precision_mean": metrics_df["precision"].mean(),
        "precision_std": metrics_df["precision"].std(),
        "recall_mean": metrics_df["recall"].mean(),
        "recall_std": metrics_df["recall"].std(),
        "f1_mean": metrics_df["f1"].mean(),
        "f1_std": metrics_df["f1"].std(),
        "roc_auc_mean": metrics_df["roc_auc"].mean(),
        "roc_auc_std": metrics_df["roc_auc"].std(),
    }

    return {
        "model": model_name,
        "fold_metrics": metrics_df,
        "best_params": best_params,
        "oof_y_true": oof_y_true,
        "oof_y_proba": oof_y_proba,
        "summary": summary,
    }
