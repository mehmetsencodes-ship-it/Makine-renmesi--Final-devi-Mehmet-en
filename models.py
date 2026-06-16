"""
Modeller ve scikit-learn Pipeline tanimlari.

Kullanilan modeller:
    - K-Nearest Neighbors (KNN)
    - Support Vector Machine (SVM)
    - Multi-Layer Perceptron (MLP)
    - XGBoost

Her model bir Pipeline icinde tanimlanir:
    [MinMaxScaler] -> [SelectKBest] -> [model]

Olcekleme ve ozellik secimini Pipeline icine koymak, nested CV sirasinda
veri sizintisini (data leakage) onler: scaler ve secici her fold'da yalnizca
egitim verisine fit edilir.
"""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

import config
from src.feature_selection import get_score_func


MODEL_NAMES = ["KNN", "SVM", "MLP", "XGBoost"]


def get_base_estimator(name):
    """Verilen model adina gore temel (taban) tahminci dondurur."""
    if name == "KNN":
        return KNeighborsClassifier()
    if name == "SVM":
        # probability=True -> ROC-AUC ve ROC egrileri icin olasilik gerekir.
        return SVC(probability=True, random_state=config.RANDOM_STATE)
    if name == "MLP":
        return MLPClassifier(max_iter=300, random_state=config.RANDOM_STATE)
    if name == "XGBoost":
        return XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
        )
    raise ValueError(f"Bilinmeyen model: {name}")


def build_pipeline(name, k=None, method=None):
    """
    Bir model icin tam Pipeline olusturur:
        MinMaxScaler -> SelectKBest -> model

    Parametreler
    ------------
    name : str  (KNN, SVM, MLP, XGBoost)
    k    : int  SelectKBest icin ozellik sayisi (varsayilan config degeri)
    method : str  ozellik secim yontemi (anova / mutual_info)
    """
    k = k or config.N_FEATURES_TO_SELECT
    score_func = get_score_func(method)
    return Pipeline([
        ("scaler", MinMaxScaler()),
        ("select", SelectKBest(score_func=score_func, k=k)),
        ("model", get_base_estimator(name)),
    ])
