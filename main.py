"""
HIGGS Machine Learning Pipeline - Ana calistirma betigi.

Asagidaki adimlari sirasiyla yurutur:
    Bolum 1: Veri on isleme (IQR aykiri deger analizi + MinMaxScaler)
    Bolum 2: Filter-based ozellik secimi (ANOVA / Mutual Information, en iyi 15)
    Bolum 3: Nested Cross-Validation (Flowchart A ve Flowchart B), 4 model
             icin metrikler ve ROC egrileri.

Kullanim:
    python main.py                 # tum pipeline (varsayilan)
    python main.py --flow A        # yalnizca Flowchart A
    python main.py --flow B        # yalnizca Flowchart B
    python main.py --sample 10000  # modelleme alt-ornek boyutu
"""

import os
import argparse
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import config
from src.data_loader import load_higgs_sample
from src import preprocessing as prep
from src import feature_selection as fs
from src.models import MODEL_NAMES
from src.nested_cv import run_nested_cv
from src import visualization as viz


def ensure_dirs():
    for d in [config.OUTPUT_DIR, config.FIGURES_DIR, config.RESULTS_DIR]:
        os.makedirs(d, exist_ok=True)


def step1_preprocessing(X, y):
    print("\n=== BOLUM 1: VERI ON ISLEME ===")
    # Aykiri deger analizi (IQR)
    report = prep.analyze_outliers_iqr(X)
    print(report[["feature", "n_outliers", "outlier_pct"]].head(10).to_string(index=False))

    # Box-plot (olceklemeden once)
    prep.plot_boxplots(X, fname="boxplots_before_scaling.png",
                       title="Ozellik Box-Plot'lari (olceklemeden once)")

    # Aykiri degerleri IQR sinirlarina kirp (winsorize)
    X_clipped = prep.clip_outliers_iqr(X)
    prep.plot_boxplots(X_clipped, fname="boxplots_after_clipping.png",
                       title="Ozellik Box-Plot'lari (IQR kirpma sonrasi)")
    print("[main] Aykiri degerler IQR sinirlarina kirpildi (winsorization).")
    return X_clipped


def step2_feature_selection(X, y):
    print("\n=== BOLUM 2: OZELLIK SECIMI ===")
    selected, ranked = fs.select_top_k(X, y)
    fs.plot_feature_scores(ranked)
    print(f"[main] Secilen en iyi {config.N_FEATURES_TO_SELECT} ozellik "
          f"({config.FEATURE_SELECTION_METHOD}):")
    for i, f in enumerate(selected, 1):
        print(f"   {i:2d}. {f}")
    return selected, ranked


def get_modeling_subset(X, y):
    """Nested CV icin dengeli (stratified) bir alt-ornek dondurur."""
    if config.MODEL_SAMPLE_SIZE is None or config.MODEL_SAMPLE_SIZE >= len(X):
        return X, y
    X_sub, _, y_sub, _ = train_test_split(
        X, y,
        train_size=config.MODEL_SAMPLE_SIZE,
        stratify=y,
        random_state=config.RANDOM_STATE,
    )
    print(f"[main] Nested CV icin {len(X_sub)} orneklik stratified alt-ornek kullaniliyor.")
    return X_sub, y_sub


def run_flowchart(X, y, flow):
    """
    Belirtilen flowchart icin tum modellerde nested CV calistirir.

    flow == "A": ic dongude ozellik secim kombinasyonlari (select__k)
    flow == "B": ic dongude hiperparametre kombinasyonlari
    """
    print(f"\n=== BOLUM 3: NESTED CV - FLOWCHART {flow} ===")
    results = []
    for name in MODEL_NAMES:
        if flow == "A":
            param_grid = {"select__k": config.FEATURE_K_GRID}
        else:
            param_grid = config.PARAM_GRIDS[name]
        print(f"\n>>> Model: {name} | Flowchart {flow} | grid={param_grid}")
        t0 = time.time()
        res = run_nested_cv(X, y, name, param_grid)
        res["elapsed_sec"] = time.time() - t0
        print(f"    Tamamlandi ({res['elapsed_sec']:.1f}s) | "
              f"Ortalama AUC={res['summary']['roc_auc_mean']:.4f}")
        results.append(res)
    return results


def save_results(results, flow):
    """Metrik tablolarini, en iyi parametreleri kaydeder ve ROC/karsilastirma cizer."""
    # Ozet metrik tablosu
    summary_df = pd.DataFrame([r["summary"] for r in results])
    summary_path = os.path.join(config.RESULTS_DIR, f"metrics_summary_flow_{flow}.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\n[main] Ozet metrikler kaydedildi: {summary_path}")
    print(summary_df.round(4).to_string(index=False))

    # Fold bazli detaylar + en iyi parametreler
    for r in results:
        fm = r["fold_metrics"].copy()
        fm.insert(0, "model", r["model"])
        fm_path = os.path.join(config.RESULTS_DIR,
                               f"fold_metrics_{r['model']}_flow_{flow}.csv")
        fm.to_csv(fm_path, index=False)

        bp = pd.DataFrame(r["best_params"])
        bp.insert(0, "fold", range(1, len(bp) + 1))
        bp_path = os.path.join(config.RESULTS_DIR,
                               f"best_params_{r['model']}_flow_{flow}.csv")
        bp.to_csv(bp_path, index=False)

    # ROC egrileri
    viz.plot_roc_all_models(results, fname=f"roc_curves_all_models_flow_{flow}.png")
    for r in results:
        viz.plot_roc_ova(r["oof_y_true"], r["oof_y_proba"], r["model"],
                         fname=f"roc_ova_{r['model']}_flow_{flow}.png")

    # Metrik karsilastirma grafigi
    viz.plot_metric_comparison(summary_df, fname=f"metric_comparison_flow_{flow}.png")

    # En iyi model
    best_row = summary_df.loc[summary_df["roc_auc_mean"].idxmax()]
    print(f"\n[main] *** Flowchart {flow} EN IYI MODEL: {best_row['model']} "
          f"(ROC-AUC = {best_row['roc_auc_mean']:.4f}) ***")
    return summary_df


def main():
    parser = argparse.ArgumentParser(description="HIGGS ML Pipeline")
    parser.add_argument("--flow", choices=["A", "B", "both"], default="both",
                        help="Calistirilacak flowchart (A, B veya both)")
    parser.add_argument("--sample", type=int, default=None,
                        help="Modelleme alt-ornek boyutu (config'i gecersiz kilar)")
    parser.add_argument("--method", choices=["anova", "mutual_info"], default=None,
                        help="Ozellik secim yontemi")
    args = parser.parse_args()

    if args.sample is not None:
        config.MODEL_SAMPLE_SIZE = args.sample
    if args.method is not None:
        config.FEATURE_SELECTION_METHOD = args.method

    ensure_dirs()
    print("=" * 70)
    print(" HIGGS MACHINE LEARNING PIPELINE")
    print(" Feature Selection & Hyperparameter Optimization")
    print("=" * 70)

    # Veri yukleme (rastgele 100.000 ornek)
    X, y = load_higgs_sample()

    # Bolum 1
    X_clipped = step1_preprocessing(X, y)

    # Bolum 2 (tam 100k uzerinde analiz/gorsel)
    step2_feature_selection(X_clipped, y)

    # Bolum 3 icin dengeli alt-ornek
    X_model, y_model = get_modeling_subset(X_clipped, y)

    flows = ["A", "B"] if args.flow == "both" else [args.flow]
    all_summaries = {}
    for flow in flows:
        results = run_flowchart(X_model, y_model, flow)
        all_summaries[flow] = save_results(results, flow)

    print("\n" + "=" * 70)
    print(" PIPELINE TAMAMLANDI")
    print(f" Grafikler: {config.FIGURES_DIR}")
    print(f" Sonuc tablolari: {config.RESULTS_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
