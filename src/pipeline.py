"""Main pipeline orchestrator: data -> splits -> train -> evaluate -> explain.

Run after data_download.py completes.
"""
import os
import sys
import time
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA_DIR, MODEL_DIR, REPORT_DIR, SEED
from features_url import URL_FEATURE_COLS as FEAT_URL_COLS
from features_html import HTML_FEATURE_COLS as FEAT_HTML_COLS
from splits import prepare_split
from train import train_all_models, save_model_bundle
from evaluate import evaluate_all_models, save_comparison, tune_cascade_thresholds, store_thresholds_in_bundle
from explain import compute_shap_summary, check_top_feature_stability


def run_pipeline():
    """Run the full training and evaluation pipeline."""
    t_start = time.time()

    # --- Load data ---
    print("=" * 60)
    print("PIPELINE: Phishing URL Detector")
    print("=" * 60)

    url_feat_path = os.path.join(DATA_DIR, "url_features.parquet")
    html_feat_path = os.path.join(DATA_DIR, "html_features.parquet")

    if not os.path.exists(html_feat_path):
        print("ERROR: html_features.parquet not found. Run data_download.py first.")
        return

    # Load HTML features (contains both URL and HTML features)
    print("\n[1] Loading data...")
    df = pd.read_parquet(html_feat_path)
    print(f"  Loaded {len(df):,} rows with URL+HTML features")
    print(f"  Phishing share: {df['label'].mean():.4f}")
    if df["date"].notna().any():
        print(f"  Date range: {df['date'].min()} to {df['date'].max()}")

    # --- Split ---
    print("\n[2] Splitting data...")
    df = prepare_split(df, train_frac=0.7, val_frac=0.15)

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()

    print(f"  Train: {len(train_df):,}, Val: {len(val_df):,}, Test: {len(test_df):,}")

    # Save split assignment
    split_path = os.path.join(DATA_DIR, "split_assignment.parquet")
    df[["url", "label", "date", "split", "domain"]].to_parquet(split_path, index=False)
    print(f"  Split assignment saved -> {split_path}")

    # --- Train ---
    print("\n[3] Training models...")
    models, feature_cols = train_all_models(train_df, val_df, list(FEAT_URL_COLS), list(FEAT_HTML_COLS))

    # --- Evaluate ---
    print("\n[4] Evaluating models...")
    results_df, mcnemar_results, test_preds = evaluate_all_models(
        models, val_df, test_df, feature_cols
    )

    # --- Tune cascade thresholds on val ---
    print("\n[4b] Tuning cascade thresholds on val set...")
    from evaluate import predict_proba
    stage1_low, stage1_high, cascade_metrics = tune_cascade_thresholds(models, val_df, feature_cols)

    # Store val-tuned thresholds per model + cascade thresholds in bundle
    val_thresholds = {r["model"]: r["threshold"] for r in results_df.to_dict("records")}
    val_thresholds["_stage1_low"] = stage1_low
    val_thresholds["_stage1_high"] = stage1_high

    # --- Save comparison ---
    print("\n[5] Saving comparison...")
    save_comparison(results_df, mcnemar_results, test_preds, test_df)

    # --- SHAP / Explain ---
    print("\n[6] Computing SHAP explanations...")
    os.makedirs(REPORT_DIR, exist_ok=True)
    for name, bundle in models.items():
        shap_path = os.path.join(REPORT_DIR, f"shap_{name}.csv")
        compute_shap_summary(bundle, test_df, feature_cols, shap_path)

    # Feature stability (across models AND k-fold)
    stability = check_top_feature_stability(models, test_df, feature_cols)
    print(f"  Top feature stability (models): {stability.get('feature_frequency_across_models', {})}")

    # Save stability report
    stab_path = os.path.join(REPORT_DIR, "feature_stability.json")
    import json
    with open(stab_path, "w") as f:
        # Convert to serializable
        serializable = {}
        for k, v in stability.items():
            if isinstance(v, dict):
                serializable[k] = {str(kk): vv for kk, vv in v.items()}
            else:
                serializable[k] = str(v)
        json.dump(serializable, f, indent=2)
    print(f"  Stability report -> {stab_path}")

    # --- Save best model ---
    print("\n[7] Saving best model...")
    best_row = results_df.loc[results_df["pr_auc"].idxmax()]
    best_name = best_row["model"]
    print(f"  Best model: {best_name} (PR-AUC={best_row['pr_auc']:.4f})")

    # Save full model bundle (with val thresholds embedded)
    os.makedirs(MODEL_DIR, exist_ok=True)
    bundle_path = os.path.join(MODEL_DIR, "model_bundle.joblib")
    save_model_bundle(models, feature_cols, bundle_path, val_thresholds=val_thresholds)

    # Also save just the best model separately
    best_path = os.path.join(MODEL_DIR, "best_model.joblib")
    best_bundle = {
        "models": {best_name: models[best_name]},
        "feature_cols": feature_cols,
        "best_name": best_name,
    }
    # Include stage1 model for cascade (prefer tree models)
    for name in ["XGB", "RF", "LR", "B2_URL_RF", "B1_TFIDF_LR"]:
        if name in models and name != best_name:
            best_bundle["models"]["stage1"] = models[name]
            break
    best_bundle["val_thresholds"] = val_thresholds
    joblib.dump(best_bundle, best_path)

    # --- Cross-test on PhiUSIIL ---
    print("\n[8] Cross-testing on PhiUSIIL (URL-only features — fair for all models)...")
    print("  NOTE: HTML features zeroed. URL+HTML models (RF/XGB/LR/MLP) see a")
    print("  distribution shift; their cross-test scores are indicative only.")
    cross_path = os.path.join(DATA_DIR, "phiusiil_cross.parquet")
    if os.path.exists(cross_path):
        cross_df = pd.read_parquet(cross_path)
        print(f"  Cross-test set: {len(cross_df):,} rows (sampling 10k for speed)")
        cross_df = cross_df.sample(n=min(10000, len(cross_df)), random_state=SEED)

        from features_url import extract_url_features
        from evaluate import predict_proba as _pp, compute_metrics as _cm
        cross_records = []
        for _, row in cross_df.iterrows():
            ufeats = extract_url_features(row["url"])
            ufeats["label"] = row["label"]
            ufeats["url"] = row["url"]
            for col in FEAT_HTML_COLS:
                ufeats[col] = 0.0  # no HTML in PhiUSIIL
            cross_records.append(ufeats)
        cross_feat_df = pd.DataFrame(cross_records)

        cross_results = []
        url_only_models = {"B1_TFIDF_LR", "B2_URL_RF"}
        for name, bundle in models.items():
            pred = _pp(bundle, cross_feat_df, feature_cols)
            m = _cm(cross_feat_df["label"].values, pred, name)
            # Flag if HTML-using model
            note = "" if name in url_only_models else "(HTML zeroed — distribution shift)"
            print(f"  {name}: PR-AUC={m['pr_auc']:.4f}, ROC-AUC={m['roc_auc']:.4f} {note}")
            m["note"] = note
            cross_results.append(m)

        cross_df_results = pd.DataFrame(cross_results)
        cross_csv = os.path.join(REPORT_DIR, "cross_test_results.csv")
        cross_df_results.to_csv(cross_csv, index=False)
        print(f"  Saved -> {cross_csv}")
    else:
        print("  PhiUSIIL cross-test data not found (skipping)")

    # --- Summary ---
    elapsed = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"PIPELINE COMPLETE in {elapsed/60:.1f} min")
    print(f"{'=' * 60}")
    print(f"\nResults summary:")
    print(results_df[["model", "pr_auc", "roc_auc", "recall@1%FPR",
                       "latency_p50_ms", "model_size_mb"]].to_string(index=False))

    return results_df, models


if __name__ == "__main__":
    run_pipeline()
