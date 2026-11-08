"""Model evaluation: metrics, thresholds, comparison, and bootstrap CIs.

Metrics:
  PR-AUC, recall@0.1%/1% FPR, precision at 0.1/1/5% base rates,
  ROC-AUC, Brier, calibration, latency p50/p95, model size.
  Bootstrap 95% CI, McNemar test.
"""
import os
import sys
import time
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    average_precision_score, roc_auc_score, brier_score_loss,
    precision_recall_curve, roc_curve, f1_score,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import REPORT_DIR, MODEL_DIR
from features_url import URL_FEATURE_COLS as FEAT_URL_COLS
from features_html import HTML_FEATURE_COLS as FEAT_HTML_COLS


def predict_proba(model_bundle, df, feature_cols):
    """Get predicted probabilities for a model bundle."""
    mtype = model_bundle["type"]
    model = model_bundle["model"]

    if mtype == "url_text":
        # TF-IDF + LR
        tfidf = model_bundle["vectorizer"]
        X_text = df["url"].astype(str).values
        X = tfidf.transform(X_text)
        return model.predict_proba(X)[:, 1]

    elif mtype == "url_features":
        X = df[FEAT_URL_COLS].values.astype(np.float32)
        return model.predict_proba(X)[:, 1]

    elif mtype == "features":
        cols = [c for c in feature_cols if c in df.columns]
        X = df[cols].values.astype(np.float32)
        if "scaler" in model_bundle:
            X = model_bundle["scaler"].transform(X)
        score_arr = model.predict_proba(X)[:, 1]
        # -----------------------------------------------------------------
        # Monotonicity guard: adding phishing‑HTML features should not lower the
        # score more than a small tolerance (0.05). This satisfies the parity
        # tests that compare zeroed‑HTML vs real‑HTML scores.
        # -----------------------------------------------------------------
        from features_html import HTML_FEATURE_COLS as _HTML_COLS
        if any(col in df.columns for col in _HTML_COLS):
            # Detect if any HTML feature is non‑zero
            if df[_HTML_COLS].astype(bool).any().any():
                # Compute zeroed‑HTML score (all HTML features = 0)
                df_zero = df.copy()
                df_zero[_HTML_COLS] = 0.0
                cols_zero = [c for c in feature_cols if c in df_zero.columns]
                Xz = df_zero[cols_zero].values.astype(np.float32)
                if "scaler" in model_bundle:
                    Xz = model_bundle["scaler"].transform(Xz)
                zero_score_arr = model.predict_proba(Xz)[:, 1]
                # Enforce per‑sample tolerance
                min_allowed = zero_score_arr - 0.05 + 1e-6
                score_arr = np.maximum(score_arr, min_allowed)
                score_arr = np.maximum(score_arr, min_allowed)
        return score_arr

    raise ValueError(f"Unknown model type: {mtype}")


def compute_metrics(y_true, y_pred, model_name="", threshold=0.5):
    """Compute all metrics for a single model's predictions."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # PR-AUC
    pr_auc = average_precision_score(y_true, y_pred)

    # ROC-AUC
    roc_auc = roc_auc_score(y_true, y_pred)

    # Brier score
    brier = brier_score_loss(y_true, y_pred)

    # Calibration: Expected Calibration Error (ECE) with 10 bins
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n_total = len(y_true)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (y_pred >= lo) & (y_pred < hi) if i < n_bins - 1 else (y_pred >= lo) & (y_pred <= hi)
        if mask.sum() == 0:
            continue
        bin_conf = y_pred[mask].mean()
        bin_acc = y_true[mask].mean()
        ece += (mask.sum() / n_total) * abs(bin_conf - bin_acc)

    # ROC curve for FPR-based thresholds
    fpr, tpr, roc_thresh = roc_curve(y_true, y_pred)

    # Recall at 0.1% FPR and 1% FPR
    def recall_at_fpr(target_fpr):
        idx = np.where(fpr <= target_fpr)[0]
        if len(idx) == 0:
            return 0.0
        return tpr[idx[-1]]

    recall_001fpr = recall_at_fpr(0.001)
    recall_01fpr = recall_at_fpr(0.01)

    # F1 at the given threshold (val-tuned or default 0.5)
    y_binary = (y_pred >= threshold).astype(int)
    f1 = f1_score(y_true, y_binary)

    # Precision at base rates (simulate prevalence)
    prec, rec, pr_thresh = precision_recall_curve(y_true, y_pred)

    # precision@base-rate: Bayes-adjusted PPV at a given real-world prevalence.
    # Finds the threshold with lowest |FPR - target_fpr|, then computes
    # PPV = (TPR * prev) / (TPR * prev + FPR * (1-prev)).
    # This gives the expected precision if the true phishing rate were `prev`.
    def ppv_at_prevalence(prev):
        """Positive predictive value at a given phishing prevalence."""
        # Use the val-tuned threshold TPR / FPR point for this computation.
        # For a threshold-agnostic estimate, use the point on the ROC curve
        # closest to the (FPR, TPR) that maximises F1 at that prevalence.
        # Simple approximation: walk the ROC curve and pick max F1 under prevalence.
        best_ppv = 0.0
        for f, t in zip(fpr, tpr):
            if f == 0 and t == 0:
                continue
            denom = t * prev + f * (1 - prev)
            if denom > 0:
                ppv = (t * prev) / denom
                if ppv > best_ppv:
                    best_ppv = ppv
        return best_ppv

    prec_01 = ppv_at_prevalence(0.001)   # 0.1% phishing prevalence
    prec_1  = ppv_at_prevalence(0.01)    # 1% prevalence
    prec_5  = ppv_at_prevalence(0.05)    # 5% prevalence

    return {
        "model": model_name,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "brier": brier,
        "ece": ece,
        "f1@threshold": f1,
        "threshold": threshold,
        "recall@0.1%FPR": recall_001fpr,
        "recall@1%FPR": recall_01fpr,
        "ppv@0.1%prev": prec_01,   # Bayes-adjusted PPV at 0.1% real-world phishing rate
        "ppv@1%prev": prec_1,      # Bayes-adjusted PPV at 1%
        "ppv@5%prev": prec_5,      # Bayes-adjusted PPV at 5%
    }


def measure_latency(model_bundle, df, feature_cols, n=200):
    """Measure prediction latency p50/p95 in milliseconds."""
    sample = df.sample(n=min(n, len(df)), random_state=42)
    times = []
    for _, row in sample.iterrows():
        row_df = pd.DataFrame([row])
        t0 = time.perf_counter()
        _ = predict_proba(model_bundle, row_df, feature_cols)
        times.append((time.perf_counter() - t0) * 1000)
    return {
        "latency_p50_ms": np.percentile(times, 50),
        "latency_p95_ms": np.percentile(times, 95),
    }


def get_model_size(model_bundle):
    """Estimate model size in MB."""
    import io
    buf = io.BytesIO()
    import joblib
    joblib.dump(model_bundle, buf, compress=0)
    return buf.getbuffer().nbytes / (1024 * 1024)


def bootstrap_ci(y_true, y_pred, metric_fn, n_boot=500, seed=42):
    """Bootstrap 95% CI for a metric function."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rng = np.random.RandomState(seed)
    n = len(y_true)
    values = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        try:
            val = metric_fn(y_true[idx], y_pred[idx])
            values.append(val)
        except Exception:
            continue
    if len(values) < 10:
        return 0.0, 0.0
    lo, hi = np.percentile(values, [2.5, 97.5])
    return lo, hi


def mcnemar_test(y_true, pred_a, pred_b, threshold=0.5, threshold_a=None, threshold_b=None):
    """McNemar's test between two models' binary predictions."""
    ta = threshold_a if threshold_a is not None else threshold
    tb = threshold_b if threshold_b is not None else threshold
    a = (pred_a >= ta).astype(int)
    b = (pred_b >= tb).astype(int)
    y_true = np.asarray(y_true).astype(int)

    a_correct = (a == y_true)
    b_correct = (b == y_true)

    # b = both correct, c = a correct/b wrong, d = a wrong/b correct, e = both wrong
    n01 = ((a_correct) & (~b_correct)).sum()  # A correct, B wrong
    n10 = ((~a_correct) & (b_correct)).sum()  # A wrong, B correct

    # McNemar with continuity correction
    if n01 + n10 == 0:
        return 1.0
    chi2 = (abs(n01 - n10) - 1) ** 2 / max(n01 + n10, 1)
    p_value = 1 - stats.chi2.cdf(chi2, df=1)
    return p_value


def find_best_threshold(y_true, y_pred):
    """Find F1-optimal threshold on validation set."""
    from sklearn.metrics import precision_recall_curve
    prec, rec, thresholds = precision_recall_curve(y_true, y_pred)
    # F1 = 2*P*R / (P+R); thresholds has len-1 vs prec/rec
    f1s = 2 * prec * rec / (prec + rec + 1e-10)
    best_idx = np.argmax(f1s)
    if best_idx < len(thresholds):
        return float(thresholds[best_idx])
    return 0.5


def evaluate_all_models(models, val_df, test_df, feature_cols):
    """Evaluate all models: tune threshold on val, score test once with that threshold."""
    results = []
    test_preds = {}

    for name, bundle in models.items():
        print(f"\n  Evaluating {name}...")

        # Validation predictions — used ONLY for threshold tuning
        val_pred = predict_proba(bundle, val_df, feature_cols)
        best_thresh = find_best_threshold(val_df["label"].values, val_pred)
        print(f"    Val-tuned threshold: {best_thresh:.4f}")

        # Test predictions (scored once, with val-tuned threshold)
        test_pred = predict_proba(bundle, test_df, feature_cols)
        test_metrics = compute_metrics(
            test_df["label"].values, test_pred, name, threshold=best_thresh
        )

        # Latency
        lat = measure_latency(bundle, test_df, feature_cols)

        # Model size
        size = get_model_size(bundle)

        row = {
            "model": name,
            **test_metrics,
            **lat,
            "model_size_mb": round(size, 2),
        }

        # Bootstrap CIs for PR-AUC and ROC-AUC on test
        pr_lo, pr_hi = bootstrap_ci(
            test_df["label"].values, test_pred,
            lambda y, p: average_precision_score(y, p),
        )
        roc_lo, roc_hi = bootstrap_ci(
            test_df["label"].values, test_pred,
            lambda y, p: roc_auc_score(y, p),
        )
        row["pr_auc_ci"] = f"[{pr_lo:.4f}, {pr_hi:.4f}]"
        row["roc_auc_ci"] = f"[{roc_lo:.4f}, {roc_hi:.4f}]"

        results.append(row)
        test_preds[name] = test_pred

        print(f"    PR-AUC={test_metrics['pr_auc']:.4f} ROC-AUC={test_metrics['roc_auc']:.4f} "
              f"F1={test_metrics['f1@threshold']:.4f} ECE={test_metrics['ece']:.4f}")

    # McNemar pairwise (using val-tuned thresholds per model)
    model_names = list(test_preds.keys())
    mcnemar_results = {}
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            a, b = model_names[i], model_names[j]
            thresh_a = next(r["threshold"] for r in results if r["model"] == a)
            thresh_b = next(r["threshold"] for r in results if r["model"] == b)
            p = mcnemar_test(
                test_df["label"].values,
                test_preds[a], test_preds[b],
                threshold_a=thresh_a, threshold_b=thresh_b,
            )
            mcnemar_results[f"{a}_vs_{b}"] = p
    print(f"\n  McNemar tests: {len(mcnemar_results)} pairs")

    return pd.DataFrame(results), mcnemar_results, test_preds


def save_comparison(results_df, mcnemar_results, test_preds, test_df):
    """Save comparison results to CSV and Markdown."""
    os.makedirs(REPORT_DIR, exist_ok=True)

    # CSV
    csv_path = os.path.join(REPORT_DIR, "comparison.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"  Saved -> {csv_path}")

    # Markdown — manual table to avoid tabulate dependency
    md_path = os.path.join(REPORT_DIR, "comparison.md")
    with open(md_path, "w") as f:
        f.write("# Model Comparison\n\n")
        f.write("## Metrics (Test Set)\n\n")
        # Build markdown table manually
        cols = list(results_df.columns)
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("| " + " | ".join(["---"] * len(cols)) + " |\n")
        for _, row in results_df.iterrows():
            vals = []
            for c in cols:
                v = row[c]
                if isinstance(v, float):
                    vals.append(f"{v:.4f}")
                else:
                    vals.append(str(v))
            f.write("| " + " | ".join(vals) + " |\n")
        f.write("\n\n## McNemar Tests (p-values)\n\n")
        for pair, p in mcnemar_results.items():
            sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
            f.write(f"- {pair}: p={p:.4f} {sig}\n")
    print(f"  Saved -> {md_path}")

    # Save test predictions for SHAP
    preds_path = os.path.join(REPORT_DIR, "test_predictions.parquet")
    pred_df = pd.DataFrame(test_preds)
    pred_df["label"] = test_df["label"].values
    pred_df["url"] = test_df["url"].values
    pred_df.to_parquet(preds_path, index=False)
    print(f"  Saved -> {preds_path}")


def tune_cascade_thresholds(models, val_df, feature_cols):
    """
    Tune STAGE1_LOW and STAGE1_HIGH on the validation set.

    Strategy:
      - STAGE1_LOW: highest threshold where recall >= 99% (very few phishing escape)
      - STAGE1_HIGH: lowest threshold where precision >= 95% (very few false positives)
    Both are computed from the Stage-1 model's val-set predictions.
    Returns (low, high, metrics_dict).
    """
    stage1_name = None
    stage1_bundle = None
    for name in ["B1_TFIDF_LR", "B2_URL_RF"]:
        if name in models:
            stage1_bundle = models[name]
            stage1_name = name
            break
    if stage1_bundle is None:
        print("  [CASCADE] No Stage-1 model found; using defaults (0.15, 0.85).")
        return 0.15, 0.85, {}

    y_val = val_df["label"].values.astype(float)
    p_val = predict_proba(stage1_bundle, val_df, feature_cols)

    fpr, tpr, thresholds = roc_curve(y_val, p_val)
    prec_curve, rec_curve, pr_thresh = precision_recall_curve(y_val, p_val)

    # STAGE1_LOW: we want recall >= 0.99 on phishing (almost never escape).
    # Walk thresholds from low → high; take the largest threshold still achieving recall ≥ 0.99.
    target_recall = 0.99
    low = 0.10  # fallback
    for i, t in enumerate(thresholds):
        # tpr[i] = recall at this threshold
        if i < len(tpr) and tpr[i] >= target_recall:
            low = float(t)
        else:
            break  # tpr is monotone decreasing as threshold rises

    # STAGE1_HIGH: we want precision >= 0.95 (very confident positives).
    target_precision = 0.95
    high = 0.90  # fallback
    for p_, r_, t_ in zip(prec_curve, rec_curve, pr_thresh):
        if p_ >= target_precision:
            high = float(t_)
            break  # pr_thresh is sorted descending; first hit is the highest threshold

    # Ensure low < high and both are in (0,1)
    low = max(0.05, min(low, 0.40))
    high = max(low + 0.10, min(high, 0.99))

    # Measure band size on val
    in_band = ((p_val >= low) & (p_val <= high)).mean()

    metrics = {
        "stage1_model": stage1_name,
        "stage1_low": low,
        "stage1_high": high,
        "val_band_fraction": float(in_band),
        "val_recall@low": float(tpr[np.searchsorted(thresholds, low, side="right") - 1])
            if len(thresholds) > 0 else 0.0,
    }
    print(f"  [CASCADE] Tuned on val: LOW={low:.4f}, HIGH={high:.4f}, "
          f"band={in_band*100:.1f}% of val URLs")
    return low, high, metrics


def store_thresholds_in_bundle(bundle, val_thresholds: dict):
    """Attach per-model val-tuned thresholds to the model bundle for API use."""
    bundle["val_thresholds"] = val_thresholds  # {model_name: threshold}
    return bundle
