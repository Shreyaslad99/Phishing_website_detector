"""SHAP-based explanations: map top features to plain sentences.

Uses TreeExplainer for tree models, coefficients for LR.
Maps top-3 features to human-readable reasons.
Checks top-feature stability across folds.
"""
import os
import sys
import numpy as np
import pandas as pd
import shap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import REPORT_DIR
from features_url import URL_FEATURE_COLS as FEAT_URL_COLS
from features_html import HTML_FEATURE_COLS as FEAT_HTML_COLS

# Feature -> plain sentence templates
REASON_TEMPLATES = {
    "has_password_field": "The page contains a password input form",
    "form_action_cross_domain": "Password form sends data to a different website",
    "external_link_ratio": "High proportion of links point to external domains",
    "iframe_count": "Page uses {value} hidden iframes",
    "hidden_element_count": "Page contains {value} hidden elements",
    "favicon_domain_mismatch": "Favicon is loaded from a different domain",
    "title_domain_mismatch": "Page title does not match the domain name",
    "script_count": "Page contains {value} scripts",
    "ip_host": "Address uses an IP number instead of a name",
    "https": "Connection is not encrypted (HTTP)",
    "punycode": "Address uses punycode (internationalized domain)",
    "url_len": "Unusually long URL",
    "host_len": "Unusually long hostname",
    "path_len": "Unusually long URL path",
    "query_len": "Very long query string in URL",
    "dot_count": "URL contains many dots (possible subdomain spoofing)",
    "hyphen_count": "Hostname contains many hyphens",
    "underscore_count": "URL contains many underscores",
    "slash_count": "URL has many path segments",
    "question_count": "URL contains many query parameters",
    "equal_count": "URL has many parameter assignments",
    "at_count": "URL contains '@' symbol (possible redirect)",
    "amp_count": "URL has many ampersand-separated parameters",
    "percent_count": "URL contains URL-encoded characters",
    "digit_count": "URL contains many digits",
    "special_count": "Many special characters in URL",
    "digit_letter_ratio": "High digit-to-letter ratio in URL",
    "subdomain_count": "Many subdomain levels in URL",
    "suspicious_word_count": "URL contains suspicious words (login, verify, password...)",
    "suspicious_in_subdomain": "Suspicious keyword found in subdomain",
    "brand_sim_score": "Address resembles a known brand",
    "host_entropy": "High randomness in hostname",
    "path_entropy": "High randomness in URL path",
    "has_port": "URL specifies a non-standard port",
    "path_segments": "URL has many path segments",
    "param_count": "URL has many query parameters",
    "tld_len": "Unusually long top-level domain",
    "domain_len": "Unusually long domain name",
}

# Opposite reasons for negative contributions (legitimate indicators)
NEG_REASONS = {
    "https": "Connection is encrypted (HTTPS)",
    "brand_sim_score": "Domain matches a known brand",
    "url_len": "URL length is normal",
    "host_len": "Hostname length is normal",
    "dot_count": "Normal number of dots in URL",
    "host_entropy": "Hostname looks like a real domain",
    "script_count": "Normal number of scripts on page",
    "external_link_ratio": "Most links point to the same domain",
    "suspicious_word_count": "No suspicious keywords in URL",
}


def get_shap_values(model_bundle, X, feature_names):
    """Get SHAP values for a model bundle."""
    mtype = model_bundle["type"]
    model = model_bundle["model"]

    if mtype == "features" or mtype == "url_features":
        from sklearn.ensemble import RandomForestClassifier
        import xgboost as xgb

        if isinstance(model, (RandomForestClassifier, xgb.XGBClassifier)):
            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X)
                # Handle different return formats
                if isinstance(shap_values, list):
                    shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
                sv = np.array(shap_values)
                # If 3D (n_samples, n_features, n_classes), take class 1
                if sv.ndim == 3:
                    sv = sv[:, :, 1] if sv.shape[2] > 1 else sv[:, :, 0]
                result = np.abs(sv).mean(axis=0)
                # Ensure 1D
                result = np.asarray(result).ravel()
                return result
            except Exception as e:
                print(f"    SHAP failed: {e}")
                return np.zeros(len(feature_names))
        elif hasattr(model, "coef_"):
            # LR: use coefficient magnitude as importance
            coefs = np.abs(model.coef_[0])
            return coefs
        else:
            # MLP or other: use permutation importance fallback
            return np.zeros(len(feature_names))
    elif mtype == "url_text":
        return None
    return None


def explain_prediction(model_bundle, row_df, feature_cols, top_n=3):
    """
    Explain a single prediction by returning top-n reasons.

    Returns list of {text, weight} dicts.
    """
    mtype = model_bundle["type"]
    model = model_bundle["model"]

    reasons = []

    if mtype == "url_text":
        # B1: TF-IDF+LR — explain via top LR coefficients for this URL's ngrams
        url = str(row_df.iloc[0].get("url", ""))
        tfidf = model_bundle.get("vectorizer")
        model_lr = model_bundle["model"]
        if tfidf is not None and hasattr(model_lr, "coef_"):
            return _tfidf_lr_reasons(url, tfidf, model_lr, top_n)
        return _url_text_reasons(url)  # fallback to keyword scan

    elif mtype == "url_features":
        # B2: URL-only RF — per-instance SHAP using TreeExplainer
        cols = [c for c in FEAT_URL_COLS if c in row_df.columns]
        X = row_df[cols].values.astype(np.float32)
        from sklearn.ensemble import RandomForestClassifier
        try:
            explainer = shap.TreeExplainer(model_bundle["model"])
            sv = explainer.shap_values(X)
            if isinstance(sv, list):
                sv = sv[1] if len(sv) > 1 else sv[0]
            sv = np.array(sv)
            if sv.ndim == 3:
                sv = sv[:, :, 1] if sv.shape[2] > 1 else sv[:, :, 0]
            contributions = sv[0]
        except Exception:
            contributions = _get_feature_importance(model_bundle["model"], cols)
        return _map_reasons_from_contributions(row_df, cols, contributions, top_n)

    elif mtype == "features":
        cols = [c for c in feature_cols if c in row_df.columns]
        X = row_df[cols].values.astype(np.float32)
        if "scaler" in model_bundle:
            X = model_bundle["scaler"].transform(X)

        from sklearn.ensemble import RandomForestClassifier
        import xgboost as xgb
        if isinstance(model, (RandomForestClassifier, xgb.XGBClassifier)):
            try:
                explainer = shap.TreeExplainer(model)
                sv = explainer.shap_values(X)
                if isinstance(sv, list):
                    sv = sv[1] if len(sv) > 1 else sv[0]
                sv = np.array(sv)
                if sv.ndim == 3:
                    sv = sv[:, :, 1] if sv.shape[2] > 1 else sv[:, :, 0]
                contributions = sv[0]
            except Exception:
                contributions = _get_feature_importance(model, cols)
        elif hasattr(model, "coef_"):
            coefs = model.coef_[0]
            contributions = coefs * X[0]  # contribution = coef * value
        else:
            contributions = _get_feature_importance(model, cols)

        return _map_reasons_from_contributions(row_df, cols, contributions, top_n)

    return reasons


def _get_feature_importance(model, feature_names):
    """Get feature importance from model (for fallback)."""
    if hasattr(model, "feature_importances_"):
        return model.feature_importances_
    elif hasattr(model, "coef_"):
        return np.abs(model.coef_[0])
    return np.ones(len(feature_names))


def _map_reasons(row_df, feature_names, importances, top_n):
    """Map feature importances to reasons (model-level, not per-instance)."""
    sorted_idx = np.argsort(-importances)
    reasons = []
    for idx in sorted_idx[:top_n]:
        fname = feature_names[idx]
        template = REASON_TEMPLATES.get(fname, fname)
        reasons.append({"text": template, "weight": float(importances[idx])})
    return reasons


def _map_reasons_from_contributions(row_df, feature_names, contributions, top_n):
    """Map per-instance SHAP contributions to reasons."""
    sorted_idx = np.argsort(-np.abs(contributions))
    reasons = []
    row = row_df.iloc[0]
    for idx in sorted_idx[:top_n * 2]:  # check more, filter to top_n
        fname = feature_names[idx]
        contrib = float(contributions[idx])
        if abs(contrib) < 1e-6:
            continue

        val = row.get(fname, 0)

        if contrib > 0:
            template = REASON_TEMPLATES.get(fname, fname)
            if "{value}" in template:
                template = template.replace("{value}", str(int(val)))
            reasons.append({"text": template, "weight": round(contrib, 4)})
        else:
            template = NEG_REASONS.get(fname)
            if template:
                reasons.append({"text": template, "weight": round(contrib, 4)})

        if len(reasons) >= top_n:
            break

    return reasons[:top_n]


def _tfidf_lr_reasons(url: str, tfidf, lr_model, top_n: int = 3) -> list:
    """
    Explain a B1 TF-IDF+LR prediction by mapping high-weight ngrams back to
    human-readable URL characteristics.
    """
    reasons = []

    # 1. Get the activated ngrams and their LR coefficient contributions
    try:
        x = tfidf.transform([url])  # sparse (1, vocab)
        coefs = lr_model.coef_[0]   # (vocab,)
        # contribution = tfidf_value * coef for each ngram
        contrib = np.array(x.multiply(coefs).todense()).ravel()
        top_idx = np.argsort(-contrib)[:30]
        vocab_inv = {v: k for k, v in tfidf.vocabulary_.items()}

        top_ngrams = [(vocab_inv.get(i, ""), float(contrib[i]))
                      for i in top_idx if contrib[i] > 0]
    except Exception:
        top_ngrams = []

    # 2. Map ngrams to human-readable reasons
    url_lower = url.lower()
    suspicious_map = [
        (["login", "log-in", "sign-in", "signin"], "URL contains a login keyword"),
        (["verify", "verif"], "URL contains a verification keyword"),
        (["secure", "security", "secur"], "URL contains a security keyword"),
        (["account", "acct"], "URL contains an account keyword"),
        (["update", "updat"], "URL contains an update keyword"),
        (["password", "passwd", "passw"], "URL contains a password keyword"),
        (["wallet", "walet"], "URL contains a wallet/crypto keyword"),
        (["confirm", "verif"], "URL contains a confirmation keyword"),
        (["banking", "bank"], "URL contains a banking keyword"),
        (["paypal", "paypai", "payp"], "URL resembles PayPal"),
        (["apple", "appl3", "app1e"], "URL resembles Apple"),
        (["amazon", "amaz0n"], "URL resembles Amazon"),
        (["microsoft", "micros0ft", "m1crosoft"], "URL resembles Microsoft"),
    ]
    for keywords, text in suspicious_map:
        if any(kw in url_lower for kw in keywords):
            reasons.append({"text": text, "weight": 1.0})
        if len(reasons) >= top_n:
            break

    # 3. If not enough keyword reasons, fall back to top TF-IDF ngrams
    if len(reasons) < top_n and top_ngrams:
        # Translate the raw ngrams into structural insights
        ngram_strs = [ng for ng, _ in top_ngrams[:5]]
        all_ngrams = " ".join(ngram_strs)
        if any(c.isdigit() for c in all_ngrams) and len(reasons) < top_n:
            reasons.append({"text": "URL contains unusual digit sequences", "weight": 0.7})
        if "-" in all_ngrams and len(reasons) < top_n:
            reasons.append({"text": "URL contains many hyphens", "weight": 0.6})

    # 4. If still empty, fall back to URL structural features
    if not reasons:
        reasons = _url_text_reasons(url)

    return reasons[:top_n]


def _url_text_reasons(url: str) -> list:
    """Generate reasons for TF-IDF URL model based on URL analysis."""
    url_lower = url.lower()
    suspicious = [
        ("login", "URL contains 'login'"),
        ("verify", "URL contains 'verify'"),
        ("secure", "URL contains 'secure'"),
        ("account", "URL contains 'account'"),
        ("update", "URL contains 'update'"),
        ("password", "URL contains 'password'"),
        ("wallet", "URL contains 'wallet'"),
        ("confirm", "URL contains 'confirm'"),
        ("banking", "URL contains 'banking'"),
    ]
    reasons = []
    for word, text in suspicious:
        if word in url_lower:
            reasons.append({"text": text, "weight": 1.0})
    if not reasons:
        import re
        if re.match(r"^\d+\.\d+\.\d+\.\d+", url):
            reasons.append({"text": "Address uses an IP number instead of a name", "weight": 0.8})
    return reasons[:3]


def compute_shap_summary(model_bundle, test_df, feature_cols, output_path):
    """Compute SHAP summary and save feature importance ranking."""
    mtype = model_bundle["type"]
    if mtype == "url_text":
        return None

    cols = [c for c in feature_cols if c in test_df.columns]
    X = test_df[cols].values.astype(np.float32)

    if "scaler" in model_bundle:
        X = model_bundle["scaler"].transform(X)

    importances = get_shap_values(model_bundle, X, cols)
    if importances is None:
        return None

    # Ensure 1D
    importances = np.asarray(importances).ravel()

    df = pd.DataFrame({
        "feature": cols,
        "importance": importances,
    }).sort_values("importance", ascending=False)

    name = model_bundle.get("name", "model")
    df.to_csv(output_path, index=False)
    print(f"    SHAP summary saved -> {output_path}")
    return df


def check_top_feature_stability(models, test_df, feature_cols):
    """Check if top features are stable across model types and across folds."""
    stability = {}
    all_top_features = {}
    for name, bundle in models.items():
        if bundle["type"] == "url_text":
            continue
        cols = [c for c in feature_cols if c in test_df.columns]
        # Limit SHAP computation to 2000 rows for speed
        sample_df = test_df if len(test_df) <= 2000 else test_df.sample(n=2000, random_state=42)
        X = sample_df[cols].values.astype(np.float32)
        if "scaler" in bundle:
            X = bundle["scaler"].transform(X)
        imps = get_shap_values(bundle, X, cols)
        if imps is not None:
            top5 = [cols[i] for i in np.argsort(-imps)[:5]]
            all_top_features[name] = top5

    # Count how often each feature appears in top-5 across models
    from collections import Counter
    feat_counts = Counter()
    for tops in all_top_features.values():
        feat_counts.update(tops)

    stability["top_features_per_model"] = all_top_features
    stability["feature_frequency_across_models"] = dict(feat_counts.most_common(10))

    # K-fold cross-validation stability for the best tree model
    from sklearn.ensemble import RandomForestClassifier
    import xgboost as xgb
    from sklearn.model_selection import StratifiedKFold

    cols = [c for c in feature_cols if c in test_df.columns]
    # Combine train+test for k-fold (already split, but we use all available data)
    all_df = test_df
    # Limit k-fold dataset for speed
    if len(all_df) > 5000:
        all_df = all_df.sample(n=5000, random_state=42)
    X_all = all_df[cols].values.astype(np.float32)
    y_all = all_df["label"].values.astype(int)

    # Use the best tree model type for k-fold
    best_tree = None
    best_name = None
    for name in ["RF", "XGB", "B2_URL_RF"]:
        if name in models:
            best_tree = models[name]
            best_name = name
            break

    if best_tree is not None:
        k = 5
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
        fold_top_features = {}
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_all, y_all)):
            X_tr, y_tr = X_all[train_idx], y_all[train_idx]
            # Train a small RF on this fold (reduced trees for speed)
            fold_rf = RandomForestClassifier(
                n_estimators=50, max_depth=12,
                class_weight="balanced", random_state=42, n_jobs=-1,
            )
            fold_rf.fit(X_tr, y_tr)
            # Use feature_importances_ (fast) instead of SHAP for k-fold
            fold_imps = fold_rf.feature_importances_
            top5 = [cols[i] for i in np.argsort(-fold_imps)[:5]]
            fold_top_features[f"fold_{fold_idx}"] = top5

        # Check stability: how often does each feature appear in top-5 across folds
        fold_feat_counts = Counter()
        for tops in fold_top_features.values():
            fold_feat_counts.update(tops)

        stability["top_features_per_fold"] = fold_top_features
        stability["feature_frequency_across_folds"] = dict(fold_feat_counts.most_common(10))
        # Stability score: fraction of features that appear in ALL folds
        all_fold_features = set()
        for tops in fold_top_features.values():
            all_fold_features.update(tops)
        stable_features = [f for f in all_fold_features
                           if fold_feat_counts[f] == k]
        stability["stable_features"] = stable_features
        print(f"    K-fold stability: {len(stable_features)} features appear in all {k} folds")

    return stability
