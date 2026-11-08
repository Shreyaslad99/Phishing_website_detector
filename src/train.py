"""Model training: baselines + main models on URL and URL+HTML features.

Baselines:
  B1: char 3-5gram TF-IDF + LogisticRegression (URL text only)
  B2: URL features + RandomForest

Main (URL+HTML features):
  LR, RF, XGBoost(hist), MLPClassifier(2 hidden layers)
All with class weights.
"""
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import MODEL_DIR, SEED
from features_url import URL_FEATURE_COLS as FEAT_URL_COLS
from features_html import HTML_FEATURE_COLS as FEAT_HTML_COLS

warnings.filterwarnings("ignore", category=FutureWarning)

# Feature column names (imported from feature modules)
URL_COLS = FEAT_URL_COLS  # 30 numeric URL features
HTML_COLS = FEAT_HTML_COLS  # 8 numeric HTML features


def get_feature_matrix(df: pd.DataFrame, include_html: bool = True):
    """Extract numeric feature matrix from DataFrame."""
    cols = list(URL_COLS)
    if include_html:
        cols = cols + list(HTML_COLS)
    # Only include columns that exist
    cols = [c for c in cols if c in df.columns]
    X = df[cols].values.astype(np.float32)
    y = df["label"].values.astype(np.float32)
    return X, y, cols


def train_b1_tfidf_lr(X_train_text, y_train):
    """B1: Character 3-5gram TF-IDF on URL text + LogisticRegression."""
    print("  [B1] Training TF-IDF + LR...")
    t0 = time.time()
    tfidf = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        max_features=50000,
        sublinear_tf=True,
    )
    X_tfidf = tfidf.fit_transform(X_train_text)
    lr = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        C=1.0,
        random_state=SEED,
        solver="liblinear",
    )
    lr.fit(X_tfidf, y_train)
    elapsed = time.time() - t0
    print(f"  [B1] Done in {elapsed:.1f}s (TF-IDF shape: {X_tfidf.shape})")
    return {"name": "B1_TFIDF_LR", "model": lr, "vectorizer": tfidf, "type": "url_text"}


def train_b2_url_rf(X_train, y_train):
    """B2: URL features + RandomForest."""
    print("  [B2] Training URL features + RF...")
    t0 = time.time()
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    elapsed = time.time() - t0
    print(f"  [B2] Done in {elapsed:.1f}s")
    return {"name": "B2_URL_RF", "model": rf, "type": "url_features"}


def train_lr(X_train, y_train):
    """LR on URL+HTML features."""
    print("  [LR] Training LogisticRegression...")
    t0 = time.time()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    lr = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        C=1.0,
        random_state=SEED,
        solver="liblinear",
    )
    lr.fit(X_scaled, y_train)
    elapsed = time.time() - t0
    print(f"  [LR] Done in {elapsed:.1f}s")
    return {"name": "LR", "model": lr, "scaler": scaler, "type": "features"}


def train_rf(X_train, y_train):
    """RF on URL+HTML features."""
    print("  [RF] Training RandomForest...")
    t0 = time.time()
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=25,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    elapsed = time.time() - t0
    print(f"  [RF] Done in {elapsed:.1f}s")
    return {"name": "RF", "model": rf, "type": "features"}


def train_xgb(X_train, y_train):
    """XGBoost (hist) on URL+HTML features."""
    print("  [XGB] Training XGBoost...")
    t0 = time.time()
    # Compute scale_pos_weight for class imbalance
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    spw = n_neg / max(n_pos, 1)

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        tree_method="hist",
        scale_pos_weight=spw,
        random_state=SEED,
        n_jobs=-1,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train)
    elapsed = time.time() - t0
    print(f"  [XGB] Done in {elapsed:.1f}s")
    return {"name": "XGB", "model": model, "type": "features"}


def train_mlp(X_train, y_train):
    """MLPClassifier with 2 hidden layers on URL+HTML features."""
    print("  [MLP] Training MLPClassifier...")
    t0 = time.time()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=15,
        random_state=SEED,
        alpha=0.001,
    )
    mlp.fit(X_scaled, y_train)
    elapsed = time.time() - t0
    print(f"  [MLP] Done in {elapsed:.1f}s")
    return {"name": "MLP", "model": mlp, "scaler": scaler, "type": "features"}


def train_all_models(train_df, val_df, feature_cols_url, feature_cols_html):
    """Train all models and return a dict of trained model bundles."""
    models = {}

    # --- B1: TF-IDF + LR (URL text) ---
    X_train_text = train_df["url"].astype(str).values
    y_train = train_df["label"].values
    models["B1_TFIDF_LR"] = train_b1_tfidf_lr(X_train_text, y_train)

    # --- B2: URL features + RF ---
    X_train_url = train_df[feature_cols_url].values.astype(np.float32)
    y_train_url = train_df["label"].values
    models["B2_URL_RF"] = train_b2_url_rf(X_train_url, y_train_url)

    # --- Main models on URL+HTML features ---
    all_cols = feature_cols_url + feature_cols_html
    all_cols = [c for c in all_cols if c in train_df.columns]
    X_train_full = train_df[all_cols].values.astype(np.float32)
    y_train_full = train_df["label"].values

    models["LR"] = train_lr(X_train_full, y_train_full)
    models["RF"] = train_rf(X_train_full, y_train_full)
    models["XGB"] = train_xgb(X_train_full, y_train_full)
    models["MLP"] = train_mlp(X_train_full, y_train_full)

    return models, all_cols


def save_model_bundle(models, feature_cols, output_path, val_thresholds=None):
    """Save all models + metadata to a single joblib file."""
    bundle = {
        "models": {},
        "feature_cols": feature_cols,
        "val_thresholds": val_thresholds or {},
    }
    for name, m in models.items():
        entry = {"type": m["type"]}
        for k, v in m.items():
            if k != "name":
                entry[k] = v
        bundle["models"][name] = entry
    joblib.dump(bundle, output_path)
    print(f"  Saved model bundle -> {output_path} ({os.path.getsize(output_path)/1024/1024:.1f} MB)")
