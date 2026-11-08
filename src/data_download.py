"""Data download and feature extraction pipeline.

Downloads PhreshPhish parquet files directly from HuggingFace, processes
them locally for robustness. Also fetches PhiUSIIL (UCI id=967) for
cross-testing.
"""
import os
import sys
import time
import shutil

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    DATA_DIR, MAX_URL_ROWS, MAX_HTML_ROWS,
    HF_DATASET_ID, UCI_REPO_ID, SEED,
)
from features_url import extract_url_features, URL_FEATURE_COLS
from features_html import extract_html_features, HTML_FEATURE_COLS


def is_error_page(html: str) -> bool:
    """Heuristic: detect empty or HTTP-error placeholder pages."""
    if not html or len(html.strip()) < 100:
        return True
    lower = html.lower()
    error_markers = [
        "<title>404", "<title>403", "<title>500", "<title>error",
        "404 not found", "page not found", "access denied",
        "this site can", "err_connection", "site can't be reached",
    ]
    return any(m in lower for m in error_markers)


def download_parquet_files(n_files=7):
    """Download first N train parquet files from HuggingFace."""
    from huggingface_hub import hf_hub_download

    os.makedirs(DATA_DIR, exist_ok=True)
    raw_dir = os.path.join(DATA_DIR, "raw_parquet")
    os.makedirs(raw_dir, exist_ok=True)

    downloaded = []
    for i in range(n_files):
        fname = f"data/train-{i:03d}.parquet"
        local_path = os.path.join(raw_dir, f"train-{i:03d}.parquet")
        if os.path.exists(local_path):
            print(f"  [DL] Already cached: {fname}")
            downloaded.append(local_path)
            continue
        print(f"  [DL] Downloading {fname}...")
        try:
            path = hf_hub_download(
                repo_id=HF_DATASET_ID,
                filename=fname,
                repo_type="dataset",
                local_dir=raw_dir,
            )
            # hf_hub_download may save to a different path
            actual = os.path.join(raw_dir, fname.replace("data/", ""))
            if os.path.exists(actual):
                downloaded.append(actual)
            elif os.path.exists(path):
                downloaded.append(path)
            else:
                downloaded.append(path)
            print(f"  [DL] Done: {os.path.getsize(path)/1024/1024:.1f} MB")
        except Exception as e:
            print(f"  [DL] ERROR downloading {fname}: {e}")
            break

    return downloaded


def process_parquet_files(file_paths):
    """Process downloaded parquet files: extract URL + HTML features."""
    url_records = []
    url_feat_records = []
    html_feat_records = []

    html_phish_count = 0
    html_benign_count = 0
    html_target_per_class = MAX_HTML_ROWS // 2

    total = 0
    skipped_error_phish = 0
    t0 = time.time()

    for fpath in file_paths:
        if total >= MAX_URL_ROWS:
            break
        print(f"  [PROC] Reading {os.path.basename(fpath)}...")
        try:
            df = pd.read_parquet(fpath)
        except Exception as e:
            print(f"  [PROC] Error reading {fpath}: {e}")
            continue

        print(f"  [PROC]   {len(df)} rows, columns: {list(df.columns)}")

        for _, row in df.iterrows():
            if total >= MAX_URL_ROWS:
                break

            url = str(row.get("url", ""))
            label_str = str(row.get("label", "")).strip().lower()
            date_val = row.get("date", None)
            target = row.get("target", None)
            html = str(row.get("html", ""))

            if not url or not label_str or label_str not in ("benign", "phish"):
                continue

            label = 1 if label_str == "phish" else 0

            # Drop empty/error pages labelled phishing
            if label == 1 and is_error_page(html):
                skipped_error_phish += 1
                continue

            # URL record
            url_records.append({
                "url": url,
                "label": label,
                "date": date_val,
                "target": target,
            })

            # URL features
            ufeats = extract_url_features(url)
            ufeats["url"] = url
            ufeats["label"] = label
            ufeats["date"] = date_val
            url_feat_records.append(ufeats)

            # HTML features (balanced subset)
            html_cap_reached = (html_phish_count >= html_target_per_class and
                                html_benign_count >= html_target_per_class)
            if not html_cap_reached and html and len(html.strip()) > 100:
                if label == 1 and html_phish_count < html_target_per_class:
                    hfeats = extract_html_features(html, url)
                    hfeats["url"] = url
                    hfeats["label"] = label
                    hfeats["date"] = date_val
                    for col in URL_FEATURE_COLS:
                        hfeats[col] = ufeats.get(col, 0.0)
                    html_feat_records.append(hfeats)
                    html_phish_count += 1
                elif label == 0 and html_benign_count < html_target_per_class:
                    hfeats = extract_html_features(html, url)
                    hfeats["url"] = url
                    hfeats["label"] = label
                    hfeats["date"] = date_val
                    for col in URL_FEATURE_COLS:
                        hfeats[col] = ufeats.get(col, 0.0)
                    html_feat_records.append(hfeats)
                    html_benign_count += 1

            total += 1
            if total % 5000 == 0:
                elapsed = time.time() - t0
                print(f"  [PROC] {total:,} rows | {elapsed:.0f}s | "
                      f"HTML: phish={html_phish_count:,} benign={html_benign_count:,}")

        # Free the parquet df
        del df

    elapsed = time.time() - t0
    print(f"\n[DATA] Processed {total:,} rows in {elapsed:.0f}s "
          f"(skipped {skipped_error_phish} error-page phish)")

    # Save URL metadata
    url_path = os.path.join(DATA_DIR, "phresh_urls.parquet")
    url_df = pd.DataFrame(url_records)
    url_df.to_parquet(url_path, index=False)
    print(f"[DATA] Saved {len(url_df):,} URL records -> {url_path}")
    print(f"  Phishing share: {url_df['label'].mean():.4f}")
    if url_df["date"].notna().any():
        print(f"  Date range: {url_df['date'].min()} to {url_df['date'].max()}")
    print(f"  Columns: {list(url_df.columns)}")

    # Save URL features
    url_feat_path = os.path.join(DATA_DIR, "url_features.parquet")
    url_feat_df = pd.DataFrame(url_feat_records)
    url_feat_df.to_parquet(url_feat_path, index=False)
    print(f"[DATA] Saved {len(url_feat_df):,} URL feature rows -> {url_feat_path}")

    # Save HTML features
    html_feat_path = os.path.join(DATA_DIR, "html_features.parquet")
    html_feat_df = pd.DataFrame(html_feat_records)
    html_feat_df.to_parquet(html_feat_path, index=False)
    print(f"[DATA] Saved {len(html_feat_df):,} HTML feature rows -> {html_feat_path}")
    print(f"  Phishing share (HTML subset): {html_feat_df['label'].mean():.4f}")
    if html_feat_df["date"].notna().any():
        print(f"  Date range (HTML subset): {html_feat_df['date'].min()} to {html_feat_df['date'].max()}")

    # Summary
    print(f"\n[DATA] === PhreshPhish Summary ===")
    print(f"  Total URL rows: {len(url_df):,}")
    print(f"  Label distribution: {url_df['label'].value_counts().to_dict()}")
    if url_df["date"].notna().any():
        print(f"  Date range: {url_df['date'].min()} -> {url_df['date'].max()}")

    return url_feat_df, html_feat_df


def download_phiusiil():
    """Fetch PhiUSIIL from UCI (id=967) for cross-testing. INVERT label (1=legit)."""
    from ucimlrepo import fetch_ucirepo

    print(f"\n[DATA] Fetching PhiUSIIL from UCI (id={UCI_REPO_ID})...")
    try:
        dataset = fetch_ucirepo(id=UCI_REPO_ID)
        df = dataset.data.original
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")

        # Find URL and label columns
        url_col = None
        label_col = None
        for c in df.columns:
            cl = c.lower().strip()
            if cl in ("url", "phisiusiil_url", "website_url"):
                url_col = c
            if cl in ("label", "class", "status", "phishing"):
                label_col = c

        if url_col is None:
            for c in df.columns:
                sample = str(df[c].iloc[0])
                if sample.startswith("http"):
                    url_col = c
                    break

        if label_col is None:
            for c in df.columns:
                if df[c].dtype in (int, float) and df[c].nunique() == 2:
                    label_col = c
                    break

        if not url_col or not label_col:
            print(f"  WARNING: Could not identify URL/label columns.")
            print(f"  Available: {list(df.columns)}")
            return None

        print(f"  URL column: {url_col}, Label column: {label_col}")
        print(f"  Original label values: {df[label_col].value_counts().to_dict()}")

        # Keep URL + label only; INVERT label (1=legit -> 0, 0=phish -> 1)
        cross_df = pd.DataFrame({
            "url": df[url_col].astype(str),
            "label": (1 - df[label_col].astype(int)),  # invert
        })
        cross_df = cross_df.dropna(subset=["url"])
        cross_df = cross_df[cross_df["url"].str.len() > 0]

        cross_path = os.path.join(DATA_DIR, "phiusiil_cross.parquet")
        cross_df.to_parquet(cross_path, index=False)
        print(f"  Saved {len(cross_df):,} cross-test rows -> {cross_path}")
        print(f"  Label after inversion (1=phish): {cross_df['label'].value_counts().to_dict()}")

        return cross_df

    except Exception as e:
        print(f"  ERROR fetching PhiUSIIL: {e}")
        return None


def cleanup_raw(raw_dir):
    """Delete downloaded raw parquet files to save space."""
    if os.path.exists(raw_dir):
        shutil.rmtree(raw_dir)
        print(f"[DATA] Cleaned up raw parquet files")


if __name__ == "__main__":
    raw_dir = os.path.join(DATA_DIR, "raw_parquet")
    files = download_parquet_files(n_files=7)
    if files:
        process_parquet_files(files)
    download_phiusiil()
    cleanup_raw(raw_dir)
