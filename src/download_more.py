"""Process already-downloaded PhreshPhish parquet files and merge into existing data.

Files are large (300-500 MB each, contain raw HTML) so we stream in small
pyarrow batches (CHUNK rows) to stay within 8 GB RAM.
"""
import os, sys, time, shutil
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA_DIR, MAX_URL_ROWS, MAX_HTML_ROWS, SEED
from features_url import extract_url_features, URL_FEATURE_COLS
from features_html import extract_html_features, HTML_FEATURE_COLS
from data_download import is_error_page

CHUNK = 300  # rows per arrow batch (small because HTML rows are ~50 KB each)
RAW_DIR = os.path.join(DATA_DIR, "raw_parquet", "data")


def process_and_merge():
    """Stream new parquets and merge features into existing parquet files."""
    import pyarrow.parquet as pq

    # Load existing data
    url_feat_path   = os.path.join(DATA_DIR, "url_features.parquet")
    html_feat_path  = os.path.join(DATA_DIR, "html_features.parquet")
    phresh_url_path = os.path.join(DATA_DIR, "phresh_urls.parquet")

    existing_url_df  = pd.read_parquet(url_feat_path)  if os.path.exists(url_feat_path)  else pd.DataFrame()
    existing_html_df = pd.read_parquet(html_feat_path) if os.path.exists(html_feat_path) else pd.DataFrame()
    existing_urls_set = set(existing_url_df["url"].astype(str)) if "url" in existing_url_df.columns else set()

    current_total      = len(existing_url_df)
    current_html_phish  = int((existing_html_df["label"] == 1).sum()) if "label" in existing_html_df.columns else 0
    current_html_benign = int((existing_html_df["label"] == 0).sum()) if "label" in existing_html_df.columns else 0
    html_target = MAX_HTML_ROWS // 2

    new_url_recs  = []
    new_html_recs = []
    new_phresh    = []
    skipped_dup   = 0
    skipped_err   = 0

    paths = sorted([
        os.path.join(RAW_DIR, f)
        for f in os.listdir(RAW_DIR)
        if f.endswith(".parquet")
    ])
    print(f"[PROC] {len(paths)} new files, starting from {current_total:,} existing rows")
    print(f"[PROC] HTML caps: phish={current_html_phish}/{html_target}, benign={current_html_benign}/{html_target}")

    t0 = time.time()

    for fpath in paths:
        if current_total >= MAX_URL_ROWS:
            break

        fname = os.path.basename(fpath)
        try:
            pf = pq.ParquetFile(fpath)
            n_rows = pf.metadata.num_rows
        except Exception as e:
            print(f"  [PROC] Cannot open {fname}: {e}")
            continue

        print(f"  [PROC] {fname}: {n_rows:,} rows ...")
        file_new = 0

        try:
            for batch in pf.iter_batches(batch_size=CHUNK):
                if current_total >= MAX_URL_ROWS:
                    break

                d = batch.to_pydict()
                n = len(d.get("url", []))

                for i in range(n):
                    if current_total >= MAX_URL_ROWS:
                        break

                    url       = str(d.get("url", [""])[i] or "")
                    label_str = str(d.get("label", [""])[i] or "").strip().lower()
                    if not url or label_str not in ("benign", "phish"):
                        continue
                    if url in existing_urls_set:
                        skipped_dup += 1
                        continue

                    date_val = d.get("date", [None])[i]
                    html     = str(d.get("html",  [""])[i] or "")
                    label    = 1 if label_str == "phish" else 0

                    if label == 1 and is_error_page(html):
                        skipped_err += 1
                        continue

                    existing_urls_set.add(url)
                    ufeats = extract_url_features(url)
                    ufeats.update({"url": url, "label": label, "date": date_val})
                    new_url_recs.append(ufeats)
                    new_phresh.append({"url": url, "label": label, "date": date_val,
                                        "target": d.get("target", [None])[i]})

                    html_cap = (current_html_phish >= html_target and
                                current_html_benign >= html_target)
                    if not html_cap and html and len(html.strip()) > 100:
                        if label == 1 and current_html_phish < html_target:
                            hf = extract_html_features(html, url)
                            hf.update({"url": url, "label": label, "date": date_val})
                            for c in URL_FEATURE_COLS:
                                hf[c] = ufeats.get(c, 0.0)
                            new_html_recs.append(hf)
                            current_html_phish += 1
                        elif label == 0 and current_html_benign < html_target:
                            hf = extract_html_features(html, url)
                            hf.update({"url": url, "label": label, "date": date_val})
                            for c in URL_FEATURE_COLS:
                                hf[c] = ufeats.get(c, 0.0)
                            new_html_recs.append(hf)
                            current_html_benign += 1

                    current_total += 1
                    file_new += 1

        except Exception as e:
            print(f"  [PROC] Error in {fname}: {e}")

        elapsed = time.time() - t0
        print(f"  [PROC]   +{file_new:,} new rows | total={current_total:,} | "
              f"HTML phish={current_html_phish} benign={current_html_benign} | {elapsed:.0f}s")

    print(f"\n[PROC] Total new: {len(new_url_recs):,} (dup={skipped_dup:,} err={skipped_err:,})")

    if not new_url_recs:
        print("[PROC] Nothing new to add. All URLs were duplicates.")
        return

    # Merge and save
    new_uf = pd.DataFrame(new_url_recs)
    new_hf = pd.DataFrame(new_html_recs)
    new_pr = pd.DataFrame(new_phresh)

    merged_uf = pd.concat([existing_url_df, new_uf], ignore_index=True)
    merged_hf = pd.concat([existing_html_df, new_hf], ignore_index=True) if len(new_hf) else existing_html_df
    ex_pr = pd.read_parquet(phresh_url_path) if os.path.exists(phresh_url_path) else pd.DataFrame()
    merged_pr = pd.concat([ex_pr, new_pr], ignore_index=True)

    merged_uf.to_parquet(url_feat_path,  index=False)
    merged_hf.to_parquet(html_feat_path, index=False)
    merged_pr.to_parquet(phresh_url_path, index=False)

    print(f"[PROC] Saved → url_features={len(merged_uf):,}  html_features={len(merged_hf):,}")
    print(f"  Phishing share: {merged_hf['label'].mean():.4f}")
    if "date" in merged_hf.columns and merged_hf["date"].notna().any():
        print(f"  Date range: {merged_hf['date'].min()} → {merged_hf['date'].max()}")


if __name__ == "__main__":
    if not os.path.exists(RAW_DIR) or not any(f.endswith(".parquet") for f in os.listdir(RAW_DIR)):
        print(f"ERROR: No parquet files found in {RAW_DIR}")
        print("Run download step from data_download.py first.")
        sys.exit(1)

    process_and_merge()

    # Cleanup raw files
    raw_root = os.path.join(DATA_DIR, "raw_parquet")
    if os.path.exists(raw_root):
        shutil.rmtree(raw_root)
        print("[PROC] Cleaned up raw parquet files")
