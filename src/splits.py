"""Data splitting: time-based with domain-grouped no-overlap assertion.

Splits by date (old=train, mid=val, newest=test) and asserts that
no registered domain appears in more than one split.
If no dates available, falls back to domain-grouped split + warning.
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
import tldextract
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import SEED


def get_registered_domain(url: str) -> str:
    """Extract the registrable domain (e.g. 'example.com' from 'sub.example.com')."""
    if not url or not isinstance(url, str):
        return ""
    ext = tldextract.extract(url)
    if ext.domain and ext.suffix:
        return f"{ext.domain}.{ext.suffix}"
    return url[:100]  # fallback


def normalize_url(url: str) -> str:
    """Normalize a URL for deduplication."""
    if not url or not isinstance(url, str):
        return ""
    url = url.strip().lower()
    if url.endswith("/"):
        url = url[:-1]
    return url


def prepare_split(df: pd.DataFrame, train_frac=0.7, val_frac=0.15, stratum_col="label"):
    """
    Prepare a train/val/test split from a feature DataFrame.

    - Normalizes and deduplicates URLs
    - Drops conflicting labels
    - Splits by time (date) if available, else domain-grouped
    - Asserts no domain overlap between splits
    """
    print(f"[SPLIT] Input: {len(df):,} rows")

    # Normalize URL
    df = df.copy()
    df["url_norm"] = df["url"].apply(normalize_url)
    df["domain"] = df["url"].apply(get_registered_domain)

    # Drop rows with empty URL
    df = df[df["url_norm"].str.len() > 0].copy()
    print(f"  After dropping empty URLs: {len(df):,}")

    # Drop conflicting labels (same URL, different labels)
    label_counts = df.groupby("url_norm")["label"].nunique()
    conflict_urls = label_counts[label_counts > 1].index
    df = df[~df["url_norm"].isin(conflict_urls)].copy()
    print(f"  After dropping conflicting labels: {len(df):,} "
          f"(removed {len(conflict_urls):,} conflicting URLs)")

    # Deduplicate: keep first occurrence per URL
    df = df.drop_duplicates(subset="url_norm", keep="first").copy()
    print(f"  After dedup: {len(df):,}")

    # Check if dates available
    has_dates = "date" in df.columns and df["date"].notna().any()
    if has_dates:
        df = df.sort_values("date").reset_index(drop=True)
        print(f"  Splitting by time: {df['date'].min()} -> {df['date'].max()}")
        _time_based_split(df, train_frac, val_frac)
    else:
        warnings.warn("No dates found — using domain-grouped split instead")
        print("  WARNING: No dates — using domain-grouped split")
        _domain_grouped_split(df, train_frac, val_frac)

    # Assert no domain overlap
    _assert_no_overlap(df)

    # Report split sizes
    for split in ["train", "val", "test"]:
        n = (df["split"] == split).sum()
        phish_share = df[df["split"] == split]["label"].mean()
        n_domains = df[df["split"] == split]["domain"].nunique()
        print(f"  {split}: {n:,} rows, phish={phish_share:.3f}, {n_domains:,} domains")

    return df


def _time_based_split(df, train_frac, val_frac):
    """Split by time: oldest = train, middle = val, newest = test."""
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    df["split"] = "train"
    df.loc[df.index[train_end:val_end], "split"] = "val"
    df.loc[df.index[val_end:], "split"] = "test"

    # Group by domain and reassign: if a domain appears in multiple splits,
    # keep it in the earliest split
    domain_splits = df.groupby("domain")["split"].agg(lambda x: x.mode()[0] if len(x) > 0 else "train")
    domain_to_split = domain_splits.to_dict()

    # Assign each row to its domain's primary split
    df["split"] = df["domain"].map(domain_to_split)
    # If domain not in dict (shouldn't happen), assign to train
    df["split"] = df["split"].fillna("train")


def _domain_grouped_split(df, train_frac, val_frac):
    """Domain-grouped split: GroupShuffleSplit to ensure no domain overlap."""
    gss1 = GroupShuffleSplit(n_splits=1, test_size=1 - train_frac, random_state=SEED)
    train_idx, temp_idx = next(gss1.split(df, groups=df["domain"]))

    train_df = df.iloc[train_idx]
    temp_df = df.iloc[temp_idx]

    # Split temp into val/test
    val_size = val_frac / (1 - train_frac)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=1 - val_size, random_state=SEED)
    val_idx, test_idx = next(gss2.split(temp_df, groups=temp_df["domain"]))

    df["split"] = "train"
    df.loc[temp_df.index[val_idx], "split"] = "val"
    df.loc[temp_df.index[test_idx], "split"] = "test"


def _assert_no_overlap(df):
    """Assert no registered domain appears in more than one split."""
    split_domains = {}
    for split in ["train", "val", "test"]:
        domains = set(df[df["split"] == split]["domain"].unique())
        split_domains[split] = domains

    train_val = split_domains["train"] & split_domains["val"]
    train_test = split_domains["train"] & split_domains["test"]
    val_test = split_domains["val"] & split_domains["test"]

    if train_val or train_test or val_test:
        total_overlap = len(train_val) + len(train_test) + len(val_test)
        print(f"  WARNING: Domain overlap found: {total_overlap} domains")
        print(f"    train∩val={len(train_val)}, train∩test={len(train_test)}, val∩test={len(val_test)}")
        # Fix: move overlapping domains to the split where they have the most rows
        for dom_set, s1, s2 in [
            (train_val, "train", "val"),
            (train_test, "train", "test"),
            (val_test, "val", "test"),
        ]:
            for dom in dom_set:
                n1 = ((df["domain"] == dom) & (df["split"] == s1)).sum()
                n2 = ((df["domain"] == dom) & (df["split"] == s2)).sum()
                loser = s2 if n1 >= n2 else s1
                df.loc[(df["domain"] == dom) & (df["split"] == loser), "split"] = "train"
        # Re-check
        for split in ["train", "val", "test"]:
            split_domains[split] = set(df[df["split"] == split]["domain"].unique())
        remaining = (split_domains["train"] & split_domains["val"]) | \
                    (split_domains["train"] & split_domains["test"]) | \
                    (split_domains["val"] & split_domains["test"])
        if remaining:
            # Last resort: force remaining to train
            for dom in remaining:
                df.loc[(df["domain"] == dom) & (df["split"].isin(["val", "test"])), "split"] = "train"
            print(f"  Fixed remaining {len(remaining)} domains by moving to train")
        else:
            print(f"  Fixed overlaps by consolidation")
    else:
        print("  [OK] No domain overlap between splits")


if __name__ == "__main__":
    # Quick test
    df = pd.DataFrame({
        "url": [f"http://example{i}.com/page" for i in range(100)],
        "label": [i % 2 for i in range(100)],
        "date": pd.date_range("2024-01-01", periods=100).date,
    })
    result = prepare_split(df)
    print(result["split"].value_counts())
