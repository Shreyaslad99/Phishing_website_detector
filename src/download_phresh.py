"""Download and validate PhreshPhish parquet files with resume and retries."""
import os
import sys
import time
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA_DIR, HF_DATASET_ID

RAW_DIR = os.path.join(DATA_DIR, "raw_parquet")
DATA_SUBDIR = os.path.join(RAW_DIR, "data")


def validate_parquet(path: str) -> bool:
    """Check if parquet file is valid and readable."""
    if not os.path.exists(path):
        return False
    try:
        pf = pq.ParquetFile(path)
        return pf.metadata.num_rows > 0
    except Exception as e:
        print(f"    [VAL] Error validating {os.path.basename(path)}: {e}")
        return False


def download_file_with_retries(filename: str, max_retries: int = 3) -> str:
    """Download single parquet file with resume and retries."""
    from huggingface_hub import hf_hub_download

    target_path = os.path.join(DATA_SUBDIR, os.path.basename(filename))
    if validate_parquet(target_path):
        print(f"  [CACHE] Valid: {os.path.basename(filename)} ({os.path.getsize(target_path)/1024/1024:.1f} MB)")
        return target_path

    for attempt in range(1, max_retries + 1):
        print(f"  [DL] {filename} (attempt {attempt}/{max_retries})...")
        try:
            downloaded_path = hf_hub_download(
                repo_id=HF_DATASET_ID,
                filename=filename,
                repo_type="dataset",
                local_dir=RAW_DIR,
                resume_download=True,
            )
            # Find the actual file
            candidate = target_path if os.path.exists(target_path) else downloaded_path
            if validate_parquet(candidate):
                print(f"  [DL] SUCCESS: {os.path.basename(filename)} ({os.path.getsize(candidate)/1024/1024:.1f} MB)")
                return candidate
            else:
                print(f"  [DL] Downloaded file corrupted: {candidate}")
                if os.path.exists(candidate):
                    os.remove(candidate)
        except Exception as e:
            print(f"  [DL] Attempt {attempt} failed: {e}")
            if attempt < max_retries:
                time.sleep(3 * attempt)

    raise RuntimeError(f"FAILED to download {filename} after {max_retries} attempts.")


def download_and_validate_all(n_files: int = 11):
    """Download and validate train-000 through train-(n_files-1)."""
    os.makedirs(DATA_SUBDIR, exist_ok=True)
    valid_files = []
    for i in range(n_files):
        fname = f"data/train-{i:03d}.parquet"
        local_f = download_file_with_retries(fname, max_retries=3)
        valid_files.append(local_f)

    print(f"\n[DATA] All {len(valid_files)} parquet files validated successfully.")
    for vf in valid_files:
        pf = pq.ParquetFile(vf)
        print(f"  {os.path.basename(vf)}: {pf.metadata.num_rows:,} rows, {os.path.getsize(vf)/1024/1024:.1f} MB")
    return valid_files


if __name__ == "__main__":
    download_and_validate_all(n_files=11)
