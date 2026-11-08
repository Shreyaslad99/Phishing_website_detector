# PROGRESS — Phishing URL Detector

## Status: COMPLETE

## Phase Tracker
- [x] 0. Setup (folders, venv, requirements) — Python 3.14, lxml skipped
- [x] 1. Data download — 12,698 PhreshPhish rows (3 parquet files), 235,795 PhiUSIIL cross-test rows
- [x] 2. Feature extraction — URL(30) + HTML(8) features, saved to parquet
- [x] 3. Splits — time-based, domain-grouped, no overlap (9.8k train / 1.2k val / 1.7k test)
- [x] 4. Models — B1(TF-IDF+LR), B2(URL+RF), LR, RF, XGB, MLP — all trained with class weights
- [x] 5. Evaluation — PR-AUC, ROC-AUC, Brier, ECE(calibration), F1@val-tuned-threshold, recall@FPR, precision@base-rates, latency, model size, bootstrap 95% CI, McNemar
- [x] 6. API — FastAPI cascade (B1 stage1 → RF stage2), SSRF-safe fetch with redirect validation
- [x] 7. Reports — comparison.csv/md, SHAP summaries, k-fold stability, cross-test results, README

## Key Results (Test Set, 1,672 rows)
- Best PR-AUC: RF (0.7845), XGB (0.7804), MLP (0.7773)
- Best ECE (calibration): RF (0.047), B1 (0.054), B2 (0.058)
- Best F1@threshold: RF (0.726), XGB (0.717), MLP (0.719)
- Val-tuned thresholds: RF=0.397, XGB=0.195, MLP=0.361
- Best cross-test PR-AUC: B1 (0.996), LR (0.908), MLP (0.901)
- RF overfits (test=0.785, cross-test=0.656); B1 generalizes best
- K-fold stability (5 folds): 4 features in ALL folds — script_count, path_len, path_entropy, external_link_ratio

## Deviations from Spec
- Python 3.14 (not 3.11 — only version available)
- lxml skipped (no Python 3.14 wheel — using html.parser instead)
- Direct parquet download instead of streaming (network instability — STREAM_CHUNK constant defined but unused)
- MAX_URL_ROWS lowered from 100k to 50k (then only 12.7k obtained due to train-003 download failure)

## Notes
- CPU only, 8 GB RAM, no pretrained models
- All 21 unit tests pass
- API tested with real URLs (legitimate, suspicious, phishing-like)
- SHAP TreeExplainer for tree models, coefficients for LR
- Top-3 features mapped to plain sentences in API reasons
