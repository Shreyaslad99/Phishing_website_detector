# Phishing URL Detector

URL-based phishing detection with lexical, structural, and HTML features. Compares Logistic Regression, Random Forest, XGBoost, and MLP against URL-only baselines. Designed for CPU-only, 8 GB RAM.

## Quick Start

```bash
# 1. Create virtual environment
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download data and extract features
python src/data_download.py

# 4. Run the full training + evaluation pipeline
python src/pipeline.py

# 5. Start the API
python -m uvicorn src.api:app --reload --port 8000
```

## API Usage

```bash
# Health check
curl http://localhost:8000/api/health

# Check a URL
curl -X POST http://localhost:8000/api/check \
  -H "Content-Type: application/json" \
  -d '{"url": "http://example.com/login"}'
```

Response:
```json
{
  "verdict": "phishing",
  "score": 0.92,
  "stage": 1,
  "reasons": [
    {"text": "URL contains 'login'", "weight": 1.0},
    {"text": "Address uses an IP number instead of a name", "weight": 0.8}
  ],
  "latency_ms": 12.5,
  "note": null
}
```

## Architecture

### Cascade Detection
1. **Stage 1** (URL-only): Fast lexical feature model. If score < 0.15 or > 0.85, answer immediately.
2. **Stage 2** (URL+HTML): If ambiguous, fetch the page (safe fetch with SSRF protection) and use URL+HTML model.

### Features (no blacklist/reputation lookups)
- **URL**: lengths, character counts, digit ratio, subdomain depth, IP host, HTTPS, entropy, suspicious words, brand similarity (rapidfuzz), punycode, TLD length
- **HTML**: password fields, cross-domain form actions, external link ratio, iframes, hidden elements, favicon mismatch, title/domain mismatch, script count

### Models
| Model | Type | Features |
|-------|------|----------|
| B1 | TF-IDF char 3-5gram + LR | URL text |
| B2 | Random Forest | URL features |
| LR | Logistic Regression | URL + HTML |
| RF | Random Forest | URL + HTML |
| XGB | XGBoost (hist) | URL + HTML |
| MLP | MLPClassifier (64,32) | URL + HTML |

### Data
- **Main**: [PhreshPhish](https://huggingface.co/phreshphish/phreshphish) (streaming, 50k URL rows, 20k with HTML)
- **Cross-test**: [PhiUSIIL](https://archive.ics.uci.edu/dataset/967) (never tuned, label inverted)

### Splits
Time-based (old=train, mid=val, newest=test) with domain-grouped no-overlap assertion.

## Project Structure
```
phishing detector/
├── src/
│   ├── config.py          # Constants and paths
│   ├── data_download.py   # Streaming data download + feature extraction
│   ├── features_url.py    # URL feature extraction
│   ├── features_html.py   # HTML feature extraction
│   ├── splits.py          # Time-based + domain-grouped splits
│   ├── train.py           # Model training (baselines + main)
│   ├── evaluate.py        # Metrics, bootstrap CIs, McNemar
│   ├── explain.py         # SHAP explanations + plain-text reasons
│   ├── pipeline.py        # Full pipeline orchestrator
│   └── api.py             # FastAPI cascade detection API
├── tests/                 # Unit tests
├── data/                  # Generated parquet files (gitignored)
├── models/                # Saved models (gitignored)
├── reports/               # Comparison results, SHAP, stability
├── requirements.txt
├── PROGRESS.md
└── README.md
```

## Metrics
PR-AUC, ROC-AUC, recall@0.1%/1% FPR, precision at base rates, Brier score, latency p50/p95, model size. Bootstrap 95% CIs. McNemar pairwise comparison.

## UI

The front‑end UI lives in the **/web** directory. It is a single‑page React app built with Vite, Tailwind CSS, Framer Motion and lucide‑react.

### Development
```bash
# Install front‑end dependencies (run once)
npm install --prefix "d:/SEM 5/phishing detector/web"

# Start the dev server
npm run dev --prefix "d:/SEM 5/phishing detector/web"
```
The dev server starts on `http://localhost:3000/`. Open this URL in a browser to see the full UI.

### Mock mode
The UI can run in mock mode, which returns a static phishing response and shows a **DEMO DATA** badge.
```bash
# Enable mock mode
echo "VITE_MOCK=true" > d:/SEM 5/phishing detector/web/.env
# Restart the dev server
npm run dev --prefix "d:/SEM 5/phishing detector/web"
```
Set `VITE_MOCK=false` (or delete the line) to use the real backend.

### Production build
```bash
npm run build --prefix "d:/SEM 5/phishing detector/web"
```
The built assets are placed in `web/dist/` and can be served by any static file server.

### Project rename
The UI uses the constant `PhishGuard` for the logo text. To rename the project, edit the string in `src/App.tsx`:
```tsx
<h1 className="text-xl font-bold">YourProjectName</h1>
```
Then rebuild the UI.
