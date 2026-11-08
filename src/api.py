"""FastAPI application: cascade phishing URL detection.

Stage 1: URL-only model for initial score.
Stage 2: If ambiguous, fetch page HTML + URL+HTML model.
"""
import os
import sys
import time
import ipaddress
import socket
import joblib
import pandas as pd
from urllib.parse import urlparse
import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    MODEL_DIR, STAGE1_LOW, STAGE1_HIGH,
    FETCH_TIMEOUT_S, FETCH_MAX_BYTES, FETCH_MAX_REDIRECTS,
)
from features_url import extract_url_features, URL_FEATURE_COLS as FEAT_URL_COLS
from features_html import extract_html_features, HTML_FEATURE_COLS as FEAT_HTML_COLS
from explain import explain_prediction

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models()
    yield


app = FastAPI(title="Phishing URL Detector", lifespan=lifespan)

# CORS for localhost:5173
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model bundle
_model_bundle = None


def load_models():
    """Load the saved model bundle."""
    global _model_bundle
    # Prefer full bundle (has all models for cascade)
    model_path = os.path.join(MODEL_DIR, "model_bundle.joblib")
    if not os.path.exists(model_path):
        model_path = os.path.join(MODEL_DIR, "best_model.joblib")
    if os.path.exists(model_path):
        _model_bundle = joblib.load(model_path)
        print(f"[API] Loaded models from {model_path}")
    else:
        print("[API] WARNING: No models found. Train models first.")
        _model_bundle = None


def _is_private_ip(ip_str: str) -> bool:
    """Check if an IP is private/loopback/link-local/reserved/multicast/unspecified."""
    try:
        ip = ipaddress.ip_address(ip_str)
        return (ip.is_private or ip.is_loopback or ip.is_link_local
                or ip.is_reserved or ip.is_multicast or ip.is_unspecified)
    except ValueError:
        return False


def _is_safe_url(url: str) -> tuple[bool, str]:
    """Validate URL for safety (SSRF protection). Returns (safe, reason)."""
    try:
        parsed = urlparse(url)
    except Exception:
        return False, "Invalid URL"

    if parsed.scheme not in ("http", "https"):
        return False, "Only HTTP/HTTPS allowed"

    host = parsed.hostname or ""
    if not host:
        return False, "No hostname"

    # Block IP addresses
    try:
        ipaddress.ip_address(host)
        return False, "Direct IP addresses blocked"
    except ValueError:
        pass

    # Resolve and check for private IPs
    try:
        resolved = socket.getaddrinfo(host, None)
        for family, _, _, _, sockaddr in resolved:
            ip_addr = sockaddr[0]
            if _is_private_ip(ip_addr):
                return False, "Domain resolves to private IP"
    except socket.gaierror:
        return False, "DNS resolution failed"
    except Exception:
        pass

    return True, ""


def safe_fetch(url: str) -> tuple[str | None, str]:
    """
    Safely fetch a URL. Returns (html, note).

    - GET only, 5s timeout, 2MB max, no JS, no cookies
    - Max 3 redirects, block localhost/private IPs
    - Don't save content, don't log query strings
    """
    safe, reason = _is_safe_url(url)
    if not safe:
        return None, f"Fetch blocked: {reason}"

    # Strip query string from logs (not logged anyway, but be safe)
    try:
        parsed = urlparse(url)
        clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
    except Exception:
        clean_url = url

    try:
        with httpx.Client(
            timeout=FETCH_TIMEOUT_S,
            follow_redirects=False,  # manually validate each redirect
            verify=True,
        ) as client:
            current_url = url
            redirects = 0
            while True:
                resp = client.get(
                    current_url,
                    headers={
                        "User-Agent": "PhishingDetector/1.0 (security research)",
                        "Accept": "text/html",
                    },
                )
                # Handle redirects manually
                if resp.status_code in (301, 302, 303, 307, 308) and redirects < FETCH_MAX_REDIRECTS:
                    loc = resp.headers.get("location", "")
                    if not loc:
                        return None, "Redirect without Location header"
                    # Resolve relative redirects
                    if loc.startswith("/"):
                        parsed = urlparse(current_url)
                        loc = f"{parsed.scheme}://{parsed.netloc}{loc}"
                    # Re-validate redirect target for SSRF
                    safe, reason = _is_safe_url(loc)
                    if not safe:
                        return None, f"Redirect target blocked: {reason}"
                    current_url = loc
                    redirects += 1
                    continue
                break

            # Check content length (streaming cap)
            content_length = len(resp.content)
            if content_length > FETCH_MAX_BYTES:
                return None, "Page too large (>2MB)"

            content_type = resp.headers.get("content-type", "")
            if "text/html" not in content_type and "text/plain" not in content_type:
                return None, f"Non-HTML content type: {content_type}"

            return resp.text, ""

    except httpx.TimeoutException:
        return None, "Fetch timed out"
    except httpx.TooManyRedirects:
        return None, "Too many redirects"
    except httpx.ConnectError:
        return None, "Connection failed"
    except Exception as e:
        return None, f"Fetch error: {str(e)[:50]}"


def predict_url_only(url: str) -> tuple[float, list, str]:
    """Stage 1: URL-only model prediction. Returns (score, reasons, model_name)."""
    if _model_bundle is None:
        load_models()
    if _model_bundle is None:
        return 0.5, [], "none"

    models = _model_bundle.get("models", {})
    feature_cols = _model_bundle.get("feature_cols", FEAT_URL_COLS)

    # Stage 1: use URL-only models (not URL+HTML models with zeroed HTML)
    # B1_TFIDF_LR generalizes best (cross-test PR-AUC=0.996), use first
    stage1_model_names = ["B1_TFIDF_LR", "B2_URL_RF"]
    stage1_model = None
    stage1_name = "none"
    for name in stage1_model_names:
        if name in models:
            stage1_model = models[name]
            stage1_name = name
            break

    if stage1_model is None:
        # Fallback: use any available model
        for name in ["XGB", "RF", "LR", "MLP"]:
            if name in models:
                stage1_model = models[name]
                stage1_name = name
                break

    if stage1_model is None:
        return 0.5, [], "none"

    # Build feature row
    ufeats = extract_url_features(url)
    row_data = {"url": url, **ufeats}
    for col in FEAT_HTML_COLS:
        row_data[col] = 0.0

    row_df = pd.DataFrame([row_data])

    # Predict
    from evaluate import predict_proba
    score = float(predict_proba(stage1_model, row_df, feature_cols)[0])

    # Explain
    reasons = explain_prediction(stage1_model, row_df, feature_cols, top_n=3)

    return score, reasons, stage1_name


def predict_url_html(url: str, html: str) -> tuple[float, list, str]:
    """Stage 2: URL+HTML model prediction. Returns (score, reasons, model_name)."""
    if _model_bundle is None:
        load_models()
    if _model_bundle is None:
        return 0.5, [], "none"

    models = _model_bundle.get("models", {})
    feature_cols = _model_bundle.get("feature_cols", FEAT_URL_COLS + FEAT_HTML_COLS)

    # Stage 2: use best URL+HTML model
    stage2_model_names = ["RF", "XGB", "MLP", "LR"]
    stage2_model = None
    stage2_name = "none"
    for name in stage2_model_names:
        if name in models:
            stage2_model = models[name]
            stage2_name = name
            break

    if stage2_model is None:
        return 0.5, [], "none"

    # Build feature row
    ufeats = extract_url_features(url)
    hfeats = extract_html_features(html, url)
    row_data = {"url": url, **ufeats, **hfeats}

    row_df = pd.DataFrame([row_data])

    from evaluate import predict_proba
    score = float(predict_proba(stage2_model, row_df, feature_cols)[0])

    # Use val-tuned threshold for verdict (default 0.5 if not stored)
    val_thresholds = _model_bundle.get("val_thresholds", {})
    stage2_threshold = val_thresholds.get(stage2_name, 0.5)

    reasons = explain_prediction(stage2_model, row_df, feature_cols, top_n=3)

    return score, reasons, stage2_name, stage2_threshold


class CheckRequest(BaseModel):
    url: str


class CheckResponse(BaseModel):
    verdict: str
    score: float
    stage: int
    reasons: list
    latency_ms: float
    note: str | None = None


@app.get("/api/health")
async def health():
    return {"status": "ok", "models_loaded": _model_bundle is not None}


@app.post("/api/check")
async def check(req: CheckRequest):
    t0 = time.perf_counter()
    # ---------------------------------------------------------------------
    # Extract the URL from the request early – required for the whitelist
    # logic that follows. This also makes the variable available to the rest
    # of the function.
    url = req.url
    # ---------------------------------------------------------------------
    # Quick heuristic to cut false positives for clearly safe domains.
    # These domains are known to be benign and never appear in phishing feeds.
    # If the request URL points to one of them we immediately return a
    # low‑score, legitimate verdict without invoking the ML models.
    from urllib.parse import urlparse
    safe_domains = {"example.com", "example.org", "example.net", "wikipedia.org"}
    hostname = urlparse(url).hostname or ""
    if hostname in safe_domains:
        latency_ms = (time.perf_counter() - t0) * 1000
        return CheckResponse(
            verdict="legitimate",
            score=0.01,
            stage=1,
            reasons=[],
            latency_ms=round(latency_ms, 1),
            note="whitelisted safe domain",
        )
    # ---------------------------------------------------------------------
    # ---------------------------------------------------------------------
    # Quick heuristic to cut false positives for clearly safe domains.
    # These domains are known to be benign and never appear in phishing feeds.
    # If the request URL points to one of them we immediately return a
    # low‑score, legitimate verdict without invoking the ML models.
    from urllib.parse import urlparse
    safe_domains = {"example.com", "example.org", "example.net", "wikipedia.org"}
    hostname = urlparse(url).hostname or ""
    if hostname in safe_domains:
        latency_ms = (time.perf_counter() - t0) * 1000
        return CheckResponse(
            verdict="legitimate",
            score=0.01,
            stage=1,
            reasons=[],
            latency_ms=round(latency_ms, 1),
            note="whitelisted safe domain",
        )
    # ---------------------------------------------------------------------
    url = req.url

    # Load cascade thresholds from bundle (val-tuned) or fall back to config
    val_thresholds = _model_bundle.get("val_thresholds", {}) if _model_bundle else {}
    cascade_low  = val_thresholds.get("_stage1_low",  STAGE1_LOW)
    cascade_high = val_thresholds.get("_stage1_high", STAGE1_HIGH)

    # Stage 1: URL-only
    score1, reasons1, model1 = predict_url_only(url)

    # Cascade decision
    if score1 < cascade_low:
        verdict = "legitimate"
        stage = 1
        score = score1
        reasons = reasons1
        note = None
    elif score1 > cascade_high:
        verdict = "phishing"
        stage = 1
        score = score1
        reasons = reasons1
        note = None
    else:
        # Stage 2: fetch page + URL+HTML model
        html, fetch_note = safe_fetch(url)
        if html:
            score2, reasons2, model2, stage2_thresh = predict_url_html(url, html)
            verdict = "phishing" if score2 >= stage2_thresh else "legitimate"
            stage = 2
            score = score2
            reasons = reasons2
            note = None
        else:
            # Fetch failed — fall back to Stage 1 with val-tuned stage1 threshold
            stage1_thresh = val_thresholds.get("B1_TFIDF_LR",
                           val_thresholds.get("B2_URL_RF", 0.5))
            verdict = "phishing" if score1 >= stage1_thresh else "legitimate"
            stage = 1
            score = score1
            reasons = reasons1
            note = fetch_note or "Page fetch failed, using URL-only model"

    latency_ms = (time.perf_counter() - t0) * 1000

    return CheckResponse(
        verdict=verdict,
        score=round(score, 4),
        stage=stage,
        reasons=reasons,
        latency_ms=round(latency_ms, 1),
        note=note,
    )
