"""URL feature extraction — lexical/statistical features only, no reputation lookups."""
import math
import re
from urllib.parse import urlparse, parse_qs
import numpy as np
import tldextract
from rapidfuzz import fuzz

# Small list of well-known domains for brand-similarity comparison
TOP_DOMAINS = [
    "google.com", "facebook.com", "youtube.com", "amazon.com", "microsoft.com",
    "apple.com", "instagram.com", "twitter.com", "linkedin.com", "paypal.com",
    "netflix.com", "ebay.com", "github.com", "dropbox.com", "adobe.com",
    "wellsfargo.com", "bankofamerica.com", "chase.com", "citibank.com",
    "outlook.com", "gmail.com", "yahoo.com", "office.com", "live.com",
    "binance.com", "coinbase.com", "metamask.io", "steampowered.com",
    "spotify.com", "tiktok.com", "whatsapp.com", "reddit.com", "zoom.us",
]

SUSPICIOUS_WORDS = [
    "login", "verify", "secure", "account", "update", "banking",
    "confirm", "password", "wallet", "signin", "activate", "suspend",
    "unlock", "review", "support", "security", "alert", "validate",
]

PUNYCODE_RE = re.compile(r"^xn--", re.IGNORECASE)
IP_RE = re.compile(
    r"^(\d{1,3}\.){3}\d{1,3}$"          # IPv4
    r"|^\[?[0-9a-fA-F:]+\]?$"           # IPv6
)


def _entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = {}
    for ch in s:
        counts[ch] = counts.get(ch, 0) + 1
    n = len(s)
    ent = 0.0
    for c in counts.values():
        p = c / n
        ent -= p * math.log2(p)
    return ent


def _brand_similarity(host: str) -> tuple[float, str]:
    """Return (max similarity score 0-100, matched domain)."""
    best_score = 0.0
    best_match = ""
    # Compare the registrable domain (without TLD) against top domain labels
    ext = tldextract.extract(host)
    reg_label = ext.domain  # e.g. "g00gle" from "g00gle.com"
    for d in TOP_DOMAINS:
        d_label = d.split(".")[0]
        score = fuzz.ratio(reg_label, d_label)
        if score > best_score:
            best_score = score
            best_match = d
    return best_score, best_match


def extract_url_features(url: str) -> dict:
    """Extract all URL features from a single URL string."""
    feats = {}
    try:
        parsed = urlparse(url if "://" in url else f"http://{url}")
    except Exception:
        parsed = urlparse("")

    host = parsed.hostname or ""
    path = parsed.path or ""
    query = parsed.query or ""

    # Lengths
    feats["url_len"] = float(len(url))
    feats["host_len"] = float(len(host))
    feats["path_len"] = float(len(path))
    feats["query_len"] = float(len(query))

    # Character counts
    feats["dot_count"] = float(url.count("."))
    feats["hyphen_count"] = float(url.count("-"))
    feats["underscore_count"] = float(url.count("_"))
    feats["slash_count"] = float(url.count("/"))
    feats["question_count"] = float(url.count("?"))
    feats["equal_count"] = float(url.count("="))
    feats["at_count"] = float(url.count("@"))
    feats["amp_count"] = float(url.count("&"))
    feats["percent_count"] = float(url.count("%"))
    feats["digit_count"] = float(sum(c.isdigit() for c in url))
    feats["special_count"] = float(
        sum(1 for c in url if not c.isalnum() and c not in ".-/_")
    )

    # Ratios
    letters = sum(c.isalpha() for c in url)
    feats["digit_letter_ratio"] = float(feats["digit_count"] / max(letters, 1))

    # Domain parsing
    ext = tldextract.extract(host)
    subdomain = ext.subdomain
    feats["subdomain_count"] = float(subdomain.count(".") + 1 if subdomain else 0)
    feats["tld_len"] = float(len(ext.suffix))
    feats["punycode"] = 1.0 if PUNYCODE_RE.search(host) else 0.0
    feats["domain_len"] = float(len(ext.domain))

    # IP host
    feats["ip_host"] = 1.0 if IP_RE.match(host) else 0.0

    # Scheme
    feats["https"] = 1.0 if parsed.scheme == "https" else 0.0
    feats["has_port"] = 1.0 if parsed.port is not None else 0.0

    # Path / query structure
    path_segs = [s for s in path.split("/") if s]
    feats["path_segments"] = float(len(path_segs))
    feats["param_count"] = float(len(parse_qs(query)))

    # Entropy
    feats["host_entropy"] = _entropy(host)
    feats["path_entropy"] = _entropy(path)

    # Suspicious words
    url_lower = url.lower()
    feats["suspicious_word_count"] = float(
        sum(1 for w in SUSPICIOUS_WORDS if w in url_lower)
    )
    feats["suspicious_in_subdomain"] = 1.0 if any(
        w in subdomain.lower() for w in SUSPICIOUS_WORDS
    ) else 0.0

    # Brand similarity
    brand_score, brand_match = _brand_similarity(host)
    feats["brand_sim_score"] = float(brand_score)
    feats["brand_sim_match"] = brand_match  # keep as string for explanations

    return feats


# Column name lists for convenience
URL_FEATURE_COLS = [
    "url_len", "host_len", "path_len", "query_len",
    "dot_count", "hyphen_count", "underscore_count", "slash_count",
    "question_count", "equal_count", "at_count", "amp_count",
    "percent_count", "digit_count", "special_count", "digit_letter_ratio",
    "subdomain_count", "tld_len", "punycode", "domain_len",
    "ip_host", "https", "has_port", "path_segments", "param_count",
    "host_entropy", "path_entropy", "suspicious_word_count",
    "suspicious_in_subdomain", "brand_sim_score",
]
