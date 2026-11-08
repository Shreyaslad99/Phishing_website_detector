"""HTML feature extraction — structural features from page HTML."""
from bs4 import BeautifulSoup
import tldextract


def _same_domain(url1: str, url2: str) -> bool:
    """Check if two URLs share the same registrable domain."""
    e1 = tldextract.extract(url1)
    e2 = tldextract.extract(url2)
    return e1.domain == e2.domain and e1.suffix == e2.suffix


def extract_html_features(html: str, page_url: str) -> dict:
    """Extract HTML structural features. Returns empty dict if parse fails."""
    feats = {}
    try:
        soup = BeautifulSoup(html, "html.parser")
    except Exception:
        return {k: 0.0 for k in HTML_FEATURE_COLS}

    page_domain = tldextract.extract(page_url)

    # Password fields
    password_inputs = soup.find_all("input", {"type": "password"})
    feats["has_password_field"] = 1.0 if password_inputs else 0.0

    # Form action to other domain
    forms = soup.find_all("form")
    form_cross_domain = 0.0
    for form in forms:
        action = form.get("action", "")
        if action and action.strip():
            if not action.startswith(("http://", "https://")):
                # Relative URL — same domain
                continue
            if not _same_domain(action, page_url):
                form_cross_domain = 1.0
                break
    feats["form_action_cross_domain"] = form_cross_domain

    # External link ratio
    links = soup.find_all("a", href=True)
    external = 0
    total_links = len(links)
    for a in links:
        href = a["href"]
        if href.startswith(("http://", "https://")):
            if not _same_domain(href, page_url):
                external += 1
    feats["external_link_ratio"] = external / max(total_links, 1)

    # Iframes
    feats["iframe_count"] = float(len(soup.find_all("iframe")))

    # Hidden elements
    hidden_count = 0
    for el in soup.find_all(style=True):
        style = el.get("style", "").lower()
        if "display:none" in style.replace(" ", "") or "visibility:hidden" in style.replace(" ", ""):
            hidden_count += 1
    hidden_inputs = soup.find_all("input", {"type": "hidden"})
    feats["hidden_element_count"] = float(hidden_count + len(hidden_inputs))

    # Favicon domain mismatch
    favicon_mismatch = 0.0
    favicons = soup.find_all("link", rel=lambda r: r and "icon" in " ".join(r).lower()) if False else []
    # Broader search
    for link in soup.find_all("link"):
        rel = link.get("rel", [])
        rel_str = " ".join(rel) if isinstance(rel, list) else str(rel)
        if "icon" in rel_str.lower():
            href = link.get("href", "")
            if href.startswith(("http://", "https://")):
                if not _same_domain(href, page_url):
                    favicon_mismatch = 1.0
                    break
    feats["favicon_domain_mismatch"] = favicon_mismatch

    # Title/domain mismatch
    title_mismatch = 0.0
    title = soup.find("title")
    if title and title.text:
        title_text = title.text.strip().lower()
        domain_name = page_domain.domain.lower()
        if domain_name and domain_name not in title_text and len(title_text) > 0:
            title_mismatch = 1.0
    feats["title_domain_mismatch"] = title_mismatch

    # Script count
    feats["script_count"] = float(len(soup.find_all("script")))

    return feats


HTML_FEATURE_COLS = [
    "has_password_field", "form_action_cross_domain", "external_link_ratio",
    "iframe_count", "hidden_element_count", "favicon_domain_mismatch",
    "title_domain_mismatch", "script_count",
]
