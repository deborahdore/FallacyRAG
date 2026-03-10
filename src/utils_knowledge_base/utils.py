import logging
import re
import time
import unicodedata
from typing import Optional

import requests
from bs4 import BeautifulSoup

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

SKIP_WIKI_SECTIONS = {"External_links", "Further_reading", "Sources", "Notes", "See_also", "Media"}
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/122.0 Safari/537.36",
    "Accept": "application/json,text/plain,*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
}
MAX_RETRIES = 3
RETRY_BACKOFF = 2.0
BROWSER_TIMEOUT_MS = 30_000  # 30 s per page navigation


def safe_filename(title: str) -> str:
    """Convert a title to a safe filename slug."""
    return re.sub(r"\s+", "_", title)


def fetch_wikipedia_sections(page: str) -> dict:
    """Return a dict of {anchor: cleaned_text} for all non-boilerplate sections."""

    def get_with_retry(url, params, max_retries=5, backoff=2.0):
        for attempt in range(max_retries):
            resp = get_from_url(url, params=params)
            if resp is None:
                return None
            if isinstance(resp, dict) and resp.get("error", {}).get("code") == "ratelimited":
                wait = backoff * (2 ** attempt)
                print(f"[429] Rate limited. Retrying in {wait:.1f}s... (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait)
                continue
            return resp
        print(f"[429] Max retries exceeded for page '{page}'.")
        return None

    sections_resp = get_with_retry(
        "https://en.wikipedia.org/w/api.php",
        params={"action": "parse", "page": page, "prop": "sections", "format": "json", "formatversion": "2"},
    )
    if sections_resp is None:
        return {}

    sections = sections_resp.get("parse", {}).get("sections", [])
    result: dict = {}

    for section in sections:
        if section.get("anchor") in SKIP_WIKI_SECTIONS:
            continue

        html_resp = get_with_retry(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "parse",
                "page": page,
                "prop": "text",
                "section": section["index"],
                "format": "json",
                "formatversion": "2",
            },
        )
        if html_resp is None:
            continue

        raw_html = html_resp.get("parse", {}).get("text", "")
        soup = BeautifulSoup(raw_html, "html.parser")
        for ref in soup.select("sup.reference"):
            ref.decompose()

        paragraphs = " ".join(
            p.get_text(" ", strip=True) for p in soup.find_all("p") if p.get_text(strip=True)
        )
        if paragraphs:
            result[f"wiki_{section['anchor']}"] = clean_text(paragraphs)

    return result


def get_from_url(url: str, params: Optional[dict] = None, return_json: bool = True):
    """GET a JSON endpoint with automatic retry/back-off."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=15)
            r.raise_for_status()
            if return_json:
                return r.json()
            return r.text
        except requests.exceptions.HTTPError as e:
            log.warning("HTTP %s for %s (attempt %d/%d)", e.response.status_code, url, attempt, MAX_RETRIES)
        except requests.exceptions.RequestException as e:
            log.warning("Request error for %s: %s (attempt %d/%d)", url, e, attempt, MAX_RETRIES)
        if attempt < MAX_RETRIES:
            time.sleep(RETRY_BACKOFF * attempt)
    log.error("Giving up on %s after %d attempts", url, MAX_RETRIES)
    return None


def clean_text(content) -> str:
    """Strip HTML tags and normalise whitespace."""
    text = BeautifulSoup(str(content), "html.parser").get_text()
    text = re.sub(r"[\t\n\r\f\v]+", " ", text)
    text = re.sub(r" {2,}", " ", text)
    text = unicodedata.normalize("NFKD", text).encode("ascii", errors="ignore").decode("ascii")
    return text.strip()


