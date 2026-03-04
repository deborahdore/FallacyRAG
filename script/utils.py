import logging
import re
import time
from typing import Optional

import requests
from bs4 import BeautifulSoup

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def safe_filename(title: str) -> str:
    """Convert a title to a safe filename slug."""
    return re.sub(r"\s+", "_", title)


def fetch_wikipedia_sections(page: str) -> dict:
    """Return a dict of {anchor: cleaned_text} for all non-boilerplate sections."""
    sections_resp = get_from_url(
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

        html_resp = get_from_url(
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


def get_from_url(url: str, params: Optional[dict] = None) -> Optional[dict]:
    """GET a JSON endpoint with automatic retry/back-off."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=15)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.HTTPError as e:
            log.warning("HTTP %s for %s (attempt %d/%d)", e.response.status_code, url, attempt, MAX_RETRIES)
        except requests.exceptions.RequestException as e:
            log.warning("Request error for %s: %s (attempt %d/%d)", url, e, attempt, MAX_RETRIES)
        if attempt < MAX_RETRIES:
            time.sleep(RETRY_BACKOFF * attempt)
    log.error("Giving up on %s after %d attempts", url, MAX_RETRIES)
    return None


SKIP_WIKI_SECTIONS = {"External_links", "Further_reading", "Sources", "Notes", "See_also", "Media"}


def clean_text(content) -> str:
    """Strip HTML tags and normalise whitespace."""
    text = BeautifulSoup(str(content), "html.parser").get_text()
    text = re.sub(r"[\t\n\r\f\v]+", " ", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


MAX_RETRIES = 3
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/122.0 Safari/537.36",
    "Accept": "application/json,text/plain,*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
}
RETRY_BACKOFF = 2.0
