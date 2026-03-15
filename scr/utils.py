import json
import os
import re
import string
import time
import unicodedata
from pathlib import Path

import requests
import rootutils
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
from requests_oauthlib import OAuth1

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)

BROWSER_TIMEOUT_MS = 30000
SKIP_WIKI_SECTIONS = {"External_links", "Further_reading", "Sources", "Notes", "See_also", "Media"}


def create_and_save_record(path: Path, id: str, text: str, source: str, url: str, xml: str = None):
    text = unicodedata.normalize("NFKD", text).encode("ascii", errors="ignore").decode("ascii")
    if len(id.split()) > 1:
        id = safe_filename(id)
    record = {
        "id": id,
        "text": text,
        "source": source,
        "url": url,
    }
    if xml is not None:
        record["xml"] = xml
    filename = path / f"{id}.json"
    idx = 1
    while filename.exists():
        filename = path / f"{id}_{idx}.json"
        idx = idx + 1
    filename.write_text(json.dumps(record, indent=4, ensure_ascii=False), encoding="utf-8")
    return filename


def get_request(url: str, params: dict | None = None, return_json: bool = False):
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    if return_json:
        return r.json()
    return BeautifulSoup(r.text, "xml")


def simulate_browser(url: str, timeout_ms: int = BROWSER_TIMEOUT_MS):
    """Fetch a JSON endpoint via a headless Chromium browser."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            context = browser.new_context(user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36"
            ))
            page = context.new_page()
            page.set_default_timeout(timeout_ms)
            page.goto(url, wait_until="networkidle")
            return json.loads(page.evaluate("() => document.body.innerText"))
        except Exception as e:
            pass
        finally:
            browser.close()
    return None


def safe_filename(x: str) -> str:
    """Convert a title to a safe filename slug."""
    x = re.sub(f"[{re.escape(string.punctuation)}]", "", x).strip()
    x = re.sub(r"\s+", "_", x)
    return x.strip()


def fetch_wikipedia_sections(page: str):
    url = "https://en.wikipedia.org/w/api.php"
    result = {}

    # ---- get list of sections ----
    params = {
        "action": "parse",
        "page": page,
        "prop": "sections",
        "format": "json",
        "formatversion": "2",
    }
    headers = {
        "User-Agent": "FallacyRAG (doredeborah3@gmail.com)"
    }

    # ACCESS_TOKEN, ACCESS_SECRET = get_request_token()
    # oauth = OAuth1(
    #     client_key=os.environ["CONSUMER_KEY"],
    #     client_secret=os.environ['CONSUMER_SECRET'],
    #     resource_owner_key=ACCESS_TOKEN,
    #     resource_owner_secret=ACCESS_SECRET,
    # )
    section_response = None
    for attempt in range(5):
        r = requests.get(url, params=params, headers=headers, timeout=15)
        if r.status_code == 429:
            retry = int(r.headers.get("Retry-After", 2))
            time.sleep(retry)
            continue
        if r.status_code != 200:
            time.sleep(2.0 * attempt)
            continue

        data = r.json()

        section_response = data
        break

    if not section_response:
        return {}

    sections = section_response.get("parse", {}).get("sections", [])

    # ---- fetch each section text ----
    for section in sections:
        anchor = section.get("anchor")
        if anchor in SKIP_WIKI_SECTIONS:
            continue

        params = {
            "action": "parse",
            "page": page,
            "prop": "text",
            "section": section["index"],
            "format": "json",
            "formatversion": "2",
        }

        html_response = None
        for attempt in range(5):
            r = requests.get(url, params=params, headers=headers, timeout=15)
            if r.status_code == 429:
                retry = int(r.headers.get("Retry-After", 2))
                time.sleep(retry)
                continue
            if r.status_code != 200:
                time.sleep(2.0 * attempt)
                continue

            data = r.json()

            html_response = data
            break

        if not html_response:
            continue

        raw_html = html_response.get("parse", {}).get("text", "")
        soup = BeautifulSoup(raw_html, "html.parser")

        # remove reference markers
        for ref in soup.select("sup.reference"):
            ref.decompose()

        paragraphs = " ".join(
            p.get_text(" ", strip=True)
            for p in soup.find_all("p")
            if p.get_text(strip=True)
        )

        if paragraphs:
            result[anchor] = paragraphs

    # ---- build final text ----
    text = ""
    for title, section_text in result.items():
        text += f"{title}: {section_text}\n\n"

    return text


def get_request_token():
    url = f"https://en.wikipedia.org/wiki/Special:OAuth/initiate"

    oauth = OAuth1(
        os.environ['CONSUMER_KEY'],
        client_secret=os.environ['CONSUMER_SECRET'],
        callback_uri="oob"
    )

    HEADERS = {"User-Agent": "FallacyRAG OAuth Test"}
    r = requests.get(url, auth=oauth, headers=HEADERS)
    r.raise_for_status()
    data = dict(x.split("=") for x in r.text.split("&"))
    return data["oauth_token"], data["oauth_token_secret"]
