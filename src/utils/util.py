from __future__ import annotations

import json
import os
import re
import string
import time
import unicodedata
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import requests
import rootutils
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
from tqdm import tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)

BROWSER_TIMEOUT_MS = 30000
SKIP_WIKI_SECTIONS = {"External_links", "Further_reading", "Sources", "Notes", "See_also", "Media"}


def normalize_text(text):
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[‘’´`]", "'", text)  # apostrophes
    text = re.sub(r"[“”]", '"', text)  # double quotes
    text = re.sub(r"[–—]", " - ", text)  # dashes — pad so they tokenize as a separate token
    # text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"([A-Za-z])\.([A-Za-z])", r"\1. \2", text).strip()
    return text


def create_and_save_record(path: Path,
                           id: str,
                           text: str,
                           source: str,
                           url: str,
                           xml: str = None,
                           year: int = 1900):
    text = normalize_text(text)

    if len(id.split()) > 1:
        id = safe_filename(id)
    record = {
        "id": id,
        "text": text,
        "source": source,
        "url": url,
        "year": year
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


def post_request(url: str, params: dict | None = None, json_obj: dict = None, return_json: bool = False):
    while True:
        response = requests.post(
            url,
            params=params,
            json=json_obj,
            timeout=120
        )
        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After", "60")
            wait = float(retry_after)
            print(f"Rate limited (429). Retrying after {wait}s...")
            time.sleep(wait)
            continue
        response.raise_for_status()
        if return_json:
            return response.json()
        return BeautifulSoup(response.text, "xml")


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


# def get_request_token():
#     url = f"https://en.wikipedia.org/wiki/Special:OAuth/initiate"
#
#     oauth = OAuth1(
#         os.environ['CONSUMER_KEY'],
#         client_secret=os.environ['CONSUMER_SECRET'],
#         callback_uri="oob"
#     )
#
#     HEADERS = {"User-Agent": "FallacyRAG OAuth Test"}
#     r = requests.get(url, auth=oauth, headers=HEADERS)
#     r.raise_for_status()
#     data = dict(x.split("=") for x in r.text.split("&"))
#     return data["oauth_token"], data["oauth_token_secret"]


def analyze_paragraph_lengths(original_path: str = "./knowledge_base"):
    counter = Counter()
    sections = os.listdir(original_path)

    for section in sections:
        path = os.path.join(original_path, section)
        if not os.path.isdir(path):
            continue

        files = os.listdir(path)

        for file in tqdm(files, total=len(files), desc=f"Processing {section}"):
            file_path = os.path.join(path, file)

            with open(file_path, "r") as f:
                j = json.load(f)

            paragraphs = j.get("text", "").split("\n\n")

            for para in paragraphs:
                length = len(para.split())
                if length > 0:
                    counter[length] += 1

    lengths = np.array(list(counter.keys()))
    counts = np.array(list(counter.values()))

    expanded = np.repeat(lengths, counts)

    print("---- Stats ----")
    print("Min:", expanded.min())
    print("Max:", expanded.max())
    print("Mean:", expanded.mean())
    print("Median (P50):", np.percentile(expanded, 50))
    print("P75:", np.percentile(expanded, 75))
    print("P90:", np.percentile(expanded, 90))
    print("P95:", np.percentile(expanded, 95))

    plot_paragraph_cdf(lengths, counts)


def plot_paragraph_cdf(lengths, counts, output_path="./data/paragraph_length_distribution.pdf"):
    expanded = np.repeat(lengths, counts)
    sorted_lengths = np.sort(expanded)
    cdf = np.arange(1, len(sorted_lengths) + 1) / len(sorted_lengths)

    thresholds = [
        (800, "#4a9e5c", "800 tokens  — {pct:.0f}%"),
        (1200, "#c47f1a", "1200 tokens — {pct:.0f}%"),
        (2048, "#c0392b", "2048 tokens — {pct:.0f}%"),
    ]

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#fafafa")
    ax.set_facecolor("#fafafa")

    ax.plot(sorted_lengths, cdf, color="#2166ac", linewidth=2, zorder=3)
    ax.fill_between(sorted_lengths, cdf, alpha=0.08, color="#2166ac")

    for x, color, label_fmt in thresholds:
        pct = float(np.searchsorted(sorted_lengths, x)) / len(sorted_lengths) * 100
        ax.axvline(x, color=color, linewidth=1.4, linestyle="--", zorder=2)
        ax.text(
            x * 1.07, 0.04 + thresholds.index((x, color, label_fmt)) * 0.07,
            label_fmt.format(pct=pct),
            color=color, fontsize=9, va="bottom",
        )

    percentiles = [50, 75, 90, 95]
    pct_values = np.percentile(expanded, percentiles)
    for p, v in zip(percentiles, pct_values):
        ax.annotate(
            f"P{p}\n{v:.0f}",
            xy=(v, p / 100),
            xytext=(v * 1.5, p / 100 - 0.08),
            fontsize=8, color="#555",
            arrowprops=dict(arrowstyle="-", color="#aaa", lw=0.8),
            ha="left",
        )

    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))

    ax.set_xlabel("Token length (log scale)", fontsize=11)
    ax.set_ylabel("Cumulative proportion", fontsize=11)
    ax.set_title("CDF of paragraph lengths", fontsize=13, fontweight="normal", pad=12)

    ax.set_xlim(left=max(1, sorted_lengths.min() * 0.8))
    ax.set_ylim(0, 1.05)
    ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.3, alpha=0.3)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    stats_text = (
        f"n={len(expanded):,}  |  "
        f"median={np.median(expanded):.0f}  |  "
        f"P90={np.percentile(expanded, 90):.0f}  |  "
        f"P95={np.percentile(expanded, 95):.0f}  |  "
        f"max={expanded.max():.0f}"
    )
    fig.text(0.5, 0.01, stats_text, ha="center", fontsize=8, color="#888")

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()


def get_item_ids(url, items, conditional='True'):
    # Check that the query URL is not an item or resource link.
    exclude = ["loc.gov/item", "loc.gov/resource"]
    if any(string in url for string in exclude):
        raise NameError('Your URL points directly to an item or '
                        'resource page (you can tell because "item" '
                        'or "resource" is in the URL). Please use '
                        'a search URL instead. For example, instead '
                        'of \"https://www.loc.gov/item/2009581123/\", '
                        'try \"https://www.loc.gov/maps/?q=2009581123\". ')

    # request pages of 100 results at a time
    params = {"fo": "json", "c": 5, "at": "results,pagination"}
    call = requests.get(url, params=params)
    # Check that the API request was successful
    if (call.status_code == 200) & ('json' in call.headers.get('content-type')):
        data = call.json()
        results = data['results']
        for result in results:
            # Filter out anything that's a collection or web page
            filter_out = ("collection" in result.get("original_format")) \
                         or ("web page" in result.get("original_format")) \
                         or (eval(conditional) == False)
            if not filter_out:
                # Get the link to the item record
                if result.get("id"):
                    item = result.get("id")
                    # Filter out links to Catalog or other platforms
                    if item.startswith("http://www.loc.gov/resource"):
                        resource = item  # Assign item to resource
                        items.append(resource)
                    if item.startswith("http://www.loc.gov/item"):
                        items.append(item)
        # Repeat the loop on the next page, unless we're on the last page.
        # if data["pagination"]["next"] is not None:
        #     next_url = data["pagination"]["next"]
        #     get_item_ids(next_url, items, conditional)

        return items
    else:
        print('There was a problem. Try running the cell again, or check your searchURL.')
