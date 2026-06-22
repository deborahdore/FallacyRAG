from __future__ import annotations

import json
import os
import re
import string
import time
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
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
    text = re.sub(r"([A-Za-z])\.([A-Za-z])", r"\1. \2", text)  # space after mid-word periods
    text = re.sub(r"\s+", " ", text).strip()  # collapse newlines/tabs/multiple spaces
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


_BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36"
)


def simulate_browser(url: str, timeout_ms: int = BROWSER_TIMEOUT_MS):
    """Fetch a JSON endpoint via a headless Chromium browser."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            context = browser.new_context(user_agent=_BROWSER_UA)
            page = context.new_page()
            page.set_default_timeout(timeout_ms)
            page.goto(url, wait_until="domcontentloaded")
            return json.loads(page.evaluate("() => document.body.innerText"))
        except Exception:
            pass
        finally:
            browser.close()
    return None


def simulate_browser_batch(urls, max_workers: int = 4,
                           timeout_ms: int = BROWSER_TIMEOUT_MS) -> dict:
    """Fetch many JSON endpoints via headless Chromium, fast.

    Two speedups over calling `simulate_browser` per URL: (1) each worker thread launches the
    browser ONCE and reuses it for its whole shard of URLs (browser launch is the dominant
    cost), and (2) `max_workers` workers run in parallel, each with its own Playwright instance
    (the sync API is per-thread, so instances are not shared across threads). Returns
    {url: parsed_json_or_None}.
    """
    urls = list(dict.fromkeys(urls))  # de-dupe, preserve order
    if not urls:
        return {}
    max_workers = max(1, min(max_workers, len(urls)))
    shards = [urls[i::max_workers] for i in range(max_workers)]

    def _worker(shard: list) -> dict:
        local: dict = {url: None for url in shard}
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                try:
                    context = browser.new_context(user_agent=_BROWSER_UA)
                    page = context.new_page()
                    page.set_default_timeout(timeout_ms)
                    for url in shard:
                        try:
                            page.goto(url, wait_until="domcontentloaded")
                            local[url] = json.loads(page.evaluate("() => document.body.innerText"))
                        except Exception:
                            local[url] = None
                finally:
                    browser.close()
        except Exception as e:
            # Browser couldn't launch (e.g. missing system libs) — skip this shard rather than
            # crashing the whole run, matching the original per-call best-effort behaviour.
            # Only keep the first line; the full Playwright browser log is very verbose.
            first_line = str(e).splitlines()[0] if str(e) else type(e).__name__
            print(f"Headless browser unavailable, skipping {len(shard)} URLs: {first_line}")
        return local

    results: dict = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for shard_result in pool.map(_worker, [s for s in shards if s]):
            results.update(shard_result)
    return results


def safe_filename(x: str) -> str:
    """Convert a title to a safe filename slug."""
    x = re.sub(f"[{re.escape(string.punctuation)}]", "", x).strip()
    x = re.sub(r"\s+", "_", x)
    return x.strip()


def fetch_wikipedia_category_members(category: str) -> list[str]:
    """Return the article-namespace titles that belong to a Wikipedia category.

    Used to build an authoritative allowlist of fallacy articles (e.g.
    "Category:Relevance fallacies") so the knowledge base only ingests
    fallacy/argumentation-relevant pages.
    """
    url = "https://en.wikipedia.org/w/api.php"
    headers = {"User-Agent": "FallacyRAG (doredeborah3@gmail.com)"}
    titles: list[str] = []
    cmcontinue = None

    while True:
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": f"Category:{category}",
            "cmnamespace": 0,  # articles only (skip sub-categories / talk pages)
            "cmlimit": "max",
            "format": "json",
            "formatversion": "2",
        }
        if cmcontinue:
            params["cmcontinue"] = cmcontinue

        response = None
        for attempt in range(5):
            r = requests.get(url, params=params, headers=headers, timeout=15)
            if r.status_code == 429:
                time.sleep(int(r.headers.get("Retry-After", 2)))
                continue
            if r.status_code != 200:
                time.sleep(2.0 * attempt)
                continue
            response = r.json()
            break

        if not response:
            break

        titles.extend(m["title"] for m in response.get("query", {}).get("categorymembers", []))
        cmcontinue = response.get("continue", {}).get("cmcontinue")
        if not cmcontinue:
            break

    return titles


def fetch_wikipedia_links(page: str, limit: int = 200) -> list[str]:
    """Return the titles of article-namespace pages linked from `page`.

    These are the 'connected' pages of a Wikipedia article — e.g. for "Fallacy"
    this yields the individual fallacy types (Ad hominem, Slippery slope, ...).
    Results are restricted to namespace 0 (main/article namespace) and paged
    until `limit` titles have been collected.
    """
    url = "https://en.wikipedia.org/w/api.php"
    headers = {"User-Agent": "FallacyRAG (doredeborah3@gmail.com)"}
    titles: list[str] = []
    plcontinue = None

    while len(titles) < limit:
        params = {
            "action": "query",
            "prop": "links",
            "titles": page,
            "plnamespace": 0,  # main/article namespace only
            "pllimit": "max",
            "format": "json",
            "formatversion": "2",
        }
        if plcontinue:
            params["plcontinue"] = plcontinue

        response = None
        for attempt in range(5):
            r = requests.get(url, params=params, headers=headers, timeout=15)
            if r.status_code == 429:
                time.sleep(int(r.headers.get("Retry-After", 2)))
                continue
            if r.status_code != 200:
                time.sleep(2.0 * attempt)
                continue
            response = r.json()
            break

        if not response:
            break

        for pg in response.get("query", {}).get("pages", []):
            for link in pg.get("links", []):
                titles.append(link["title"])

        plcontinue = response.get("continue", {}).get("plcontinue")
        if not plcontinue:
            break

    return titles[:limit]


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


def fetch_wikipedia_section(page: str, section_name: str) -> str:
    """Return the text of a single named section of a Wikipedia page.

    Unlike `fetch_wikipedia_sections` (which only keeps <p> paragraphs across the whole page),
    this targets one section by its heading (e.g. "Events") and extracts BOTH paragraphs and
    list items (<li>) — necessary for pages like "{year} in the United States" whose events are
    bulleted lists. Requesting a top-level section also returns its nested subsections (e.g. the
    month groupings under "Events"). Returns "" if the section is not found.
    """
    url = "https://en.wikipedia.org/w/api.php"
    headers = {"User-Agent": "FallacyRAG (doredeborah3@gmail.com)"}

    # ---- locate the section index by heading ----
    params = {
        "action": "parse",
        "page": page,
        "prop": "sections",
        "format": "json",
        "formatversion": "2",
    }
    section_response = None
    for attempt in range(5):
        r = requests.get(url, params=params, headers=headers, timeout=15)
        if r.status_code == 429:
            time.sleep(int(r.headers.get("Retry-After", 2)))
            continue
        if r.status_code != 200:
            time.sleep(2.0 * attempt)
            continue
        section_response = r.json()
        break
    if not section_response:
        return ""

    target_index = None
    for section in section_response.get("parse", {}).get("sections", []):
        if section.get("line", "").strip().lower() == section_name.strip().lower():
            target_index = section["index"]
            break
    if target_index is None:
        return ""

    # ---- fetch that section's HTML and extract paragraphs + list items ----
    params = {
        "action": "parse",
        "page": page,
        "prop": "text",
        "section": target_index,
        "format": "json",
        "formatversion": "2",
    }
    html_response = None
    for attempt in range(5):
        r = requests.get(url, params=params, headers=headers, timeout=15)
        if r.status_code == 429:
            time.sleep(int(r.headers.get("Retry-After", 2)))
            continue
        if r.status_code != 200:
            time.sleep(2.0 * attempt)
            continue
        html_response = r.json()
        break
    if not html_response:
        return ""

    soup = BeautifulSoup(html_response.get("parse", {}).get("text", ""), "html.parser")
    for ref in soup.select("sup.reference"):
        ref.decompose()

    parts = [
        el.get_text(" ", strip=True)
        for el in soup.find_all(["p", "li"])
        if el.get_text(strip=True)
    ]
    return "\n".join(parts)


def fetch_wikipedia_extracts(titles: list[str], intro_only: bool = True,
                             batch_size: int = 20, max_workers: int = 8) -> dict[str, str]:
    """Fetch the plain-text of many Wikipedia pages quickly.

    This is dramatically faster than calling `fetch_wikipedia_sections` per page: it uses the
    `extracts` API (prop=extracts&explaintext) and runs requests concurrently.

    intro_only : if True (default) fetch only each page's lead/intro section, which the API
                 allows to be batched up to 20 titles per request — by far the fastest option
                 and a compact, on-topic summary, ideal for bulk link expansion. If False, the
                 full article text is fetched, but Wikipedia caps *full* extracts at one page
                 per request, so `batch_size` is forced to 1 (one request per page, still run
                 concurrently).

    Returns {canonical_title: plaintext} for pages that have content (missing pages skipped).
    """
    if not titles:
        return {}
    url = "https://en.wikipedia.org/w/api.php"
    headers = {"User-Agent": "FallacyRAG (doredeborah3@gmail.com)"}
    if not intro_only:
        batch_size = 1  # full-article extracts are limited to one page per request

    def _fetch_batch(batch: list[str]) -> dict[str, str]:
        params = {
            "action": "query",
            "prop": "extracts",
            "explaintext": "1",
            "exlimit": "max",
            "redirects": "1",
            "titles": "|".join(batch),
            "format": "json",
            "formatversion": "2",
        }
        if intro_only:
            params["exintro"] = "1"
        data = None
        for attempt in range(5):
            try:
                r = requests.get(url, params=params, headers=headers, timeout=30)
            except requests.RequestException:
                time.sleep(2.0 * attempt)
                continue
            if r.status_code == 429:
                time.sleep(int(r.headers.get("Retry-After", 2)))
                continue
            if r.status_code != 200:
                time.sleep(2.0 * attempt)
                continue
            data = r.json()
            break
        if not data:
            return {}
        out: dict[str, str] = {}
        for pg in data.get("query", {}).get("pages", []):
            extract = (pg.get("extract") or "").strip()
            if extract:
                out[pg["title"]] = extract
        return out

    batches = [titles[i:i + batch_size] for i in range(0, len(titles), batch_size)]
    results: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for batch_result in pool.map(_fetch_batch, batches):
            results.update(batch_result)
    return results


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
