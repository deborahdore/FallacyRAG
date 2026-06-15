import json
import logging
import os
import re
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import easyocr
import numpy as np
import pandas as pd
import pytesseract
import requests
import rootutils
from datasets import load_dataset
from pdf2image import convert_from_path
from tqdm import tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)
from src.utils.util import create_and_save_record, get_request, simulate_browser, safe_filename, \
    fetch_wikipedia_sections, fetch_wikipedia_links, fetch_wikipedia_category_members, \
    post_request, get_item_ids

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# =================== PATH =================== #
PERSIST_DIRECTORY = "./chroma_db/"
KNOWLEDGE_BASE_PATH = "./knowledge_base"

GOV_MANUAL_URL = "https://www.govinfo.gov/content/pkg/GOVMAN-2025-01-13/xml/GOVMAN-2025-01-13.xml"
BILLS_URL = "https://www.govinfo.gov/sitemap/BILLS_sitemap_index.xml"
PLAWS_URL = "https://www.govinfo.gov/sitemap/PLAW_sitemap_index.xml"
GOVINFO_URL = "https://api.govinfo.gov/search"
UDHR_PATH = "./misc/udhr.pdf"

TOPIC_RESULTS_PATH = "./notebooks/topic_results.json"
TOPIC_WITH_YEAR_RESULTS_PATH = "./notebooks/topic_results_with_year.json"
CONGRESS_MEMBERS_PATH = "./misc/HSall_members.csv"
BIOGUIDE_URL = "https://bioguide.congress.gov/search/bio/{id}.json"

CENSUS_BASE = "https://data.census.gov/api"

CHRONICLING_AMERICA_SEARCH_URL = "https://www.loc.gov/newspapers/"
CHRONICLING_AMERICA_MIN_YEAR = 1770
CHRONICLING_AMERICA_MAX_YEAR = 1963

SCHOLAR_QUERIES = [
    "fallacy political debate",
    "fallacious argument political discourse",
    "logical fallacy detection argumentation",
    "ad hominem appeal to emotion political speech",
    "propaganda technique detection political text",
    "argument mining fallacy classification debate",
]
OPENALEX_URL = "https://api.openalex.org/works"
CONTACT_EMAIL = os.environ.get("CONTACT_EMAIL", "research@example.com")
# OpenAlex robustness: longer timeout + retries so a single slow response can't drop a
# whole query (which previously let one query fill the entire quota with low-relevance hits).
OPENALEX_TIMEOUT = 60
OPENALEX_RETRIES = 3
# Cap how many papers any one query may contribute, so the top relevant results of EVERY
# query are kept instead of diving hundreds deep into a single query's ranking.
PAPERS_PER_QUERY = 40
# A paper is kept only if its title+abstract mentions one of these (precision gate).
PAPER_RELEVANCE_KEYWORDS = (
    "fallac", "fallacious", "argument", "argumentation", "rhetoric", "debate",
    "propaganda", "persuas", "ad hominem", "appeal to", "sophis", "demagog",
    "political discourse", "political speech", "reasoning",
)

# HF datasets to ingest as labeled exemplars: (hf_id, config, split, text_col, label_col).
# MAFALDA is NOT a HF dataset — it's handled separately from its GitHub JSONL (see below).
EXEMPLAR_DATASETS = [
    ("tasksource/logical-fallacy", None, "train", "source_article", "logical_fallacies"),
]
MAFALDA_JSONL_URL = (
    "https://raw.githubusercontent.com/ChadiHelwe/MAFALDA/main/"
    "datasets/gold_standard_dataset.jsonl"
)
# Map the many fine-grained fallacy names used across corpora (LOGIC, MAFALDA, propaganda)
# onto the six project categories. Keys are lowercased with hyphens normalised to spaces.
# Names that have no equivalent among the six categories deliberately fall through to "Other"
# and (with drop_other=True) are skipped, so only on-taxonomy exemplars are kept.
FALLACY_LABEL_MAP = {
    # --- Ad Hominem ---
    "ad hominem": "Ad Hominem",
    "abusive ad hominem": "Ad Hominem",
    "abusive": "Ad Hominem",
    "tu quoque": "Ad Hominem",
    "guilt by association": "Ad Hominem",
    "name calling": "Ad Hominem",
    "name_calling": "Ad Hominem",
    "poisoning the well": "Ad Hominem",
    # --- Appeal to Emotion ---
    "appeal to emotion": "Appeal to Emotion",
    "appeal_to_emotion": "Appeal to Emotion",
    "appeal to (positive) emotion": "Appeal to Emotion",
    "appeal to positive emotion": "Appeal to Emotion",
    "appeal to anger": "Appeal to Emotion",
    "appeal to fear": "Appeal to Emotion",
    "appeal to pity": "Appeal to Emotion",
    "appeal to ridicule": "Appeal to Emotion",
    "appeal to worse problems": "Appeal to Emotion",
    "loaded language": "Appeal to Emotion",
    "flag waving": "Appeal to Emotion",
    # --- Appeal to Authority (incl. appeal to popularity / majority opinion per project def) ---
    "appeal to authority": "Appeal to Authority",
    "appeal_to_authority": "Appeal to Authority",
    "appeal to false authority": "Appeal to Authority",
    "appeal to (false) authority": "Appeal to Authority",
    "fallacy of credibility": "Appeal to Authority",
    "ad populum": "Appeal to Authority",
    "appeal to popularity": "Appeal to Authority",
    "appeal to the people": "Appeal to Authority",
    "bandwagon": "Appeal to Authority",
    # --- Slippery Slope ---
    "slippery slope": "Slippery Slope",
    "slippery_slope": "Slippery Slope",
    # --- False Cause ---
    "false cause": "False Cause",
    "false_cause": "False Cause",
    "false causality": "False Cause",
    "causal oversimplification": "False Cause",
    "post hoc": "False Cause",
    "post hoc ergo propter hoc": "False Cause",
    # --- Slogan ---
    "slogan": "Slogan",
    "slogans": "Slogan",
}

# --- Relevance filter for "connected" Wikipedia pages (keep fallacy/argumentation only) ---
# Authoritative Wikipedia sub-categories that contain the actual fallacy articles. Their
# members form a high-precision allowlist (e.g. "Ad hominem", "Slippery slope").
FALLACY_WIKI_CATEGORIES = [
    "Relevance fallacies",
    "Causal fallacies",
    "Verbal fallacies",
    "Syllogistic fallacies",
    "Propositional fallacies",
    "Informal fallacies",
    "Quantificational fallacies",
]
# Keyword fallback: catches general rhetoric/argument pages and named fallacies not captured
# by the category membership above (matched as lowercased substrings of the title).
FALLACY_WIKI_KEYWORDS = (
    "fallac", "argument", "rhetoric", "reasoning", "syllogism", "premise", "inference",
    "sophis", "appeal to", "ad hominem", "ad nauseam", "nauseam", "consequent", "disjunct",
    "amphibol", "equivocat", "non sequitur", "begging the question", "red herring",
    "straw man", "slippery slope", "circular", "false cause", "false dilemma", "anecdot",
)


def _fetch_and_save_result(result: dict, path: Path, api_params: dict) -> str:
    pkg_id = result["packageId"]
    gran_id = result.get("granuleId") or pkg_id
    id = gran_id.replace(" ", "_")

    # Extract year from dateIssued or dateIngested
    doc_year = None
    for date_field in ["dateIssued", "dateIngested"]:
        if result.get(date_field):
            doc_year = int(result[date_field][:4])
            break

    # Fetch content with fallback URL
    content = None
    urls = [
        f"https://api.govinfo.gov/packages/{pkg_id}/granules/{gran_id}/htm",
        f"https://api.govinfo.gov/packages/{pkg_id}/granules/{pkg_id}.htm",
    ]
    correct_url = urls[0]
    for url in urls:
        try:
            content = get_request(url=url, params=api_params)
            correct_url = url
            break
        except Exception as e:
            # log.debug(f"Failed to fetch from {url}: {e}")
            continue

    if content is None:
        # log.error(f"Failed to fetch content for package {pkg_id}")
        raise Exception(f"Could not fetch content for {pkg_id}")

    filename = create_and_save_record(
        path=path,
        id=id,
        text=content.get_text(separator=" ", strip=True),
        source="U.S. Government Information Official Website",
        url=correct_url,
        xml=content.prettify(),
        year=doc_year,
    )
    # log.info(f"Saved {filename}")
    return filename


def bills_and_laws(path: Path):
    path.mkdir(parents=True, exist_ok=True)

    # Cache once — avoids repeated env lookups on every request
    api_params = {"api_key": os.environ["GOVINFO_API_KEY"]}

    # =================== get bills and laws =================== #
    for topic, year in (
            (t, int(y))
            for t, y in json.loads(Path(TOPIC_WITH_YEAR_RESULTS_PATH).read_text(encoding="utf-8"))["TOPICS"]
    ):
        topic = topic.replace("_", " ")
        for y in list(range(year - 4, year + 1)):
            try:
                res = post_request(
                    url=GOVINFO_URL,
                    params=api_params,
                    json_obj={
                        "query": topic,
                        "pageSize": 15,
                        "offsetMark": "*",
                        "sorts": [
                            {
                                "field": "score",
                                "sortOrder": "DESC"
                            }
                        ],
                        "facets": {"publishdatehier": y},
                        "resultLevel": "default"
                    },
                    return_json=True,
                )
                log.info("Found %d results for topic %s", len(res["results"]), topic)
            except Exception as e:
                log.error(e)
                continue

            with ThreadPoolExecutor(max_workers=10) as pool:
                futures = {
                    pool.submit(_fetch_and_save_result, result, path, api_params): result
                    for result in res["results"]
                }
        #     for fut in as_completed(futures):
        #         fut.result()  # re-raises any exception from the worker

    # =================== get government manual =================== #
    id = "GOVMAN-2025-01-13"
    source = "U.S. Government Information Official US Government Manual"
    r = get_request(url=GOV_MANUAL_URL)
    filename = create_and_save_record(path=path,
                                      id=id,
                                      text=r.text,
                                      source=source,
                                      url=GOV_MANUAL_URL,
                                      xml=r.prettify())
    log.info(f"Saved {filename}")

    del filename, id, r, source

    # =================== get universal declaration =================== #
    id = "UDHR"
    text = "\n".join(
        pytesseract.image_to_string(page)
        for page in convert_from_path(UDHR_PATH, dpi=800)
    )
    filename = create_and_save_record(
        path=path,
        id=id,
        text=text,
        source="Official Universal Declaration of Human Rights",
        url="https://www.un.org/sites/un2.un.org/files/2021/03/udhr.pdf",
        year=1948
    )

    log.info(f"Saved {filename}")

    del id, filename, text

    # =================== get bills and laws =================== #
    # for url in [BILLS_URL, PLAWS_URL]:
    #     root = get_request(url=url)
    #     assert root is not None
    #
    #     for year in root.find_all("loc"):
    #         y_resp = get_request(url=year.text)
    #         if y_resp is None: continue
    #
    #         for bill in y_resp.find_all("loc"):
    #             b_id = bill.text.split("/")[-1]
    #             if b_id + ".json" in os.listdir(path):
    #                 continue
    #             custom_url = f"https://www.govinfo.gov/content/pkg/{b_id}/html/{b_id}.htm"
    #             response = get_request(url=custom_url)
    #             if response is None: continue
    #
    #             filename = create_and_save_record(path=path,
    #                                               id=b_id,
    #                                               text=response.text,
    #                                               source="U.S. Government Information Official WebSite",
    #                                               url=custom_url,
    #                                               xml=response.prettify(),
    #                                               year=int(year.text.split("_")[1]))
    #             log.info(f"Saved {filename}")
    #
    # del filename, response, custom_url


def _reconstruct_abstract(inverted_index: dict | None) -> str:
    """Rebuild plain abstract text from OpenAlex's abstract_inverted_index."""
    if not inverted_index:
        return ""
    positions = [(pos, word) for word, idxs in inverted_index.items() for pos in idxs]
    positions.sort()
    return " ".join(word for _, word in positions)


def _download_and_ocr_pdf(url: str) -> str:
    """Best-effort download + OCR of a PDF. Returns '' on any failure."""
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        resp = requests.get(
            url, timeout=120, stream=True,
            headers={"User-Agent": "Mozilla/5.0 (compatible; FallacyRAG/1.0)"},
        )
        resp.raise_for_status()
        tmp_path.write_bytes(resp.content)
        pages = convert_from_path(str(tmp_path), dpi=300)
        return "\n".join(pytesseract.image_to_string(page) for page in pages)
    except Exception as e:
        log.warning("Full-text fetch failed for %s: %s", url, e)
        return ""
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


def _openalex_get(params: dict) -> dict | None:
    """GET OpenAlex with a longer timeout and retries/backoff. Returns None on failure."""
    for attempt in range(1, OPENALEX_RETRIES + 1):
        try:
            r = requests.get(OPENALEX_URL, params=params, timeout=OPENALEX_TIMEOUT)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            wait = 2.0 * attempt
            log.warning("OpenAlex request failed (attempt %d/%d): %s — retrying in %.0fs",
                        attempt, OPENALEX_RETRIES, e, wait)
            time.sleep(wait)
    return None


def _papers_from_openalex(queries: list[str], n_papers: int, since_year: int,
                          per_query_cap: int = PAPERS_PER_QUERY):
    """Yield normalized paper dicts from the OpenAlex API (no API key required).

    Each query contributes at most `per_query_cap` papers, and within a query results are
    returned by OpenAlex in descending relevance order — so we keep the top, most relevant
    hits of every query instead of letting one query dive hundreds deep into noise.
    """
    seen: set[str] = set()
    collected = 0
    per_page = min(per_query_cap, 200)  # no need to pull more than the per-query cap
    for query in queries:
        cursor = "*"
        from_query = 0
        while collected < n_papers and from_query < per_query_cap and cursor:
            params = {
                "search": query,
                "filter": f"from_publication_date:{since_year}-01-01,type:article",
                "per-page": per_page,
                "cursor": cursor,
                "mailto": CONTACT_EMAIL,
                # NOTE: `search` already returns results in descending relevance order;
                # an explicit `sort=relevance_score:desc` is incompatible with cursor paging.
                "select": "id,title,publication_year,abstract_inverted_index,"
                          "primary_location,open_access,relevance_score",
            }
            data = _openalex_get(params)
            if data is None:
                log.error("OpenAlex query '%s' dropped after retries.", query)
                break
            results = data.get("results", [])
            if not results:
                break
            for r in results:
                title = (r.get("title") or "").strip()
                if not title:
                    continue
                slug = safe_filename(title).lower()
                if slug in seen:
                    continue
                seen.add(slug)
                ploc = r.get("primary_location") or {}
                oa = r.get("open_access") or {}
                source = ploc.get("source") or {}
                yield {
                    "title": title,
                    "abstract": _reconstruct_abstract(r.get("abstract_inverted_index")),
                    "year": r.get("publication_year"),
                    "url": ploc.get("landing_page_url") or r.get("id"),
                    "pdf_url": ploc.get("pdf_url") or oa.get("oa_url"),
                    "venue": source.get("display_name") or "OpenAlex",
                }
                collected += 1
                from_query += 1
                if collected >= n_papers or from_query >= per_query_cap:
                    break
            cursor = (data.get("meta") or {}).get("next_cursor")


def _papers_from_google_scholar(queries: list[str], n_papers: int, since_year: int):
    """Yield normalized paper dicts by scraping Google Scholar via `scholarly`.

    Google Scholar has no official API and aggressively rate-limits / blocks scraping,
    so this backend is best-effort. Install with `pip install scholarly`; for large,
    reliable pulls prefer backend='openalex'.
    """
    try:
        from scholarly import scholarly
    except ImportError:
        log.error("`scholarly` is not installed (pip install scholarly). "
                  "Use backend='openalex' instead.")
        return
    seen: set[str] = set()
    collected = 0
    for query in queries:
        try:
            search = scholarly.search_pubs(query)
        except Exception as e:
            log.error("Google Scholar query '%s' failed: %s", query, e)
            continue
        for pub in search:
            if collected >= n_papers:
                return
            bib = pub.get("bib", {}) if isinstance(pub, dict) else {}
            title = (bib.get("title") or "").strip()
            if not title:
                continue
            try:
                year = int(bib.get("pub_year"))
            except (TypeError, ValueError):
                year = None
            if year is not None and year < since_year:
                continue
            slug = safe_filename(title).lower()
            if slug in seen:
                continue
            seen.add(slug)
            yield {
                "title": title,
                "abstract": bib.get("abstract", ""),
                "year": year,
                "url": pub.get("pub_url") or pub.get("eprint_url") or "",
                "pdf_url": pub.get("eprint_url"),
                "venue": bib.get("venue", "Google Scholar"),
            }
            collected += 1
            time.sleep(1.0)  # be polite — Scholar blocks rapid requests


def fallacy_papers(path: Path, n_papers: int = 200, since_year: int = 2015,
                   backend: str = "openalex", fetch_fulltext: bool = False,
                   queries: list[str] | None = None) -> int:
    """Search an academic index for recent papers on fallacies in political debates
    and add each as a knowledge-base record.
    """
    path.mkdir(parents=True, exist_ok=True)
    queries = queries or SCHOLAR_QUERIES
    if backend == "google_scholar":
        stream = _papers_from_google_scholar(queries, n_papers, since_year)
    else:
        stream = _papers_from_openalex(queries, n_papers, since_year)

    saved = 0
    skipped_irrelevant = 0
    for paper in tqdm(stream, total=n_papers, desc=f"Fallacy papers ({backend})"):
        # Relevance gate: keep only papers whose title+abstract actually concern
        # fallacies / argumentation / political debate. This drops tangential matches
        # (e.g. coastal degradation, space-weather) that merely share a query word.
        haystack = f"{paper['title']} {paper.get('abstract', '')}".lower()
        if not any(kw in haystack for kw in PAPER_RELEVANCE_KEYWORDS):
            skipped_irrelevant += 1
            continue

        text = ""
        if fetch_fulltext and paper.get("pdf_url"):
            text = _download_and_ocr_pdf(paper["pdf_url"])
        if not text.strip():
            text = f"{paper['title']}. {paper.get('abstract', '')}".strip()
        # Skip records with no usable content (e.g. no abstract available).
        if len(text.split()) < 5:
            continue
        try:
            filename = create_and_save_record(
                path=path,
                id=paper["title"],
                text=text,
                source=f"{paper.get('venue', 'Academic paper')} (via {backend})",
                url=paper.get("url", ""),
                year=paper.get("year") or 1900,
            )
            saved += 1
            log.info("Saved paper %s", filename)
        except Exception as e:
            log.error("Failed to save paper '%s': %s", paper.get("title"), e)
    log.info("Saved %d fallacy papers to %s (%d skipped as off-topic)",
             saved, path, skipped_irrelevant)
    return saved


def _normalize_fallacy_label(raw) -> str:
    """Map a raw corpus label (possibly a list / span-tagged value) to a project category."""
    if raw is None:
        return "Other"
    if isinstance(raw, (list, tuple)):
        # Multi-label: pick the first that maps to a known category.
        for item in raw:
            mapped = _normalize_fallacy_label(item)
            if mapped != "Other":
                return mapped
        return "Other"
    key = str(raw).strip().lower().replace("-", " ")
    return FALLACY_LABEL_MAP.get(key, "Other")


def _save_exemplar_record(path: Path, record_id: str, example: str, category: str,
                          source: str, url: str) -> bool:
    """Write a single labeled exemplar record. Returns True on success."""
    # Embed the label in the text so the encoder sees the structural cue, and store it as a
    # field for fallacy-keyed retrieval.
    record = {
        "id": record_id,
        "text": f"Fallacy type: {category}. Example: {example}",
        "source": source,
        "url": url,
        "year": 1900,
        "fallacy_label": category,
    }
    try:
        (path / f"{record_id}.json").write_text(
            json.dumps(record, indent=4, ensure_ascii=False), encoding="utf-8")
        return True
    except Exception as e:
        log.error("Failed to save exemplar %s: %s", record_id, e)
        return False


def _load_mafalda_exemplars(path: Path, drop_other: bool) -> int:
    """Ingest MAFALDA exemplars from its GitHub JSONL (it is not a HuggingFace dataset).

    Each row has a `text` and `labels` = list of [start, end, fallacy_name] spans; we keep
    the exact fallacious span as the example, labeled with its project category.
    """
    try:
        raw = requests.get(MAFALDA_JSONL_URL, timeout=OPENALEX_TIMEOUT).text
    except Exception as e:
        log.error("Could not fetch MAFALDA JSONL: %s — skipping.", e)
        return 0

    saved = 0
    for i, line in enumerate(raw.strip().splitlines()):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        text = row.get("text", "")
        for j, span in enumerate(row.get("labels", [])):
            if not (isinstance(span, (list, tuple)) and len(span) == 3):
                continue
            start, end, name = span
            category = _normalize_fallacy_label(name)
            if drop_other and category == "Other":
                continue
            example = str(text[start:end]).strip()
            if not example:
                continue
            if _save_exemplar_record(
                path, f"MAFALDA_{i}_{j}", example, category,
                "Fallacy exemplar corpus: MAFALDA", "https://github.com/ChadiHelwe/MAFALDA",
            ):
                saved += 1
    log.info("Ingested %d on-taxonomy exemplars from MAFALDA", saved)
    return saved


def fallacy_exemplars(path: Path, datasets_spec: list | None = None,
                      max_per_dataset: int | None = None, drop_other: bool = True) -> int:
    """Ingest labeled fallacy examples (LOGIC + MAFALDA) as KB records.

    drop_other : if True (default), skip examples whose fallacy type has no equivalent among
                 the six project categories — keeping only on-taxonomy exemplars so that
                 fallacy-keyed retrieval returns useful matches.
    """
    path.mkdir(parents=True, exist_ok=True)
    datasets_spec = datasets_spec or EXEMPLAR_DATASETS
    saved = 0
    dropped = 0

    # --- HuggingFace datasets (e.g. tasksource/logical-fallacy) ---
    for hf_id, config, split, text_col, label_col in datasets_spec:
        try:
            ds = load_dataset(hf_id, config, split=split)
        except Exception as e:
            log.error("Could not load exemplar dataset %s (%s): %s — skipping.",
                      hf_id, config, e)
            continue
        n = len(ds) if max_per_dataset is None else min(len(ds), max_per_dataset)
        kept = 0
        for i in range(n):
            row = ds[i]
            example = str(row.get(text_col, "")).strip()
            if not example:
                continue
            category = _normalize_fallacy_label(row.get(label_col))
            if drop_other and category == "Other":
                dropped += 1
                continue
            if _save_exemplar_record(
                path, f"{safe_filename(hf_id)}_{i}", example, category,
                f"Fallacy exemplar corpus: {hf_id}",
                f"https://huggingface.co/datasets/{hf_id}",
            ):
                saved += 1
                kept += 1
        log.info("Ingested %d on-taxonomy exemplars from %s", kept, hf_id)

    # --- MAFALDA (GitHub JSONL) ---
    saved += _load_mafalda_exemplars(path, drop_other)

    log.info("Saved %d fallacy exemplars to %s (%d off-taxonomy dropped)",
             saved, path, dropped)
    return saved


def definitions(path: Path):
    path.mkdir(parents=True, exist_ok=True)

    # =================== recent papers on fallacies in political debates =================== #
    # Search an academic index for ~200 recent papers and add each as a KB record. Kept in a
    # dedicated subfolder so the searched papers don't dilute the curated definition pages.
    fallacy_papers(path / "papers", n_papers=200, since_year=2015, backend="openalex",
                   fetch_fulltext=False)

    # =================== wikipedia page for fallacies =================== #
    id = "fallacy"
    text = fetch_wikipedia_sections(id)
    filename = create_and_save_record(path=path,
                                      id=id,
                                      text=text,
                                      source="Wikipedia webpage for {}".format(id),
                                      url=f"https://en.wikipedia.org/wiki/{id}")
    log.info("Saved %s", filename)

    # =================== connected wikipedia pages =================== #
    # Expand coverage by also downloading the article-namespace pages linked from the main
    # "fallacy" page (the individual fallacy types and related rhetorical concepts), but
    # filtered to fallacy/argumentation-relevant pages only — otherwise the "fallacy" page's
    # links pull in noise (e.g. "Adam Smith", "Advertisement") that would dilute retrieval.
    # A linked page is kept if it is EITHER a member of an authoritative fallacy category
    # OR its title matches a fallacy/rhetoric keyword.
    fallacy_allowlist = set()
    for category in FALLACY_WIKI_CATEGORIES:
        try:
            fallacy_allowlist.update(fetch_wikipedia_category_members(category))
        except Exception as e:
            log.error("Could not fetch category '%s': %s", category, e)
    log.info("Built fallacy allowlist of %d articles from %d categories",
             len(fallacy_allowlist), len(FALLACY_WIKI_CATEGORIES))

    def _is_fallacy_relevant(title: str) -> bool:
        if title in fallacy_allowlist:
            return True
        lowered = title.lower()
        return any(keyword in lowered for keyword in FALLACY_WIKI_KEYWORDS)

    seen_pages = {safe_filename(id)}
    for linked_title in fetch_wikipedia_links(id, limit=200):
        if not _is_fallacy_relevant(linked_title):
            continue
        slug = safe_filename(linked_title)
        if slug in seen_pages:
            continue
        seen_pages.add(slug)
        try:
            linked_text = fetch_wikipedia_sections(linked_title)
            if not linked_text:
                continue
            filename = create_and_save_record(
                path=path,
                id=slug,
                text=linked_text,
                source=f"Wikipedia webpage for {linked_title}",
                url=f"https://en.wikipedia.org/wiki/{linked_title.replace(' ', '_')}",
            )
            log.info("Saved %s", filename)
        except Exception as e:
            log.error("Failed to fetch connected page '%s': %s", linked_title, e)


def people(path: Path):
    path.mkdir(parents=True, exist_ok=True)

    # =================== official congress bibliography =================== #
    congress_members_ids = pd.read_csv(CONGRESS_MEMBERS_PATH)["bioguide_id"].dropna().unique().tolist()
    people = []

    for bioguide_id in tqdm(congress_members_ids):
        try:
            raw = simulate_browser(BIOGUIDE_URL.format(id=bioguide_id))
            if not raw: continue
            data = raw.get("data")

            name = f"{data.get('givenName', '').strip()} {data.get('familyName', '').strip()}"
            text = f" {name} is {data.get('profileText', '')}".strip()
            filename = create_and_save_record(path=path,
                                              id=name,
                                              text=text,
                                              source="Official Biographical Directory of the United States Congress",
                                              url=BIOGUIDE_URL.format(id=bioguide_id)
                                              )
            log.info(f"Saved {filename}")
            people.append(name)
            del filename, name, text
        except:
            continue

    # =================== people discussed in the debates =================== #
    people.extend(json.loads(Path(TOPIC_RESULTS_PATH).read_text(encoding="utf-8"))["PEOPLE"])
    people = set(people)
    for name in tqdm(set(people)):
        id = safe_filename(name)
        paragraph = fetch_wikipedia_sections(id)
        if not paragraph: continue
        filename = create_and_save_record(path=path,
                                          id=id,
                                          text=paragraph,
                                          source="Wikipedia's Web Page for {}".format(name),
                                          url="https://en.wikipedia.org/wiki/" + id)
        log.info(f"Saved {filename}")


def geography(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    # ================== get states from census ==================
    r = get_request(CENSUS_BASE + "/search", params={"g": "010XX00US", "services": "entities"}, return_json=True)
    profiles = r.get("response", {}).get("profiles", [])
    log.info("Processing %d states/territories", len(profiles))

    states = []
    for state in tqdm(profiles):
        try:
            geo_id = state["facets"]["geography"][0]
            meta = get_request(
                f"{CENSUS_BASE}/profile/metadata",
                params={"g": geo_id, "includeHighlights": "false"},
                return_json=True
            )
            if not meta: continue

            name = state["title"]
            states.append(name)
            file_id = safe_filename(name)

            paragraph = (
                    state.get("teaser", "")
                    + "\n\n"
                    + meta.get("header", {}).get("description", "")
                    + "\n\n"
            )

            # -------- collect topics --------
            topics = set()
            for section in meta.get("sections", []):
                topics.add(section.get("title"))

                for sub in section.get("subsections", []):
                    if sub.get("type") == "topic":
                        topics.add(sub.get("topic"))

            topics.discard("Explore Your State")

            # -------- fetch topic data --------
            for topic in topics:
                response = get_request(
                    f"{CENSUS_BASE}/profile/content/topic",
                    params={"g": geo_id, "infoSection": topic},
                    return_json=True
                )

                if not response:
                    continue

                viz = response.get("vizData", {})
                if viz and viz.get("stats"):
                    text = (
                        f"The {viz['title']} with source {viz['source']} "
                        f"shows that {viz['stats']}"
                    )
                    paragraph += f"{topic}: {text}\n\n"

            # -------- save record --------
            filename = create_and_save_record(
                path=path,
                id=file_id,
                text=paragraph,
                source="UNITED STATES CENSUS BUREAU",
                url=f"https://data.census.gov/profile/{state['title']}?g={geo_id}",
            )

            log.info("Saved %s", filename)

        except Exception as e:
            log.error("Error processing %s: %s", state.get("title"), e)

    # ================== get states from wikipedia ==================
    states.extend(json.loads(Path(TOPIC_RESULTS_PATH).read_text())["PLACES"])
    states = set(states)
    for state in tqdm(states):
        id = safe_filename(state)
        try:
            paragraph = fetch_wikipedia_sections(id)
            filename = create_and_save_record(path=path,
                                              id=id,
                                              text=paragraph,
                                              source="Wikipedia's Web Page for {}".format(state),
                                              url=f"https://en.wikipedia.org/wiki/{id}"
                                              )
        except Exception as e:
            log.error("Error processing %s: %s", id, e)
            continue
        log.info("Saved %s", filename)


def newspapers(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    searchURL = 'https://www.loc.gov/collections/chronicling-america/?dl=page&ops=PHRASE&qs={query}&searchType=advanced&start_date=1950-01-01&fo=json'
    fileExtension = 'pdf'
    saveTo = path / "tmp_output"
    saveTo.mkdir(parents=True, exist_ok=True)

    # ---- collect all unique item IDs across every topic ----
    topics_with_year = Path(TOPIC_WITH_YEAR_RESULTS_PATH).read_text(encoding="utf-8")
    topics = [topic for topic, _ in json.loads(topics_with_year)["TOPICS"]]
    all_item_ids: set[str] = set()
    for topic in tqdm(topics, desc="Collecting item IDs"):
        url = searchURL.format(query=topic.replace(" ", "+"))
        try:
            items = get_item_ids(url, items=[])
            for item_id in items:
                if not item_id.endswith('&fo=json'):
                    item_id += '&fo=json'
                all_item_ids.add(item_id)
        except Exception as e:
            log.error("Failed to get item IDs: %s", e)
            continue

    log.info("Found %d unique item IDs across %d topics", len(all_item_ids), len(topics))

    # ---- fetch page URLs from all items in parallel ----
    def fetch_page_urls(item_url: str, session: requests.Session) -> list[str]:
        try:
            r = session.get(item_url, timeout=120)
            if r.status_code != 200:
                return []
            return [
                page['url']
                for page in r.json().get('page', [])
                if 'url' in page and page['url'].endswith(fileExtension)
            ]
        except Exception as e:
            log.warning("Failed to fetch item %s: %s", item_url, e)
            return []

    page_urls: set[str] = set()
    with requests.Session() as session:
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(fetch_page_urls, item, session): item for item in all_item_ids}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Fetching item metadata"):
                page_urls.update(future.result())

    log.info("Found %d unique PDF page URLs", len(page_urls))

    # ---- download PDFs in parallel, skipping already-downloaded files ----
    def download_pdf(page_url: str, session: requests.Session) -> None:
        dest = saveTo / page_url.split('/')[-1]
        if dest.exists():
            log.info("Skipping already-downloaded %s", dest.name)
            return
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Referer': 'https://tile.loc.gov/',
                'Accept': 'application/pdf',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1'
            }
            r = session.get(page_url, timeout=120, headers=headers)
            r.raise_for_status()
            dest.write_bytes(r.content)
            log.info("Downloaded %s", dest.name)
        except Exception as e:
            log.warning("Failed to download %s: %s", page_url, e)

    with requests.Session() as session:
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(download_pdf, url, session): url for url in page_urls}
            for _ in tqdm(as_completed(futures), total=len(futures), desc="Downloading PDFs"):
                pass

    # ---- OCR PDFs from tmp_output and save records to the newspaper folder ----
    pdf_files = list(saveTo.glob("*.pdf"))
    log.info("Found %d PDF files to OCR", len(pdf_files))

    reader = easyocr.Reader(['en'], gpu=False)

    for pdf_path in tqdm(pdf_files, desc="OCR-ing PDFs"):
        try:
            file_id = safe_filename(pdf_path.stem)

            # Try to extract a 4-digit year from the filename that falls within
            # the Chronicling America date range.
            year = CHRONICLING_AMERICA_MIN_YEAR
            for match in re.finditer(r'(\d{4})', pdf_path.stem):
                candidate = int(match.group(1))
                if CHRONICLING_AMERICA_MIN_YEAR <= candidate <= CHRONICLING_AMERICA_MAX_YEAR:
                    year = candidate
                    break

            # Convert each PDF page to an image then OCR with EasyOCR.
            pages = convert_from_path(str(pdf_path), dpi=300)
            page_texts = []
            for page_img in pages:
                results = reader.readtext(np.array(page_img), detail=0, paragraph=True)
                page_texts.append(" ".join(results))

            text = "\n\n".join(page_texts)
            if not text.strip():
                log.warning("No text extracted from %s, skipping", pdf_path.name)
                continue

            filename = create_and_save_record(
                path=path,
                id=file_id,
                text=text,
                source="Chronicling America - Library of Congress Historic Newspapers",
                url=f"https://chroniclingamerica.loc.gov/{pdf_path.stem}",
                year=year
            )
            # log.info("Saved %s", filename)
        except Exception as e:
            log.error("Error OCR-ing %s: %s", pdf_path.name, e)

    # ---- clean up the temporary PDF folder ----
    # import shutil
    # shutil.rmtree(saveTo, ignore_errors=True)
    # log.info("Removed temporary folder %s", saveTo)


def main():
    # bills_and_laws(Path(KNOWLEDGE_BASE_PATH + "/bills"))
    # geography(Path(KNOWLEDGE_BASE_PATH + "/geography"))
    # people(Path(KNOWLEDGE_BASE_PATH + "/people"))
    definitions(Path(KNOWLEDGE_BASE_PATH + "/definitions"))
    # newspapers(Path(KNOWLEDGE_BASE_PATH + "/newspapers"))
    fallacy_exemplars(Path(KNOWLEDGE_BASE_PATH + "/exemplars"))


if __name__ == "__main__":
    main()
