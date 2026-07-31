"""Build the FallacyRAG knowledge base: bills, people, geography, history, papers and exemplars."""

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
from src.utils.util import create_and_save_record, get_request, simulate_browser, \
    safe_filename, fetch_wikipedia_sections, fetch_wikipedia_section, \
    fetch_wikipedia_links, fetch_wikipedia_extracts, fetch_wikipedia_category_members, \
    post_request, get_item_ids

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

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

HISTORY_START_YEAR = 1900
HISTORY_WIKI_TITLE = "{year} in the United States"

HISTORY_ERA_PAGES = [
    ("History of the United States (1917–1945)", 1917, 1945),
    ("History of the United States (1945–1964)", 1945, 1964),
    ("History of the United States (1964–1980)", 1964, 1980),
    ("History of the United States (1980–1991)", 1980, 1991),
    ("History of the United States (1991–2016)", 1991, 2016),
    ("History of the United States (2016–present)", 2016, None),
]

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
OPENALEX_TIMEOUT = 60
OPENALEX_RETRIES = 3
PAPERS_PER_QUERY = 40
PAPER_RELEVANCE_KEYWORDS = (
    "fallac", "fallacious", "argument", "argumentation", "rhetoric", "debate",
    "propaganda", "persuas", "ad hominem", "appeal to", "sophis", "demagog",
    "political discourse", "political speech", "reasoning",
)

EXEMPLAR_DATASETS = [
    ("tasksource/logical-fallacy", None, "train", "source_article", "logical_fallacies"),
]
MAFALDA_JSONL_URL = (
    "https://raw.githubusercontent.com/ChadiHelwe/MAFALDA/main/"
    "datasets/gold_standard_dataset.jsonl"
)
FALLACY_LABEL_MAP = {
    "ad hominem": "Ad Hominem",
    "abusive ad hominem": "Ad Hominem",
    "abusive": "Ad Hominem",
    "tu quoque": "Ad Hominem",
    "guilt by association": "Ad Hominem",
    "name calling": "Ad Hominem",
    "name_calling": "Ad Hominem",
    "poisoning the well": "Ad Hominem",
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
    "appeal to authority": "Appeal to Authority",
    "appeal_to_authority": "Appeal to Authority",
    "appeal to false authority": "Appeal to Authority",
    "appeal to (false) authority": "Appeal to Authority",
    "fallacy of credibility": "Appeal to Authority",
    "ad populum": "Appeal to Authority",
    "appeal to popularity": "Appeal to Authority",
    "appeal to the people": "Appeal to Authority",
    "bandwagon": "Appeal to Authority",
    "slippery slope": "Slippery Slope",
    "slippery_slope": "Slippery Slope",
    "false cause": "False Cause",
    "false_cause": "False Cause",
    "false causality": "False Cause",
    "causal oversimplification": "False Cause",
    "post hoc": "False Cause",
    "post hoc ergo propter hoc": "False Cause",
    "slogan": "Slogan",
    "slogans": "Slogan",
}

FALLACY_WIKI_CATEGORIES = [
    "Relevance fallacies",
    "Causal fallacies",
    "Verbal fallacies",
    "Syllogistic fallacies",
    "Propositional fallacies",
    "Informal fallacies",
    "Quantificational fallacies",
]
FALLACY_WIKI_KEYWORDS = (
    "fallac", "argument", "rhetoric", "reasoning", "syllogism", "premise", "inference",
    "sophis", "appeal to", "ad hominem", "ad nauseam", "nauseam", "consequent", "disjunct",
    "amphibol", "equivocat", "non sequitur", "begging the question", "red herring",
    "straw man", "slippery slope", "circular", "false cause", "false dilemma", "anecdot",
)


def _fetch_and_save_result(result: dict, path: Path, api_params: dict) -> str:
    """Fetch one GovInfo search result's HTML content and save it as a KB record."""
    pkg_id = result["packageId"]
    gran_id = result.get("granuleId") or pkg_id
    id = gran_id.replace(" ", "_")

    doc_year = None
    for date_field in ["dateIssued", "dateIngested"]:
        if result.get(date_field):
            doc_year = int(result[date_field][:4])
            break

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
            continue

    if content is None:
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
    return filename


def bills_and_laws(path: Path):
    """Ingest GovInfo bills/laws by topic, the US Government Manual, and the UDHR."""
    path.mkdir(parents=True, exist_ok=True)

    api_params = {"api_key": os.environ["GOVINFO_API_KEY"]}

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
    """Yield normalized paper dicts from the OpenAlex API, capped per query for relevance."""
    seen: set[str] = set()
    collected = 0
    per_page = min(per_query_cap, 200)
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
    """Yield normalized paper dicts by scraping Google Scholar (best-effort, rate-limited)."""
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
            time.sleep(1.0)


def fallacy_papers(path: Path, n_papers: int = 200, since_year: int = 2015,
                   backend: str = "openalex", fetch_fulltext: bool = False,
                   queries: list[str] | None = None) -> int:
    """Search an academic index for recent fallacy/argumentation papers and save each as a record."""
    path.mkdir(parents=True, exist_ok=True)
    queries = queries or SCHOLAR_QUERIES
    if backend == "google_scholar":
        stream = _papers_from_google_scholar(queries, n_papers, since_year)
    else:
        stream = _papers_from_openalex(queries, n_papers, since_year)

    saved = 0
    skipped_irrelevant = 0
    for paper in tqdm(stream, total=n_papers, desc=f"Fallacy papers ({backend})"):
        haystack = f"{paper['title']} {paper.get('abstract', '')}".lower()
        if not any(kw in haystack for kw in PAPER_RELEVANCE_KEYWORDS):
            skipped_irrelevant += 1
            continue

        text = ""
        if fetch_fulltext and paper.get("pdf_url"):
            text = _download_and_ocr_pdf(paper["pdf_url"])
        if not text.strip():
            text = f"{paper['title']}. {paper.get('abstract', '')}".strip()
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
    """Ingest MAFALDA exemplars from its GitHub JSONL, keeping each labeled fallacious span."""
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
    """Ingest labeled fallacy examples (LOGIC + MAFALDA); drop_other keeps only on-taxonomy ones."""
    path.mkdir(parents=True, exist_ok=True)
    datasets_spec = datasets_spec or EXEMPLAR_DATASETS
    saved = 0
    dropped = 0

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

    saved += _load_mafalda_exemplars(path, drop_other)

    log.info("Saved %d fallacy exemplars to %s (%d off-taxonomy dropped)",
             saved, path, dropped)
    return saved


def definitions(path: Path):
    """Ingest fallacy papers, the Wikipedia 'fallacy' page, and its relevant linked pages."""
    path.mkdir(parents=True, exist_ok=True)

    fallacy_papers(path / "papers", n_papers=200, since_year=2015, backend="openalex",
                   fetch_fulltext=False)

    id = "fallacy"
    text = fetch_wikipedia_sections(id)
    filename = create_and_save_record(path=path,
                                      id=id,
                                      text=text,
                                      source="Wikipedia webpage for {}".format(id),
                                      url=f"https://en.wikipedia.org/wiki/{id}")
    log.info("Saved %s", filename)

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
    relevant_titles = []
    for linked_title in fetch_wikipedia_links(id, limit=200):
        if not _is_fallacy_relevant(linked_title):
            continue
        slug = safe_filename(linked_title)
        if slug in seen_pages:
            continue
        seen_pages.add(slug)
        relevant_titles.append(linked_title)

    linked_texts = _fetch_pages_parallel(relevant_titles)
    for linked_title in relevant_titles:
        linked_text = linked_texts.get(safe_filename(linked_title))
        if not linked_text:
            continue
        create_and_save_record(
            path=path,
            id=safe_filename(linked_title),
            text=linked_text,
            source=f"Wikipedia webpage for {linked_title}",
            url=f"https://en.wikipedia.org/wiki/{linked_title.replace(' ', '_')}",
        )


def _fetch_pages_parallel(titles, max_workers: int = 6) -> dict:
    """Fetch full plain-text for many Wikipedia pages concurrently; return {slug: text}."""
    extracts = fetch_wikipedia_extracts(titles, intro_only=False, max_workers=max_workers)
    return {safe_filename(title): text for title, text in extracts.items()}


def _expand_first_level_links(path: Path, seed_titles, source_template: str,
                              limit: int = 500, max_workers: int = 8,
                              seen: set | None = None) -> int:
    """Fetch and save the unique first-level Wikipedia pages linked from each seed page."""
    seen = seen if seen is not None else set()
    seeds = [t for t in dict.fromkeys(seed_titles) if t]
    for t in seeds:
        seen.add(safe_filename(t))

    def _links(title: str):
        try:
            return fetch_wikipedia_links(title, limit=limit)
        except Exception as e:
            log.error("Could not fetch links for '%s': %s", title, e)
            return []

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        link_lists = list(pool.map(_links, seeds))

    new_titles: list[str] = []
    for linked_titles in link_lists:
        for linked_title in linked_titles:
            slug = safe_filename(linked_title)
            if slug in seen:
                continue
            seen.add(slug)
            new_titles.append(linked_title)

    log.info("Expanding %d unique first-level linked pages", len(new_titles))
    extracts = fetch_wikipedia_extracts(new_titles, max_workers=max_workers)

    saved = 0
    for canon_title, text in extracts.items():
        create_and_save_record(
            path=path,
            id=safe_filename(canon_title),
            text=text,
            source=source_template.format(title=canon_title),
            url=f"https://en.wikipedia.org/wiki/{canon_title.replace(' ', '_')}",
        )
        saved += 1
    log.info("Saved %d first-level linked pages to %s", saved, path)
    return saved


def people(path: Path):
    """Ingest Congress member bios, debate-mentioned people's Wikipedia pages, and their links."""
    path.mkdir(parents=True, exist_ok=True)

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

    people.extend(json.loads(Path(TOPIC_RESULTS_PATH).read_text(encoding="utf-8"))["PEOPLE"])
    people = set(people)
    seen_pages: set = set()

    person_titles = {name: safe_filename(name) for name in people}
    person_texts = _fetch_pages_parallel(list(person_titles.values()))
    for name, page_id in person_titles.items():
        seen_pages.add(page_id)
        paragraph = person_texts.get(page_id)
        if not paragraph:
            continue
        create_and_save_record(path=path,
                               id=page_id,
                               text=paragraph,
                               source="Wikipedia's Web Page for {}".format(name),
                               url="https://en.wikipedia.org/wiki/" + page_id)

    _expand_first_level_links(
        path, people, "Wikipedia connected page for {title}", seen=seen_pages,
    )


def geography(path: Path):
    """Ingest US Census state profiles plus debate-mentioned places' Wikipedia pages and links."""
    path.mkdir(parents=True, exist_ok=True)
    r = get_request(CENSUS_BASE + "/search", params={"g": "010XX00US", "services": "entities"}, return_json=True)
    profiles = r.get("response", {}).get("profiles", [])
    log.info("Processing %d states/territories", len(profiles))

    def _process_state(state) -> str | None:
        """Fetch one state's census profile and topic data and save it as a record."""
        try:
            geo_id = state["facets"]["geography"][0]
            meta = get_request(
                f"{CENSUS_BASE}/profile/metadata",
                params={"g": geo_id, "includeHighlights": "false"},
                return_json=True
            )
            if not meta:
                return None

            name = state["title"]
            file_id = safe_filename(name)

            paragraph = (
                    state.get("teaser", "")
                    + "\n\n"
                    + meta.get("header", {}).get("description", "")
                    + "\n\n"
            )

            topics = set()
            for section in meta.get("sections", []):
                topics.add(section.get("title"))
                for sub in section.get("subsections", []):
                    if sub.get("type") == "topic":
                        topics.add(sub.get("topic"))
            topics.discard("Explore Your State")

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

            create_and_save_record(
                path=path,
                id=file_id,
                text=paragraph,
                source="UNITED STATES CENSUS BUREAU",
                url=f"https://data.census.gov/profile/{state['title']}?g={geo_id}",
            )
            return name
        except Exception as e:
            log.error("Error processing %s: %s", state.get("title"), e)
            return None

    states = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for name in tqdm(pool.map(_process_state, profiles), total=len(profiles),
                         desc="Census states"):
            if name:
                states.append(name)

    states.extend(json.loads(Path(TOPIC_RESULTS_PATH).read_text())["PLACES"])
    states = set(states)
    seen_pages: set = set()

    place_titles = {state: safe_filename(state) for state in states}
    place_texts = _fetch_pages_parallel(list(place_titles.values()))
    for state, page_id in place_titles.items():
        seen_pages.add(page_id)
        paragraph = place_texts.get(page_id)
        if not paragraph:
            continue
        create_and_save_record(path=path,
                               id=page_id,
                               text=paragraph,
                               source="Wikipedia's Web Page for {}".format(state),
                               url=f"https://en.wikipedia.org/wiki/{page_id}")

    _expand_first_level_links(
        path, states, "Wikipedia connected page for {title}", seen=seen_pages,
    )


def newspapers(path: Path):
    """Download Chronicling America newspaper PDFs by topic, OCR them, and save records."""
    path.mkdir(parents=True, exist_ok=True)
    searchURL = 'https://www.loc.gov/collections/chronicling-america/?dl=page&ops=PHRASE&qs={query}&searchType=advanced&start_date=1950-01-01&fo=json'
    fileExtension = 'pdf'
    saveTo = path / "tmp_output"
    saveTo.mkdir(parents=True, exist_ok=True)

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

    def fetch_page_urls(item_url: str, session: requests.Session) -> list[str]:
        """Return the PDF page URLs listed in one item's metadata."""
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

    def download_pdf(page_url: str, session: requests.Session) -> None:
        """Download one PDF to the temp folder, skipping if already present."""
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

    pdf_files = list(saveTo.glob("*.pdf"))
    log.info("Found %d PDF files to OCR", len(pdf_files))

    reader = easyocr.Reader(['en'], gpu=False)

    for pdf_path in tqdm(pdf_files, desc="OCR-ing PDFs"):
        try:
            file_id = safe_filename(pdf_path.stem)

            year = CHRONICLING_AMERICA_MIN_YEAR
            for match in re.finditer(r'(\d{4})', pdf_path.stem):
                candidate = int(match.group(1))
                if CHRONICLING_AMERICA_MIN_YEAR <= candidate <= CHRONICLING_AMERICA_MAX_YEAR:
                    year = candidate
                    break

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
        except Exception as e:
            log.error("Error OCR-ing %s: %s", pdf_path.name, e)


def _make_year_aware(year: int, text: str) -> str:
    """Prefix every paragraph with the year so each split chunk keeps its temporal context."""
    header = f"Year: {year}. History of the United States in {year}.\n\n"
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    body = "\n\n".join(f"[{year}] {p}" for p in paragraphs)
    return header + body


def _make_period_aware(start_year: int, end_year: int, text: str) -> str:
    """Prefix every paragraph with the period span so each split chunk stays era-anchored."""
    span = f"{start_year}–{end_year}"
    header = f"Period: {start_year} to {end_year}. History of the United States, {span}.\n\n"
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    body = "\n\n".join(f"[{span}] {p}" for p in paragraphs)
    return header + body


def _save_history_page(path: Path, title: str, start_year: int, end_year: int,
                       seen: set) -> bool:
    """Fetch one Wikipedia page and save it as a period-anchored history record (year=start_year)."""
    slug = safe_filename(title)
    if slug in seen:
        return False
    seen.add(slug)
    try:
        text = fetch_wikipedia_sections(title)
        if not text or not text.strip():
            log.warning("No content for '%s', skipping", title)
            return False
        filename = create_and_save_record(
            path=path,
            id=slug,
            text=_make_period_aware(start_year, end_year, text),
            source=f"Wikipedia webpage for {title}",
            url=f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
            year=start_year,
        )
        log.info("Saved %s", filename)
        return True
    except Exception as e:
        log.error("Failed to fetch page '%s': %s", title, e)
        return False


def history_eras(path: Path, era_pages: list | None = None,
                 explore_links: bool = True, links_per_page: int = 500,
                 seen: set | None = None):
    """Ingest the broad "History of the United States (start–end)" era pages and their links."""
    path.mkdir(parents=True, exist_ok=True)
    era_pages = era_pages or HISTORY_ERA_PAGES
    if seen is None:
        seen = set()
    saved = 0
    for title, start_year, end_year in tqdm(era_pages, desc="US history eras"):
        resolved_end = end_year if end_year is not None else time.localtime().tm_year

        if _save_history_page(path, title, start_year, resolved_end, seen):
            saved += 1

        if not explore_links:
            continue
        try:
            linked_titles = fetch_wikipedia_links(title, limit=links_per_page)
        except Exception as e:
            log.error("Could not fetch links for '%s': %s", title, e)
            continue
        for linked_title in tqdm(linked_titles, desc=f"Links of {start_year}", leave=False):
            if _save_history_page(path, linked_title, start_year, resolved_end, seen):
                saved += 1

    log.info("Saved %d US-history era + linked records to %s", saved, path)
    return saved


def american_history(path: Path, start_year: int = HISTORY_START_YEAR,
                     end_year: int | None = None,
                     explore_links: bool = True, links_per_year: int = 500,
                     max_workers: int = 8):
    """Ingest US history one record per year (each year's Wikipedia 'Events' section) plus links."""
    path.mkdir(parents=True, exist_ok=True)
    if end_year is None:
        end_year = time.localtime().tm_year

    years = list(range(start_year, end_year + 1))
    seen: set = set()
    saved = 0

    def _fetch_events(year: int):
        """Fetch one year's 'Events' section text."""
        title = HISTORY_WIKI_TITLE.format(year=year)
        try:
            return year, title, fetch_wikipedia_section(title, "Events")
        except Exception as e:
            log.error("Failed to fetch Events for %d: %s", year, e)
            return year, title, ""

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        events_results = list(tqdm(pool.map(_fetch_events, years),
                                   total=len(years), desc="US history events"))

    for year, title, events in events_results:
        seen.add(safe_filename(title))
        if events and events.strip():
            create_and_save_record(
                path=path,
                id=f"US_history_{year}",
                text=_make_year_aware(year, events),
                source=f"Wikipedia webpage for {title} (Events)",
                url=f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}#Events",
                year=year,
            )
            saved += 1
        else:
            log.warning("No Events section for '%s'", title)

    if explore_links:
        def _fetch_links(year: int):
            """Fetch the article-namespace links from one year's page."""
            title = HISTORY_WIKI_TITLE.format(year=year)
            try:
                return year, fetch_wikipedia_links(title, limit=links_per_year)
            except Exception as e:
                log.error("Could not fetch links for '%s': %s", title, e)
                return year, []

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            link_lists = list(tqdm(pool.map(_fetch_links, years),
                                   total=len(years), desc="US history link lists"))

        title_to_year: dict[str, int] = {}
        for year, linked_titles in link_lists:
            for linked_title in linked_titles:
                slug = safe_filename(linked_title)
                if slug in seen:
                    continue
                seen.add(slug)
                title_to_year[linked_title] = year

        log.info("Fetching %d unique linked pages in batches", len(title_to_year))
        extracts = fetch_wikipedia_extracts(list(title_to_year), max_workers=max_workers)

        requested = {safe_filename(t): t for t in title_to_year}
        for canon_title, text in extracts.items():
            year = title_to_year.get(canon_title)
            if year is None:
                year = title_to_year.get(requested.get(safe_filename(canon_title), ""), start_year)
            create_and_save_record(
                path=path,
                id=safe_filename(canon_title),
                text=_make_year_aware(year, text),
                source=f"Wikipedia webpage for {canon_title}",
                url=f"https://en.wikipedia.org/wiki/{canon_title.replace(' ', '_')}",
                year=year,
            )
            saved += 1

    log.info("Saved %d yearly US-history records to %s (%d-%d)",
             saved, path, start_year, end_year)

    saved += history_eras(path, seen=seen)
    return saved


def main():
    """Build every knowledge-base section into its subfolder."""
    bills_and_laws(Path(KNOWLEDGE_BASE_PATH + "/bills"))
    geography(Path(KNOWLEDGE_BASE_PATH + "/geography"))
    people(Path(KNOWLEDGE_BASE_PATH + "/people"))
    american_history(Path(KNOWLEDGE_BASE_PATH + "/history"))
    definitions(Path(KNOWLEDGE_BASE_PATH + "/definitions"))
    fallacy_exemplars(Path(KNOWLEDGE_BASE_PATH + "/exemplars"))
    newspapers(Path(KNOWLEDGE_BASE_PATH + "/newspapers"))


if __name__ == "__main__":
    main()
