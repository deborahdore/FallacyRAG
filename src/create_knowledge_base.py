import json
import logging
import os
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import easyocr
import numpy as np
import pandas as pd
import pytesseract
import requests
import rootutils
from pdf2image import convert_from_path
from tqdm import tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)
from src.utils.util import create_and_save_record, get_request, simulate_browser, safe_filename, \
    fetch_wikipedia_sections, post_request, get_item_ids

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


# ============================================ #

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


def definitions(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    papers = {
        "goffredo2022fallacious": {
            "url": "https://www.ijcai.org/proceedings/2022/0575.pdf",
            "title": "Fallacious argument classification in political debates"
        },
        "goffredo2023argument": {
            "title": "Argument-based Detection and Classification of Fallacies in Political Debates",
            "url": "https://aclanthology.org/2023.emnlp-main.684.pdf"
        }
    }

    # =================== papers =================== #
    for key in papers.keys():
        link = papers[key]["url"]
        title = papers[key]["title"]
        try:
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp_path = Path(tmp.name)

            response = requests.get(link, timeout=120, stream=True)
            response.raise_for_status()
            tmp_path.write_bytes(response.content)

            pages = convert_from_path(str(tmp_path), dpi=800)
            text = "\n".join(pytesseract.image_to_string(page) for page in pages)
            filename = create_and_save_record(path=path,
                                              id=key,
                                              text=text,
                                              source=title,
                                              url=link)
            log.info("Saved %s", filename)
        finally:
            tmp_path.unlink(missing_ok=True)
    # =================== wikipedia page for fallacies =================== #
    id = "fallacy"
    text = fetch_wikipedia_sections(id)
    filename = create_and_save_record(path=path,
                                      id=id,
                                      text=text,
                                      source="Wikipedia webpage for {}".format(id),
                                      url=f"https://en.wikipedia.org/wiki/{id}")
    log.info("Saved %s", filename)


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
    bills_and_laws(Path(KNOWLEDGE_BASE_PATH + "/bills"))
    geography(Path(KNOWLEDGE_BASE_PATH + "/geography"))
    people(Path(KNOWLEDGE_BASE_PATH + "/people"))
    definitions(Path(KNOWLEDGE_BASE_PATH + "/definitions"))
    newspapers(Path(KNOWLEDGE_BASE_PATH + "/newspapers"))


if __name__ == "__main__":
    main()
