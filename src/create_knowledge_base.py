import json
import logging
import os
import tempfile
from pathlib import Path

import pandas as pd
import pytesseract
import requests
import rootutils
from pdf2image import convert_from_path
from tqdm import tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)
from src.utils.retrieve_docs import create_and_save_record, get_request, simulate_browser, safe_filename, \
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
GOVINFO_URL = f"https://api.govinfo.gov/search"
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

def bills_and_laws(path: Path):
    path.mkdir(parents=True, exist_ok=True)

    # =================== get bills and laws =================== #
    for topic, _ in json.loads(Path(TOPIC_WITH_YEAR_RESULTS_PATH).read_text(encoding="utf-8"))["TOPICS"]:
        res = post_request(url=GOVINFO_URL,
                           params={"api_key": os.environ["GOVINFO_API_KEY"]},
                           json_obj={"query": topic, "pageSize": 15, "offsetMark": "*"},
                           return_json=True)
        for result in res["results"]:
            pkg_id = result["packageId"]
            gran_id = result.get("granuleId", pkg_id)
            year = int(result.get("dateIssued", "1900-01-01")[:4])

            download_url = f"https://api.govinfo.gov/packages/{pkg_id}/granules/{gran_id}/htm"
            content = get_request(
                url=download_url,
                params={"api_key": os.environ["GOVINFO_API_KEY"]},
                return_json=False
            )

            filename = create_and_save_record(
                path=path,
                id=gran_id,
                text=content.get_text(separator="\n", strip=True),
                source="U.S. Government Information Official Website",
                url=download_url,
                xml=content.prettify(),
                year=year
            )
            log.info(f"Saved {filename}")
    del filename

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

            response = requests.get(link, timeout=30, stream=True)
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
    saveTo = os.path.join(path, "tmp_output")
    for topic, _ in json.loads(Path(TOPIC_WITH_YEAR_RESULTS_PATH).read_text(encoding="utf-8"))["TOPICS"]:
        url = searchURL.format(query=topic.replace(" ", "+"))
        ids_list = get_item_ids(url, items=[])
        new_ids = []
        for id in ids_list:
            if not id.endswith('&fo=json'):
                id += '&fo=json'
            new_ids.append(id)

        for item in new_ids:
            call = requests.get(item)
            if call.status_code == 200:
                data = call.json()
                page = data['page']
                for page in page:
                    if 'url' in page:
                        page_url = page['url']
                        if page_url.endswith(fileExtension):
                            print(page_url)

        # Get the page URLs
        page_urls = []
        for item in new_ids:
            call = requests.get(item)
            if call.status_code == 200:
                data = call.json()
                page = data['page']
                for page in page:
                    if 'url' in page:
                        page_url = page['url']
                        if page_url.endswith(fileExtension):
                            page_urls.append(page_url)

        for page_url in page_urls:
            filename = page_url.split('/')[-1]
            response = requests.get(page_url)
            file_path = os.path.join(saveTo, filename)
            with open(file_path, 'wb') as f:
                f.write(response.content)


def main():
    # bills_and_laws(Path(KNOWLEDGE_BASE_PATH + "/bills"))
    # geography(Path(KNOWLEDGE_BASE_PATH + "/geography"))
    # people(Path(KNOWLEDGE_BASE_PATH + "/people"))
    # definitions(Path(KNOWLEDGE_BASE_PATH + "/definitions"))
    newspapers(Path(KNOWLEDGE_BASE_PATH + "/newspapers"))


if __name__ == "__main__":
    main()
