import json
import logging
import tempfile
from pathlib import Path

import pandas as pd
import pytesseract
import requests
import rootutils
from pdf2image import convert_from_path
from tqdm import tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)
from scr.utils import create_and_save_record, get_request, simulate_browser, safe_filename, \
    fetch_wikipedia_sections

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
UDHR_PATH = "./data/udhr.pdf"

TOPIC_RESULTS_PATH = "./notebooks/topic_results.json"
CONGRESS_MEMBERS_PATH = "./data/HSall_members.csv"
BIOGUIDE_URL = "https://bioguide.congress.gov/search/bio/{id}.json"

CENSUS_BASE = "https://data.census.gov/api"


# ============================================ #

def bills_and_laws(path: Path):
    path.mkdir(parents=True, exist_ok=True)

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
        url="https://www.un.org/sites/un2.un.org/files/2021/03/udhr.pdf"
    )

    log.info(f"Saved {filename}")

    del id, filename, text

    # =================== get bills and laws =================== #
    for url in [BILLS_URL, PLAWS_URL]:
        root = get_request(url=url)
        assert root is not None

        for year in root.find_all("loc"):
            y_resp = get_request(url=year.text)
            if y_resp is None: continue

            for bill in y_resp.find_all("loc"):
                b_id = bill.text.split("/")[-1]
                custom_url = f"https://www.govinfo.gov/content/pkg/{b_id}/html/{b_id}.htm"
                response = get_request(url=custom_url)
                if response is None: continue

                filename = create_and_save_record(path=path,
                                                  id=b_id,
                                                  text=response.text,
                                                  source="U.S. Government Information Official WebSite",
                                                  url=custom_url,
                                                  xml=response.prettify())
                log.info(f"Saved {filename}")

    del filename, response, custom_url


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
        data = simulate_browser(BIOGUIDE_URL.format(id=bioguide_id)).get("data")
        if not data: continue

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


def main():
    bills_and_laws(Path(KNOWLEDGE_BASE_PATH + "/bills"))
    geography(Path(KNOWLEDGE_BASE_PATH + "/geography"))
    people(Path(KNOWLEDGE_BASE_PATH + "/people"))
    definitions(Path(KNOWLEDGE_BASE_PATH + "/definitions"))


if __name__ == "__main__":
    main()
