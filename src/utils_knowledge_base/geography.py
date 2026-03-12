import json
import logging
from pathlib import Path

import rootutils
from tqdm import tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)

from src.utils_knowledge_base.utils import safe_filename, fetch_wikipedia_sections, get_from_url, clean_text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

CENSUS_BASE = "https://data.census.gov/api"


# ======================== CENSUS ======================== #

def fetch_census_topics(geo_id: str, meta: dict) -> dict:
    """Fetch all Census topic data for a geography, using already-fetched metadata."""
    topics: set[str] = set()
    for section in meta.get("sections", []):
        topics.add(section["title"])
        for sub in section.get("subsections", []):
            if sub.get("type") == "topic":
                topics.add(sub["topic"])
    topics.discard("Explore Your State")

    data: dict = {}
    for topic in topics:
        response = get_from_url(f"{CENSUS_BASE}/profile/content/topic", params={"g": geo_id, "infoSection": topic})
        if response is None:
            continue
        viz = response.get("vizData", {})
        if viz and viz.get("stats"):
            data['topic'] = f"The {viz['title']} with source {viz['source']} shows that {viz['stats']}"
    return data


# ======================== MAIN ======================== #

def extract_US(path: Path) -> None:
    """Fetch and save data for all 50 US states (+ DC, territories)."""
    response = get_from_url(f"{CENSUS_BASE}/search", params={"g": "010XX00US", "services": "entities"})
    if response is None:
        log.error("Could not retrieve US state list")
        return

    profiles = response.get("response", {}).get("profiles", [])
    log.info("Processing %d states/territories", len(profiles))

    for state in tqdm(profiles, desc="US states"):
        try:
            geo_id = state["facets"]["geography"][0]

            meta = get_from_url(f"{CENSUS_BASE}/profile/metadata", params={"g": geo_id, "includeHighlights": "false"})
            if meta is None:
                log.warning("No metadata for %s (%s)", state["title"], geo_id)
                return

            n = 1
            record = {"id": safe_filename(state["title"])}
            record[f"paragraph_{n}"] = clean_text(state.get("teaser", ""))
            record[f"source_{n}"] = "UNITED STATES CENSUS BUREAU"
            record[f"source_url_{n}"] = f"https://data.census.gov/profile/{state['title']}?g={geo_id}"
            n += 1

            record[f"paragraph_{n}"] = clean_text(meta.get("header", {}).get("description", ""))
            record[f"source_{n}"] = "UNITED STATES CENSUS BUREAU"
            record[f"source_url_{n}"] = f"https://data.census.gov/profile/{state['title']}?g={geo_id}"
            n += 1

            census_data = fetch_census_topics(geo_id, meta)
            for data in census_data.items():
                title, text = data[0], data[1]
                record[f"paragraph_{n}"] = title + ": " + text
                record[f"source_{n}"] = "UNITED STATES CENSUS BUREAU"
                record[f"source_url_{n}"] = f"https://data.census.gov/profile/{state['title']}?g={geo_id}"
                n += 1

            wiki_data = fetch_wikipedia_sections(state["title"])
            for section in wiki_data.items():
                title, text = section[0], section[1]
                record[f"paragraph_{n}"] = title + ": " + clean_text(text)
                record[f"source_{n}"] = "Wikipedia webpage for {}".format(state["title"])
                record[f"source_url_{n}"] = f"https://en.wikipedia.org/wiki/{record['id']}"
                n += 1

            filename = path / f"{record['id']}.json"
            filename.write_text(json.dumps(record, indent=4, ensure_ascii=False), encoding="utf-8")
            log.info("Saved %s", filename)
        except Exception:
            log.exception("Unexpected error processing %s", state.get("title"))


def extract_NER_entities(path: Path, source_file: str = "./notebooks/topic_results.json") -> None:
    """Fetch Wikipedia data for named entities listed in a JSON file."""
    try:
        entities: list[str] = json.loads(Path(source_file).read_text())["PLACES"]
    except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
        log.error("Could not load NER entities from %s: %s", source_file, e)
        return

    for entity in tqdm(entities, desc="NER entities"):
        try:
            sections = fetch_wikipedia_sections(entity)
            if not sections:
                log.warning("No Wikipedia content found for %s", entity)
                continue
            record = {
                "id": safe_filename(entity),
            }
            for idx, section in enumerate(sections.items()):
                title, text = section[0], section[1]
                record[f"paragraph_{idx + 1}"] = title + ": " + text
                record[f"source_{idx + 1}"] = "Wikipedia's Web Page for {}".format(entity)
                record[f"source_url_{idx + 1}"] = "https://en.wikipedia.org/wiki/" + record['id']

            filename = path / "{}.json".format(record['id'])
            filename.write_text(json.dumps(record, indent=4, ensure_ascii=False), encoding="utf-8")
            log.info("Saved %s", filename)
        except Exception:
            log.exception("Unexpected error processing %s", entity)


def geography(knowledge_base_path: str):
    path = Path(f"{knowledge_base_path}/geography/")
    path.mkdir(parents=True, exist_ok=True)

    extract_NER_entities(path)
    extract_US(path)
