import json
import logging
from pathlib import Path
from typing import Optional

import rootutils
from tqdm import tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)

from script.utils import safe_filename, fetch_wikipedia_sections, get_from_url, clean_text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def extract_topics(geo_id: str) -> Optional[set]:
    """Return the set of available Census topic names for a geography."""
    url = f"https://data.census.gov/api/profile/metadata?g={geo_id}&includeHighlights=false"
    response = get_from_url(url)
    if response is None:
        return None

    topics: set[str] = set()
    for section in response.get("sections", []):
        topics.add(section["title"])
        for sub in section.get("subsections", []):
            if sub.get("type") == "topic":
                topics.add(sub["topic"])
    topics.discard("Explore Your State")
    return topics


def fetch_census_topics(geo_id: str) -> dict:
    """Fetch all Census topic data for a geography and return as a dict."""
    data: dict = {}
    topics = extract_topics(geo_id)
    if not topics:
        return data

    for topic in topics:
        response = get_from_url(
            f"https://data.census.gov/api/profile/content/topic",
            params={"g": geo_id, "infoSection": topic},
        )
        if response is None:
            continue
        viz = response.get("vizData", {})
        if viz and viz.get("stats"):
            data[topic] = {
                "title": viz["title"],
                "source": viz["source"],
                "stats": viz["stats"],
            }
    return data


def save_record(record: dict, path: Path) -> None:
    filename = path / f"{safe_filename(record['title'])}.json"
    filename.write_text(json.dumps(record, indent=4, ensure_ascii=False), encoding="utf-8")
    log.info("Saved %s", filename)


def process_state(state: dict, path: Path) -> None:
    """Fetch and save Census + Wikipedia data for a single state."""
    page = state["title"]
    geo_id = state["facets"]["geography"][0]

    meta = get_from_url(
        f"https://data.census.gov/api/profile/metadata?g={geo_id}&includeHighlights=false"
    )
    if meta is None:
        log.warning("No metadata for %s (%s)", page, geo_id)
        return

    record: dict = {
        "title": page,
        "level": state["level"],
        "geoID": geo_id,
        "short description": clean_text(state.get("teaser", "")),
        "description": clean_text(meta.get("header", {}).get("description", "")),
    }

    record.update(fetch_census_topics(geo_id))
    record.update(fetch_wikipedia_sections(page))
    save_record(record, path)


def extract_US(path: Path) -> None:
    """Fetch and save data for all 50 US states (+ DC, territories)."""
    entities_resp = get_from_url("https://data.census.gov/api/search?g=010XX00US&services=entities")
    if entities_resp is None:
        log.error("Could not retrieve US state list")
        return

    profiles = entities_resp.get("response", {}).get("profiles", [])
    log.info("Processing %d states/territories", len(profiles))

    for state in tqdm(profiles, desc="US states"):
        try:
            process_state(state, path)
        except Exception as e:
            log.exception("Unexpected error processing %s: %s", state.get("title"), e)


def extract_NER_entities(path: Path, source_file: str = "./notebooks/topic_results.json") -> None:
    """Fetch Wikipedia data for named entities listed in a JSON file."""
    try:
        gpe_entities: list[str] = json.loads(Path(source_file).read_text())["PLACES"]
    except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
        log.error("Could not load NER entities from %s: %s", source_file, e)
        return

    for entity in tqdm(gpe_entities, desc="NER entities"):
        sections = fetch_wikipedia_sections(entity)
        if not sections:
            log.warning("No Wikipedia content found for %s", entity)
            continue

        record = {"title": entity, "source": "Wikipedia", **sections}
        save_record(record, path)


if __name__ == "__main__":
    output_path = Path("knowledge_base/geography/")
    output_path.mkdir(parents=True, exist_ok=True)

    # extract_US(output_path)
    extract_NER_entities(output_path)
