# Sources:
#   https://voteview.com/data | Member ideology + votes
#   https://voteview.com/static/data/out/members/HSall_members.csv
import json
import logging
import os
from pathlib import Path

import pandas as pd
import rootutils
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout
from tqdm import tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)

from script.utils import safe_filename, fetch_wikipedia_sections

BIOGUIDE_URL = "https://bioguide.congress.gov/search/bio/{id}.json"
NER_SOURCE = "./notebook/topic_results.json"
BROWSER_TIMEOUT_MS = 30_000  # 30 s per page navigation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

STATE_ABBR2NAME: dict[str, str] = {
    "CT": "CONNECTICUT", "ME": "MAINE", "MA": "MASSACHUSETTS",
    "NH": "NEW HAMPSHIRE", "RI": "RHODE ISLAND", "VT": "VERMONT",
    "DE": "DELAWARE", "NJ": "NEW JERSEY", "NY": "NEW YORK",
    "PA": "PENNSYLVANIA", "IL": "ILLINOIS", "IN": "INDIANA",
    "MI": "MICHIGAN", "OH": "OHIO", "WI": "WISCONSIN",
    "IA": "IOWA", "KS": "KANSAS", "MN": "MINNESOTA",
    "MO": "MISSOURI", "NE": "NEBRASKA", "ND": "NORTH DAKOTA",
    "SD": "SOUTH DAKOTA", "VA": "VIRGINIA", "AL": "ALABAMA",
    "AR": "ARKANSAS", "FL": "FLORIDA", "GA": "GEORGIA",
    "LA": "LOUISIANA", "MS": "MISSISSIPPI", "NC": "NORTH CAROLINA",
    "SC": "SOUTH CAROLINA", "TX": "TEXAS", "KY": "KENTUCKY",
    "MD": "MARYLAND", "OK": "OKLAHOMA", "TN": "TENNESSEE",
    "DC": "WASHINGTON, D.C.", "WV": "WEST VIRGINIA", "AZ": "ARIZONA",
    "CO": "COLORADO", "ID": "IDAHO", "MT": "MONTANA",
    "NV": "NEVADA", "NM": "NEW MEXICO", "UT": "UTAH",
    "WY": "WYOMING", "CA": "CALIFORNIA", "OR": "OREGON",
    "WA": "WASHINGTON", "AK": "ALASKA", "HI": "HAWAII",
    "US": "UNITED STATES",
}


def simulate_browser(url: str, timeout_ms: int = BROWSER_TIMEOUT_MS) -> dict | None:
    """Fetch a JSON endpoint via a headless Chromium browser.
    Returns the parsed dict, or None on any failure.
    """
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            context = browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36"
                )
            )
            page = context.new_page()
            page.set_default_timeout(timeout_ms)
            page.goto(url, wait_until="networkidle")
            text = page.evaluate("() => document.body.innerText")
            return json.loads(text)
        except PlaywrightTimeout:
            log.warning("Timeout fetching %s", url)
        except json.JSONDecodeError as e:
            log.warning("JSON parse error for %s: %s", url, e)
        except Exception as e:
            log.warning("Browser error for %s: %s", url, e)
        finally:
            browser.close()
    return None


def parse_job(job: dict) -> dict:
    """Extract a structured job record from a raw bioguide jobPosition entry."""
    record: dict = {"role": job["job"]["name"]}
    affil = job.get("congressAffiliation")
    if not affil:
        return record

    congress = affil.get("congress", {})
    record["congressAffiliation"] = congress.get("name")
    record["startDate"] = congress.get("startDate")
    if end := congress.get("endDate"):
        record["endDate"] = end

    parties = affil.get("partyAffiliation", [])
    if parties:
        record["partyAffiliation"] = parties[0]["party"]["name"]

    region = affil.get("represents", {}).get("regionCode")
    if region:
        record["represents"] = STATE_ABBR2NAME.get(region, region)

    return record


def parse_congressman(raw: dict) -> dict | None:
    """Build a clean congressman record from a raw bioguide API response.
    Returns None if the response is missing required fields.
    """
    data = raw.get("data")
    if not data:
        return None

    given = data.get("givenName", "").strip()
    family = data.get("familyName", "").strip()
    if not given and not family:
        log.warning("Bioguide record missing name: %s", data)
        return None

    record: dict = {
        "name": f"{given} {family}".strip(),
        "birthDate": data.get("birthDate"),
        "deathDate": data.get("deathDate"),
        "bio": data.get("profileText", ""),
    }

    jobs = []
    for job_entry in data.get("jobPositions", []):
        try:
            jobs.append(parse_job(job_entry))
        except Exception as e:
            log.warning("Could not parse job for %s: %s", record["name"], e)

    if jobs:
        record["jobs"] = jobs

    return record


def process_congressman(bioguide_id: str, path: Path) -> bool:
    """Fetch, parse, and save one congressman. Returns True on success."""
    raw = simulate_browser(BIOGUIDE_URL.format(id=bioguide_id))
    if raw is None:
        log.warning("No data for bioguide ID %s", bioguide_id)
        return False

    record = parse_congressman(raw)
    if record is None:
        log.warning("Could not parse record for bioguide ID %s", bioguide_id)
        return False

    filename = path / f"{safe_filename(record['name'])}.json"
    filename.write_text(json.dumps(record, indent=4, ensure_ascii=False), encoding="utf-8")
    log.info("Saved %s", filename)
    return True


def process_wiki_person(name: str, path: Path) -> bool:
    """Fetch Wikipedia sections for a named person and save. Returns True on success."""
    sections = fetch_wikipedia_sections(name)
    if not sections:
        log.warning("No Wikipedia content for %s", name)
        return False

    record = {"name": name, **sections}
    filename = path / f"{safe_filename(name)}.json"
    if os.path.isfile(filename):
        filename = path / f"{safe_filename(name)}_1.json"
    filename.write_text(json.dumps(record, indent=4, ensure_ascii=False), encoding="utf-8")
    log.info("Saved %s", filename)
    return True


def extract_congress(path: Path, csv_path: str) -> None:
    """Fetch and save bioguide data for all congress members in the CSV."""
    ids: list[str] = pd.read_csv(os.path.expanduser(csv_path))["bioguide_id"].dropna().unique().tolist()
    log.info("Processing %d unique congress members", len(ids))

    ok = err = 0
    for bid in tqdm(ids, desc="Congress members"):
        if process_congressman(bid, path):
            ok += 1
        else:
            err += 1

    log.info("Congress: %d saved, %d failed", ok, err)


def extract_wiki_people(path: Path, source_file: str = NER_SOURCE) -> None:
    """Fetch Wikipedia data for NER-extracted person names."""
    people: list[str] = json.loads(Path(source_file).read_text(encoding="utf-8"))["PEOPLE"]
    log.info("Processing %d NER people", len(people))

    ok = err = 0
    for name in tqdm(people, desc="NER people"):
        if process_wiki_person(name, path):
            ok += 1
        else:
            err += 1
    log.info("NER people: %d saved, %d failed", ok, err)


if __name__ == "__main__":
    output_path = Path("./knowledge_base/people/")
    output_path.mkdir(parents=True, exist_ok=True)

    extract_congress(output_path, csv_path="./data/HSall_members.csv")
    extract_wiki_people(output_path)
