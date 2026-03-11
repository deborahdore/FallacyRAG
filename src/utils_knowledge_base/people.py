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

from src.utils_knowledge_base.utils import safe_filename, fetch_wikipedia_sections, BROWSER_TIMEOUT_MS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

BIOGUIDE_URL = "https://bioguide.congress.gov/search/bio/{id}.json"
NER_SOURCE = "./notebooks/topic_results.json"

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


# ======================== BROWSER ======================== #

def simulate_browser(url: str, timeout_ms: int = BROWSER_TIMEOUT_MS) -> dict | None:
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
        except PlaywrightTimeout:
            log.warning("Timeout fetching %s", url)
        except json.JSONDecodeError as e:
            log.warning("JSON parse error for %s: %s", url, e)
        except Exception as e:
            log.warning("Browser error for %s: %s", url, e)
        finally:
            browser.close()
    return None


# ======================== PARSING ======================== #

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


def parse_congressman(raw: dict, source: str):
    """Build a clean congressman record from a raw bioguide API response."""
    data = raw.get("data")
    if not data:
        return None

    given = data.get("givenName", "").strip()
    family = data.get("familyName", "").strip()
    name = given + " " + family
    n = 1
    record: dict = {
        "id": safe_filename(name.strip()),
        f"paragraph_{n}": f" {name} is {data.get('profileText', '')}".strip(),
        f"source_{n}": "Official Biographical Directory of the United States Congress",
        f"source_url_{n}": source
    }

    for job_entry in data.get("jobPositions", []):
        try:
            job = parse_job(job_entry)
            n += 1
            role = job.get("role", "")
            represents = job.get("represents", "")
            congressAffiliation = job.get("congressAffiliation", "")
            startDate = job.get("startDate", "")
            endDate = job.get("endDate", "")
            record[
                f'paragraph_{n}'] = f"{name} worked as {role} for {represents} in {congressAffiliation} from {startDate} to {endDate}."
            record[f'source_{n}'] = source
        except Exception as e:
            log.warning("Could not parse job for %s: %s", name, e)

    return record, name, n + 1


# ======================== MAIN ======================== #

def extract_congress(path: Path, csv_path: str = "./data/HSall_members.csv") -> None:
    """Fetch and save bioguide data for all congress members in the CSV."""
    ids: list[str] = pd.read_csv(os.path.expanduser(csv_path))["bioguide_id"].dropna().unique().tolist()
    log.info("Processing %d unique congress members", len(ids))

    ok = err = 0
    for bid in tqdm(ids, desc="Congress members"):
        raw = simulate_browser(BIOGUIDE_URL.format(id=bid))
        record, name, last_paragraph = parse_congressman(raw, source=BIOGUIDE_URL.format(id=bid)) if raw else None
        wiki = fetch_wikipedia_sections(name) if name else None
        for section in wiki.items():
            title, text = section[0], section[1]
            record[f"paragraph_{last_paragraph}"] = title + ": " + text
            record[f"source_{last_paragraph}"] = "Wikipedia's Web Page for {}".format(name)
            record[f"source_url_{last_paragraph}"] = "https://en.wikipedia.org/wiki/" + safe_filename(name)
            last_paragraph += 1

        if record is None:
            log.warning("Skipping bioguide ID %s", bid)
            err += 1
            continue

        filename = path / f"{record['id']}.json"
        filename.write_text(json.dumps(record, indent=4, ensure_ascii=False), encoding="utf-8")
        log.info("Saved %s", filename)
        ok += 1

    log.info("Congress: %d saved, %d failed", ok, err)


def extract_wiki_people(path: Path, source_file: str = NER_SOURCE) -> None:
    """Fetch Wikipedia data for NER-extracted person names."""
    people: list[str] = json.loads(Path(source_file).read_text(encoding="utf-8"))["PEOPLE"]
    log.info("Processing %d NER people", len(people))

    ok = err = 0
    for name in tqdm(people, desc="NER people"):
        name = safe_filename(name)
        record = {
            "id": name,
        }
        filename = path / f"{record['id']}.json"
        if filename.exists():
            continue

        sections = fetch_wikipedia_sections(name)

        if not sections:
            log.warning("No Wikipedia content for %s", name)
            err += 1
            continue
        for idx, section in enumerate(sections.items()):
            title, text = section[0], section[1]
            record[f"paragraph_{idx + 1}"] = title + ": " + text
            record[f"source_{idx + 1}"] = "Wikipedia's Web Page for {}".format(name)
            record[f"source_url_{idx + 1}"] = "https://en.wikipedia.org/wiki/" + safe_filename(name)

        filename.write_text(json.dumps(record, indent=4, ensure_ascii=False), encoding="utf-8")
        log.info("Saved %s", filename)
        ok += 1

    log.info("NER people: %d saved, %d failed", ok, err)

def people():
    path = Path("./knowledge_base/people/")
    path.mkdir(parents=True, exist_ok=True)

    extract_congress(path)
    extract_wiki_people(path)