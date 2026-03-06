import logging
from pathlib import Path

import rootutils
from bs4 import BeautifulSoup

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)

from script.utils import get_from_url, clean_text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def extract_gov_manual(path):
    URL = "https://www.govinfo.gov/content/pkg/GOVMAN-2025-01-13/xml/GOVMAN-2025-01-13.xml"
    response = get_from_url(URL, return_json=False)
    if response is None:
        raise FileNotFoundError("US GOV MANUAL NOT FOUND.")

    manual = BeautifulSoup(response, "xml").text
    filename = path / "US_government_manual.txt"
    filename.write_text(clean_text(manual), encoding="utf-8")
    log.info("Saved %s", filename)


def extract_from_govinfo(main_url, path):
    # This url contains the links to each year
    bills_response = get_from_url(main_url, return_json=False)
    assert bills_response is not None

    bills_response = BeautifulSoup(bills_response, "xml")
    locations = bills_response.find_all("loc")
    for location in locations:
        # This url retrieves the links for each bill of a certain year
        year_response = get_from_url(location.text, return_json=False)
        if year_response is None: continue

        year_response = BeautifulSoup(year_response, "xml")

        bill_locations = year_response.find_all("loc")
        for bill in bill_locations:
            id = bill.text.split("/")[-1]
            #  now we retrieve the bill
            response = get_from_url(f"https://www.govinfo.gov/content/pkg/{id}/html/{id}.htm", return_json=False)
            if response is None: continue

            response = BeautifulSoup(response, "xml").text

            filename = path / f"{id}.txt"
            filename.write_text(clean_text(response), encoding="utf-8")
            log.info("Saved %s", filename)


if __name__ == '__main__':
    output_path = Path("knowledge_base/bills/")
    output_path.mkdir(parents=True, exist_ok=True)

    extract_gov_manual(output_path)
    extract_from_govinfo("https://www.govinfo.gov/sitemap/BILLS_sitemap_index.xml", output_path)
    extract_from_govinfo("https://www.govinfo.gov/sitemap/PLAW_sitemap_index.xml", output_path)

