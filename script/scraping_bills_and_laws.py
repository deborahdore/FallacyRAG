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


if __name__ == '__main__':
    output_path = Path("knowledge_base/bills/")
    output_path.mkdir(parents=True, exist_ok=True)

    extract_gov_manual(output_path)
