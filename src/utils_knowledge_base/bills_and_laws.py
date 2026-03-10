import json
import logging
import os
from pathlib import Path

import pytesseract
import rootutils
from bs4 import BeautifulSoup
from pdf2image import convert_from_path

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


# def extract_international_law(out_path: Path, volume_range: range = range(1, 3198), dpi: int = 200):
#     base_url = "https://treaties.un.org/doc/Publication/UNTS/Volume%20{id}/v{id}.pdf"
#     out_path = Path(out_path)
#
#     for volume_id in tqdm(volume_range, desc="Processing volumes"):
#         out_file = out_path / f"v{volume_id}.json"
#         url = base_url.format(id=volume_id)
#
#         with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
#             tmp_path = Path(tmp.name)
#
#         try:
#             response = requests.get(url, timeout=30, stream=True)
#             response.raise_for_status()
#             tmp_path.write_bytes(response.content)
#
#             pages = convert_from_path(str(tmp_path), dpi=dpi)
#             text = "\n".join(
#                 pytesseract.image_to_string(page)
#                 for page in pages
#             )
#
#             record = {
#                 "title": str(volume_id),
#                 "source": url,
#                 "text": clean_text(text),
#             }
#             out_file.write_text(
#                 json.dumps(record, indent=4, ensure_ascii=False), encoding="utf-8"
#             )
#             log.info("Saved %s", out_file)
#
#         except requests.RequestException as e:
#             log.warning("Failed to download volume %d: %s", volume_id, e)
#
#         except Exception as e:
#             log.error("Error processing volume %d: %s", volume_id, e, exc_info=True)
#
#         finally:
#             tmp_path.unlink(missing_ok=True)
#
#     log.info("Done.")

def extract_universal_declaration(out_path):
    # Download -> "https://www.un.org/sites/un2.un.org/files/2021/03/udhr.pdf"
    out_path = Path(out_path)

    out_file = out_path / "dec_human_rights.json"

    try:
        pages = convert_from_path(os.path.expanduser("~/Downloads/udhr.pdf"), dpi=600)
        text = "\n".join(
            pytesseract.image_to_string(page)
            for page in pages
        )

        record = {
            "title": "Universal Declaration of Human Rights",
            "source": "https://www.un.org/sites/un2.un.org/files/2021/03/udhr.pdf",
            "text": clean_text(text),
        }
        out_file.write_text(
            json.dumps(record, indent=4, ensure_ascii=False), encoding="utf-8"
        )
        log.info("Saved %s", out_file)

    finally:
        os.remove(os.path.expanduser("~/Downloads/udhr.pdf"))


if __name__ == '__main__':
    output_path = Path("knowledge_base/bills/")
    output_path.mkdir(parents=True, exist_ok=True)

    # extract_gov_manual(output_path)
    # extract_from_govinfo("https://www.govinfo.gov/sitemap/BILLS_sitemap_index.xml", output_path)
    # extract_from_govinfo("https://www.govinfo.gov/sitemap/PLAW_sitemap_index.xml", output_path)
    extract_universal_declaration(output_path)
