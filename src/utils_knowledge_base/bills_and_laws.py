import json
import logging
import os
from pathlib import Path

import rootutils
from bs4 import BeautifulSoup
from pdf2image import convert_from_path
from pytesseract import pytesseract

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)

from src.utils_knowledge_base.utils import get_from_url, clean_text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def extract_gov_manual(path: Path) -> None:
    """Fetch and save the US Government Manual as plain text."""
    URL = "https://www.govinfo.gov/content/pkg/GOVMAN-2025-01-13/xml/GOVMAN-2025-01-13.xml"
    response = get_from_url(URL, return_json=False)
    if response is None:
        raise FileNotFoundError("US GOV MANUAL NOT FOUND.")

    filename = path / "US_government_manual.txt"
    filename.write_text(clean_text(BeautifulSoup(response, "xml").text), encoding="utf-8")
    log.info("Saved %s", filename)


def extract_from_govinfo(main_url: str, path: Path) -> None:
    """Crawl a govinfo sitemap and save each bill as plain text."""
    root = get_from_url(main_url, return_json=False)
    assert root is not None

    for year_loc in BeautifulSoup(root, "xml").find_all("loc"):
        year_response = get_from_url(year_loc.text, return_json=False)
        if year_response is None:
            continue

        for bill_loc in BeautifulSoup(year_response, "xml").find_all("loc"):
            bill_id = bill_loc.text.split("/")[-1]
            response = get_from_url(
                f"https://www.govinfo.gov/content/pkg/{bill_id}/html/{bill_id}.htm",
                return_json=False,
            )
            if response is None:
                continue

            filename = path / f"{bill_id}.txt"
            filename.write_text(clean_text(BeautifulSoup(response, "xml").text), encoding="utf-8")
            log.info("Saved %s", filename)


def extract_universal_declaration(path: Path) -> None:
    """OCR the locally downloaded UDHR PDF and save as JSON."""
    pdf_path = Path("./data/udhr.pdf")
    out_file = path / "dec_human_rights.json"

    text = "\n".join(
        pytesseract.image_to_string(page)
        for page in convert_from_path(pdf_path, dpi=600)
    )
    record = {
        "title": "Universal Declaration of Human Rights",
        "source": "https://www.un.org/sites/un2.un.org/files/2021/03/udhr.pdf",
        "text": clean_text(text),
    }
    out_file.write_text(json.dumps(record, indent=4, ensure_ascii=False), encoding="utf-8")
    log.info("Saved %s", out_file)

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
