import json
import tempfile
from pathlib import Path

import requests
import rootutils
from pdf2image import convert_from_path
from pytesseract import pytesseract

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)

from src.utils_knowledge_base.utils import clean_text, log, fetch_wikipedia_sections


def extract_from_pdf(url, file_name, out_path, dpi: int = 600):
    out_path = Path(out_path)
    out_file = out_path / f"{file_name}.json"

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        response = requests.get(url, timeout=30, stream=True)
        response.raise_for_status()
        tmp_path.write_bytes(response.content)

        pages = convert_from_path(str(tmp_path), dpi=dpi)
        text = "\n".join(
            pytesseract.image_to_string(page)
            for page in pages
        )

        record = {
            "title": file_name,
            "source": url,
            "text": clean_text(text),
        }
        out_file.write_text(
            json.dumps(record, indent=4, ensure_ascii=False), encoding="utf-8"
        )
        log.info("Saved %s", out_file)

    except Exception as e:
        log.error("Error processing url %s", url)

    finally:
        tmp_path.unlink(missing_ok=True)

    log.info("Done.")


def extract_fallacy_def(argument, out_path):
    record = {
        "title": argument,
        **fetch_wikipedia_sections(argument),
    }

    filename = out_path / "fallacies.json"
    filename.write_text(json.dumps(record, indent=4, ensure_ascii=False), encoding="utf-8")
    log.info("Saved %s", filename)
