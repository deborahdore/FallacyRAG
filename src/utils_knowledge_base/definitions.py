import json
import tempfile
from pathlib import Path

import requests
import rootutils
from pdf2image import convert_from_path
from pytesseract import pytesseract

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)

from src.utils_knowledge_base.utils import clean_text, log, fetch_wikipedia_sections


def extract_from_pdf(url, id, source, out_path, dpi: int = 800):
    out_path = Path(out_path)
    out_file = out_path / f"{id}.json"

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
            "id": id,
            "source": source,
            "source_url": url,
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
        "id": argument,
    }
    sections = fetch_wikipedia_sections(argument)
    for idx, section in enumerate(sections.items()):
        title, text = section[0], section[1]
        record[f"paragraph_{idx + 1}"] = title + ": " + clean_text(text)
        record[f"source_{idx + 1}"] = "Wikipedia webpage for {}".format(argument)
        record[f"source_url_{idx + 1}"] = f"https://en.wikipedia.org/wiki/{argument}"

    filename = out_path / f"{argument}.json"
    filename.write_text(json.dumps(record, indent=4, ensure_ascii=False), encoding="utf-8")
    log.info("Saved %s", filename)


def definitions():
    path = Path("./knowledge_base/definitions/")
    path.mkdir(parents=True, exist_ok=True)

    extract_from_pdf(url="https://www.ijcai.org/proceedings/2022/0575.pdf",
                     id="goffredo2022fallacious",
                     source="Fallacious argument classification in political debates",
                     out_path=path)
    extract_from_pdf(url="https://aclanthology.org/2023.emnlp-main.684.pdf",
                     id="goffredo2023argument",
                     source="Argument-based Detection and Classification of Fallacies in Political Debates",
                     out_path=path)
    extract_fallacy_def("Fallacy", path)
