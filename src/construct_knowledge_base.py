from pathlib import Path

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)

from src.utils_knowledge_base.bills_and_laws import extract_universal_declaration, extract_from_govinfo, \
    extract_gov_manual
from src.utils_knowledge_base.geography import extract_US, extract_NER_entities
from src.utils_knowledge_base.people import extract_wiki_people, extract_congress
from src.utils_knowledge_base.definitions import extract_from_pdf, extract_fallacy_def

if __name__ == '__main__':
    Path("./knowledge_base/bills/").mkdir(parents=True, exist_ok=True)
    Path("./knowledge_base/geography/").mkdir(parents=True, exist_ok=True)
    Path("./knowledge_base/people/").mkdir(parents=True, exist_ok=True)
    Path("./knowledge_base/definitions/").mkdir(parents=True, exist_ok=True)

    # ======================== PEOPLE ======================== #
    extract_congress(Path("./knowledge_base/people/"))
    extract_wiki_people(Path("./knowledge_base/people/"))
    # ======================== BILLS ======================== #

    extract_gov_manual(Path("knowledge_base/bills/"))
    extract_from_govinfo("https://www.govinfo.gov/sitemap/BILLS_sitemap_index.xml", Path("knowledge_base/bills/"))
    extract_from_govinfo("https://www.govinfo.gov/sitemap/PLAW_sitemap_index.xml", Path("knowledge_base/bills/"))
    extract_universal_declaration(Path("knowledge_base/bills/"))
    # ======================== GEOGRAPHY ======================== #

    extract_NER_entities(Path("knowledge_base/geography/"))
    extract_US(Path("knowledge_base/geography/"))
    # ======================== DEFINITIONS ======================== #

    extract_from_pdf(url="https://www.ijcai.org/proceedings/2022/0575.pdf",
                     id="goffredo2022fallacious",
                     source="Fallacious argument classification in political debates",
                     out_path=Path("knowledge_base/definitions/"))
    extract_from_pdf(url="https://aclanthology.org/2023.emnlp-main.684.pdf",
                     id="goffredo2023argument",
                     source="Argument-based Detection and Classification of Fallacies in Political Debates",
                     out_path=Path("knowledge_base/definitions/"))
    extract_fallacy_def("Fallacy", Path("knowledge_base/definitions/"))
