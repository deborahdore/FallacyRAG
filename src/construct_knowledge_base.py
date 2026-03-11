import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)

from src.utils_knowledge_base.bills_and_laws import bills_and_laws
from src.utils_knowledge_base.geography import geography
from src.utils_knowledge_base.people import people
from src.utils_knowledge_base.definitions import definitions

if __name__ == '__main__':
    people()
    bills_and_laws()
    geography()
    definitions()
