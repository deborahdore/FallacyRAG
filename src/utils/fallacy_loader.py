import json
import os

from langchain_community.document_loaders.base import BaseLoader
from langchain_core.documents import Document
from tqdm import tqdm


class FallacyLoader(BaseLoader):
    """Load knowledge-base JSON records from a set of folders into LangChain Documents."""

    def __init__(self, source_path: str, directories: list):
        """Store the root path and the list of (sub)folders to scan."""
        self.source_path = source_path
        self.directories = directories

    def load(self):
        """Read every JSON file in each folder and return them as Documents."""
        documents = []

        for directory in self.directories:
            # An entry may be a plain folder name, or a (folder, category_label) tuple so a
            # nested folder (e.g. "definitions/papers") can get a clean category label.
            if isinstance(directory, (tuple, list)):
                folder, category = directory
            else:
                folder, category = directory, directory

            dir_path = os.path.join(self.source_path, folder)

            if not os.path.exists(dir_path):
                continue

            files = os.listdir(dir_path)
            for filename in tqdm(files, total=len(files), desc=f"Processing section {category}"):
                if not filename.endswith(".json"):
                    continue

                file_path = os.path.join(dir_path, filename)

                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                entity_id = data.get("id", filename)
                text = data.get("text", "")
                source = data.get("source", "")
                url = data.get("url", "")
                year = data.get("year", 1900)

                metadata = {
                    "id": entity_id,
                    "source": source,
                    "url": url,
                    "year": year,
                    "category": category,
                }
                # Carry the fallacy label (present on exemplar records) so it can be used
                # for fallacy-keyed retrieval / metadata filtering.
                fallacy_label = data.get("fallacy_label")
                if fallacy_label is not None:
                    metadata["fallacy_label"] = fallacy_label

                documents.append(Document(page_content=text, metadata=metadata))

        return documents
