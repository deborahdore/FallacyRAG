import json
import os

from langchain_community.document_loaders.base import BaseLoader
from langchain_core.documents import Document
from tqdm import tqdm


class FallacyLoader(BaseLoader):
    def __init__(self, source_path: str, directories: list):
        self.source_path = source_path
        self.directories = directories

    def load(self):
        documents = []

        for directory in self.directories:
            dir_path = os.path.join(self.source_path, directory)

            if not os.path.exists(dir_path):
                continue

            files = os.listdir(dir_path)
            for filename in tqdm(files, total=len(files), desc=f"Processing section {directory}"):
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

                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "id": entity_id,
                            "source": source,
                            "url": url,
                            "year": year,
                            "category": directory
                        },
                    )
                )

        return documents
