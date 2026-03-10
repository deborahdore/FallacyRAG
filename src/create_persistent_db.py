from datetime import datetime

import chromadb
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)

if __name__ == '__main__':
    client = chromadb.PersistentClient("./memory")
    collection = client.create_collection(
        name="my_collection",
        metadata={
            "description": "Fallacy RAG collection",
            "created": str(datetime.now())
        }
    )
