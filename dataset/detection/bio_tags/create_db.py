import ast
import json

import pandas as pd
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)


def convert(tag):
    if tag == "O":
        return "O"
    elif tag.startswith("B-"):
        return "B-Fallacy"
    elif tag.startswith("I-"):
        return "I-Fallacy"
    raise ValueError(f"Unknown tag: {tag}")


for split in ["train", "dev", "test"]:
    dataset = []
    df = pd.read_csv(f"dataset/{split}.csv")
    tokens = df['Context'].tolist()
    ner_tags = df['bio_tags'].tolist()

    for idx, (tok, tag) in enumerate(zip(tokens, ner_tags)):
        ner_tags = [convert(t) for t in ast.literal_eval(tag)]
        json_obj = {
            "id": idx,
            "tokens": tok.split(),
            "ner_tags": ner_tags,
        }
        dataset.append(json_obj)

    with open(f"dataset/detection/bio_tags/{split}.json", 'w') as f:
        json.dump(dataset, f, indent=4)
