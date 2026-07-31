# Retrieval-Augmented Generation for Argumentative Fallacies in US Political Debates

<!-- TOC -->

* [Retrieval-Augmented Generation for Argumentative Fallacies in US Political Debates](#retrieval-augmented-generation-for-argumentative-fallacies-in-us-political-debates)
    * [🚀 Overview](#-overview)
    * [🎯 Tasks](#-tasks)
        * [Fallacy Detection](#fallacy-detection)
        * [Fallacy Classification](#fallacy-classification)
        * [Relation Classification](#relation-classification)
    * [⛁ Knowledge Base](#-knowledge-base)
        * [Knowledge Base Format](#knowledge-base-format)
    * [🗂️ Datasets](#-datasets)
    * [⚙️ Setup & Pipeline](#-setup--pipeline)
    * [📁 Project Structure](#-project-structure)

<!-- TOC -->

## 🚀 Overview

**Fallacies** are arguments that rely on faulty or misleading
reasoning [(Goffredo et al., 2023)](https://aclanthology.org/2023.emnlp-main.684.pdf).

The goal of this project is to improve **fallacy detection and classification** in political discourse using
**Retrieval-Augmented Generation (RAG)**. By grounding models in external knowledge sources, the system aims to provide
more accurate classifications and clearer explanations of fallacious arguments in US political debates.

The main experiments use the **[ElecDeb60to20](https://3ia-data.inria.fr/en/#)** dataset, which contains transcripts
from U.S. political debates from 1960 to 2020. The same pipeline also runs on the **propaganda** dataset, and a
separate relation-classification pipeline draws on the **LIARArg**, **DEBAGREEMENT**, and **propaganda** corpora.

Both an **encoder** track (BERT-style fine-tuning) and an **LLM** track (zero-shot, few-shot, and LoRA/QLoRA
fine-tuning) are supported for every task, each with and without RAG context.

## 🎯 Tasks

The project covers three tasks, each documented in its own README.

### Fallacy Detection

Binary decision of whether a segment of speech contains a **fallacious argument** (`Fallacy` / `NoFallacy`).
See [`src/detection/README.md`](src/detection/README.md).

### Fallacy Classification

Given a sentence already identified as fallacious, classify **which type** of fallacy it contains. Unlike detection,
this task never sees non-fallacious sentences (`NoFallacy` sentences are dropped during dataset construction).
See [`src/classification/README.md`](src/classification/README.md).

The label inventory is **dataset-aware** and is selected automatically from the input path. Single- vs. multi-label is
auto-detected from the dataset columns.

- **ElecDeb60to20** — six categories, exactly one per sentence:
    - _Ad Hominem_: attacking the opponent personally instead of their position or reasoning.
    - _Appeal to Emotion_: emotionally charged language used to manipulate feelings rather than provide logical support.
    - _Appeal to Authority_: relying on an authority figure or perceived consensus (including non-experts or majority
      opinion) without sufficient evidence.
    - _Slippery Slope_: claiming a small first step will inevitably lead to extreme or unlikely consequences.
    - _False Cause_: assuming a causal link between two events merely because they co-occur.
    - _Slogan_: a short, striking phrase meant to provoke agreement rather than argue.
### Relation Classification

Classify the **argumentative relation** between a pair of components as `support`, `attack`, or `no_relation`. This
task serves the RAG pipeline: the extracted relations feed the `relations` query mode of the dataset builders (see
below), enriching the retrieval query with the fallacy's supporting/attacking context. Models are trained on
relation-labeled corpora (**LIARArg**, **DEBAGREEMENT**) and then used to predict relations on the **propaganda**
data. See [`src/relation-extraction/README.md`](src/relation-extraction/README.md).

## ⛁ Knowledge Base

To ground the model, we constructed a **knowledge base of approximately 5GB of political and contextual documents**. The
corpus combines official government sources with curated background information, organized into eight sections.

- **Bills and Laws**: congressional bills and public/private laws approved by the U.S. Congress, collected from their
  official repository [GovInfo](https://www.govinfo.gov/sitemaps),
  the [U.S. Government Manual](https://www.govinfo.gov/app/collection/GOVMAN), and
  the [Universal Declaration of Human Rights](https://www.un.org/sites/un2.un.org/files/2021/03/udhr.pdf).
- **Geography**: geographic and demographic information about U.S. states, from
  the [U.S. Census](https://data.census.gov/webpages) and Wikipedia.
- **Newspapers**: news articles providing political context, from
  the [Chronicling America](https://www.loc.gov/collections/chronicling-america/) collection at the Library of Congress,
  OCR-processed into text.
- **People**: biographical information about members of the U.S. Congress, from
  the [Biographical Directory of the United States Congress](https://bioguide.congress.gov/search), plus Wikipedia
  entries for public figures referenced during the debates.
- **History**: U.S. historical context, built from per-year Wikipedia "*{year} in the United States*" pages (from 1900
  onward) together with broad "*History of the United States*" era-overview pages and their linked articles. Each record
  is tagged with its year to support temporal filtering during retrieval.
- **Definitions**: conceptual definitions and explanations of argumentative fallacies compiled from peer-reviewed
  research ([Goffredo et al., 2023](https://aclanthology.org/2023.emnlp-main.684.pdf),
  [Goffredo et al., 2022](https://www.ijcai.org/proceedings/2022/0575.pdf)), a curated set of recent fallacy /
  argumentation papers retrieved from [OpenAlex](https://openalex.org/), and the Wikipedia fallacy pages.
- **Exemplars**: labeled examples of fallacious arguments drawn from public fallacy datasets
  ([LOGIC](https://huggingface.co/datasets/tasksource/logical-fallacy) and
  [MAFALDA](https://github.com/ChadiHelwe/MAFALDA)), mapped onto the project's fallacy taxonomy to provide concrete
  reference instances of each fallacy type.

### Knowledge Base Format

Each entry in the knowledge base is stored as a JSON document.

```json
{
  "id": "",
  "text": "",
  "xml": "[when available]",
  "source": "",
  "url": "",
  "year": "[when available]",
  "fallacy_label": "[exemplars only]"
}
```

## 🗂️ Datasets

| Dataset           | Tasks                                              | Role                                                                          |
|-------------------|----------------------------------------------------|-------------------------------------------------------------------------------|
| **ElecDeb60to20** | detection, classification                          | Main dataset (U.S. political debates, 1960–2020).                             |
| **LIARArg**       | relation-classification                            | Training data for the relation classifier (`support / attack / no_relation`). |
| **DEBAGREEMENT**  | relation-classification                            | Training data for the agreement/relation classifier.                          |
| **propaganda**    | detection, relation-classification                 | Relation predictions produced by the LIARArg-trained model.                   |

Each dataset lives under `datasets/<name>-dataset/` with raw `train.csv` / `dev.csv` / `test.csv` splits (and an
`original/` folder holding the source files). RAG-enriched variants are written into per-task, per-config subfolders by
`create_db.py` (see below).

## ⚙️ Setup & Pipeline

Create the conda environment (named `rags`):

```bash
conda env create -f environment.yaml
conda activate rags
```

The `script/` folder is numbered to reflect the pipeline order:

1. **`00-create_knowledge_base.sh`** — build the knowledge base (`src/create_knowledge_base.py`).
2. **`01-create_persistent_db.sh`** — index the knowledge base into a persistent Chroma vector store, once per
   embedding model (`src/create_persistent_db.py`).
3. **`02-create_database*.sh`** — run each dataset's `create_db.py` to produce RAG-enriched train/dev/test splits.
4. **`03-fallacy_no_rag*.sh` / `04-fallacy_rag*.sh`** — training/evaluation sweeps, without and with RAG, for the
   encoder (`*.sh`) and LLM (`*_llm.sh`) tracks.

After a sweep, `extract_results.py` / `extract_results_5runs.py` aggregate the per-run metrics into CSVs, and
`generate_best_hparam_script.py` emits a `05-best_hparams_*.sh` that re-runs only the best configuration per model
across several seeds. `check_one_run.py` / `check_five_runs.py` verify grid completeness.

The `03-*.sh` / `04-*.sh` scripts default to ElecDeb60to20. To target another dataset, edit the `data` variable near
the top of the script:

```bash
data="elecdeb60to20"  # elecdeb60to20 | propaganda
```

which feeds both the input paths (`datasets/$data-dataset/...`) and the output paths (`output/one-run/$data-$task/...`).

## 📁 Project Structure

```
src/
  create_knowledge_base.py          # build the KB (bills, people, geography, history, papers, exemplars…)
  create_persistent_db.py           # index the KB into a persistent Chroma vector store
  perform_retrieval.py              # CLI to query the vector store (dense / BM25)
  detection/                        # binary fallacy detection (encoder + LLM) — see its README
  classification/                   # fallacy-type classification (encoder + LLM) — see its README
  relation-extraction/              # relation classification + relation prediction — see its README
  utils/                            # KB loader, HTTP/Wikipedia helpers, dataset-aware prompts

datasets/                           # source data + per-dataset RAG-enrichment scripts
  elecdeb60to20-dataset/            # main dataset
    train.csv, dev.csv, test.csv    # raw splits
    detection/create_db.py          # builds RAG-enriched detection datasets
    classification/create_db.py     # builds RAG-enriched classification datasets
    <task>/<config>/                # generated RAG-enriched splits (one folder per config)
  propaganda-dataset/, LIARArg-dataset/, debagreeement-dataset/   # relation-classification data

script/                             # sbatch / driver scripts (numbered by pipeline stage)
  00-create_knowledge_base.sh
  01-create_persistent_db.sh
  02-create_database*.sh            # runs create_db.py to build RAG-enriched datasets
  03-fallacy_no_rag*.sh             # no-RAG training sweep (encoder / LLM)
  04-fallacy_rag*.sh                # RAG training sweep (encoder / LLM)
  generate_best_hparam_script.py    # generates a 05-best_hparams_*.sh from a finished sweep
  extract_results.py                # aggregates output/one-run into per-config CSVs
  extract_results_5runs.py          # aggregates output/five-runs into per-config CSVs
  check_one_run.py, check_five_runs.py   # verify grid completeness

notebooks/                          # topic / knowledge-base / error analysis
output/
  one-run/<dataset>-<task>/<config>/<model>/<run>/            # single-seed grid-search results
  five-runs/<dataset>-<task>/<config>/<model>/seed_<n>/<run>/ # multi-seed best-hparam results
  trained-models/                   # checkpoints reused by the relation-prediction scripts
```

Each `create_db.py` reads its raw splits from `--dataset-dir` (default `datasets/<name>-dataset`) and writes the
RAG-enriched splits back under `<dataset-dir>/<task>/<config>/{train,dev,test}.csv`, where `<config>` encodes the
retrieval settings (temporal, retriever, search, rerank, query, summarizer).
