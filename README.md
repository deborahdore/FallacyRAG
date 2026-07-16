# Retrieval-Augmented Generation for Argumentative Fallacies in US Political Debates

<!-- TOC -->
* [Retrieval-Augmented Generation for Argumentative Fallacies in US Political Debates](#retrieval-augmented-generation-for-argumentative-fallacies-in-us-political-debates)
  * [🚀 Overview](#-overview)
  * [🎯 Tasks](#-tasks)
  * [⛁ Knowledge Base](#-knowledge-base)
    * [Knowledge Base Format](#knowledge-base-format)
  * [📁 Project Structure](#-project-structure)
<!-- TOC -->

## 🚀 Overview

**Fallacies** are arguments that rely on faulty or misleading reasoning [(Goffredo et al., 2023)](https://aclanthology.org/2023.emnlp-main.684.pdf).

The goal of this project is to improve **fallacy detection and classification** in political discourse using **Retrieval-Augmented Generation (RAG)**. 
By grounding large language models in external knowledge sources, the system aims to provide more accurate classifications and clearer explanations of fallacious arguments in US political debates.

Experiments are conducted using the **[ElecDeb60to20](https://3ia-data.inria.fr/en/#)** dataset, which contains transcripts from U.S. political debates from 1960 to 2020.

## 🎯 Tasks

The project focuses on three main tasks.

- **Fallacy Detection**: detect whether a segment of speech contains a **fallacious argument**.
- **Fallacy Classification**: given a sentence already identified as fallacious, classify which type of fallacy it
  contains. Unlike detection, this task never sees non-fallacious sentences (`NoFallacy` sentences are dropped during
  dataset construction). The system currently supports the following fallacy categories:
    - _Ad Hominem_: an argument that attacks the opponent personally instead of addressing their position or reasoning.
    - _Appeal to Emotion_: the use of emotionally charged language intended to manipulate the audience’s feelings rather
      than provide logical support.
    - _Appeal to Authority_: an argument that relies on the endorsement of an authority figure or perceived consensus
      without sufficient evidence. It may also involve citing non-experts or majority opinion to support a claim.
    - _Slippery Slope_: an argument suggesting that a relatively small first step will inevitably lead to a chain of
      extreme or unlikely consequences.
    - _False Cause_: the incorrect assumption that a causal relationship exists between two events simply because they
      occur together.
    - _Slogan_: a short and striking phrase designed to provoke enthusiasm or agreement rather than present a logical
      argument.

## ⛁ Knowledge Base

To ground the language model, we constructed a **knowledge base of approximately 5GB of political and contextual
documents**. The corpus combines official government sources with curated background information.
The knowledge base is organized into seven main categories.

- **Bills and Laws**: this collection includes congressional bills and public and private laws approved by the U.S.
  Congress, collected from their official
  repository: [GovInfo](https://www.govinfo.gov/sitemaps), [U.S. Government Manual](https://www.govinfo.gov/app/collection/GOVMAN)
  and [Universal Declaration of Human Rights](https://www.un.org/sites/un2.un.org/files/2021/03/udhr.pdf)
- **Geography**: geographic and demographic information about U.S. states was collected from
  the [U.S. Census](https://data.census.gov/webpages)and Wikipedia.
- **Newspapers**: news articles providing political context for the debates, collected from
  the [Chronicling America](https://www.loc.gov/collections/chronicling-america/) collection at the Library of Congress
  and OCR-processed into text.
- **People**: biographical information about members of the U.S. Congress was collected from
  the [Biographical Directory of the United States Congress](https://bioguide.congress.gov/search). Additional entries
  from Wikipedia were added for public figures referenced during the debates.
- **History**: U.S. historical context, built from per-year Wikipedia "*{year} in the United States*" pages (from 1900
  onward) together with broad "*History of the United States*" era-overview pages and their linked articles. Each record
  is tagged with its year to support temporal filtering during retrieval.
- **Definitions**: conceptual definitions and explanations of argumentative fallacies were compiled from peer-reviewed
  research papers and reference
  sources ([Goffredo et al., 2023](https://aclanthology.org/2023.emnlp-main.684.pdf), [Goffredo et al., 2022](https://www.ijcai.org/proceedings/2022/0575.pdf)),
  a curated set of recent fallacy / argumentation papers retrieved from [OpenAlex](https://openalex.org/), and the
  Wikipedia fallacy pages.
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
  "year": "[when available]"
}
```

## 📁 Project Structure

```
datasets/                          # source data + per-dataset RAG-enrichment scripts
  elecdeb60to20-dataset/            # ElecDeb60to20 (main dataset)
    train.csv, dev.csv, test.csv    # raw splits
    detection/create_db.py          # builds RAG-enriched detection datasets
    classification/create_db.py     # builds RAG-enriched classification datasets
  reddit-dataset/                   # same layout, Reddit dataset
  falcon-dataset/                   # same layout, Falcon dataset

script/                             # sbatch / driver scripts
  00-create_knowledge_base.sh
  01-create_persistent_db.sh
  02-create_database*.sh            # runs create_db.py to build RAG-enriched datasets
  03-fallacy_no_rag*.sh              # no-RAG training sweep (encoder / LLM)
  04-fallacy_rag*.sh                 # RAG training sweep (encoder / LLM)
  generate_best_hparam_script.py     # generates a 05-best_hparams_*.sh from a finished sweep
  extract_results.py                 # aggregates output/one-run into per-config CSVs
  extract_results_5runs.py           # aggregates output/five-runs into per-config CSVs

output/
  one-run/<dataset>-<task>/<config>/<model>/<run>/    # single-seed grid search results
  five-runs/<dataset>-<task>/<config>/<model>/seed_<n>/<run>/  # multi-seed best-hparam results
```

Each `create_db.py` reads its raw splits from `--dataset-dir` (default `datasets/<name>-dataset`) and writes the
RAG-enriched splits back under `<dataset-dir>/<task>/<config>/{train,dev,test}.csv`.

The `03-*.sh` / `04-*.sh` training scripts default to the ElecDeb60to20 dataset. To point them at Reddit or Falcon
instead, edit the `data` variable near the top of the script:

```bash
data="elecdeb60to20"  # elecdeb60to20 | falcon | reddit
```

which is used to build both the input paths (`datasets/$data-dataset/...`) and the output paths
(`output/one-run/$data-$task/...`).
