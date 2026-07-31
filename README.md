# Retrieval-Augmented Generation for Argumentative Fallacies in US Political Debates

<!-- TOC -->

* [Retrieval-Augmented Generation for Argumentative Fallacies in US Political Debates](#retrieval-augmented-generation-for-argumentative-fallacies-in-us-political-debates)
    * [🚀 Overview](#-overview)
    * [🔬 Pipeline](#-pipeline)
    * [🎯 Tasks](#-tasks)
        * [Fallacy Detection](#fallacy-detection)
        * [Fallacy Classification](#fallacy-classification)
        * [Relation Classification](#relation-classification)
    * [⛁ Knowledge Base](#-knowledge-base)
        * [Knowledge Base Format](#knowledge-base-format)
    * [🗂️ Datasets](#-datasets)
        * [Configuration Folder Names](#configuration-folder-names)
    * [⚙️ Setup](#-setup)
    * [▶️ Running the Pipeline](#-running-the-pipeline)
        * [Switching Dataset or Task](#switching-dataset-or-task)
        * [Experiment Grid](#experiment-grid)
    * [📊 Results & Analysis](#-results--analysis)
    * [📁 Project Structure](#-project-structure)
    * [📄 License](#-license)

<!-- TOC -->

## 🚀 Overview

**Fallacies** are arguments that rely on faulty or misleading
reasoning [(Goffredo et al., 2023)](https://aclanthology.org/2023.emnlp-main.684.pdf).

The goal of this project is to improve **fallacy detection and classification** in political discourse using
**Retrieval-Augmented Generation (RAG)**. By grounding models in external knowledge sources, the system aims to provide
more accurate classifications and clearer explanations of fallacious arguments in US political debates.

The main experiments use the **[ElecDeb60to20](https://3ia-data.inria.fr/en/#)** dataset, which contains transcripts
from U.S. political debates from 1960 to 2020. The same pipeline also runs on the **propaganda** dataset, and a
separate relation-classification pipeline draws on the **LIARArg** and **DEBAGREEMENT** corpora.

Both an **encoder** track (BERT-style fine-tuning) and an **LLM** track (zero-shot, few-shot, and LoRA/QLoRA
fine-tuning) are supported for every task, each with and without RAG context.

## 🔬 Pipeline

![FallacyRAG pipeline](docs/pipeline.png)

Each sentence is turned into a retrieval **query** — the bare sentence, or the sentence enriched with the argumentative
components that support or attack it. The query is sent to the **knowledge base** through one of four retrievers (three
dense embedding models or BM25), optionally restricted to documents predating the debate and optionally reranked by a
cross-encoder. The retrieved passages are then concatenated with the sentence and its speech turn into the **model
input**, and an encoder or a decoder produces the final prediction.

Crucially, **retrieval happens offline**. The `create_db.py` scripts run the whole retrieval stage once per
configuration and write the passages into a `context` column, so the training scripts only ever read a pre-built
column. This makes every training run cheap, deterministic, and directly comparable across retrieval settings.

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

To ground the model, we constructed a **knowledge base of approximately 15GB of political and contextual documents**.
The
corpus combines official government sources with curated background information, organized into seven sections (the
research papers inside *Definitions* are indexed as their own `papers` category, so the vector store carries eight).

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

Sections are built independently by [`src/create_knowledge_base.py`](src/create_knowledge_base.py) into
`knowledge_base/<section>/`, then indexed into a per-embedder Chroma store by
[`src/create_persistent_db.py`](src/create_persistent_db.py). To inspect what a query retrieves without running the
full pipeline, use [`src/perform_retrieval.py`](src/perform_retrieval.py):

```bash
python src/perform_retrieval.py \
  --query "social security reform" \
  --retriever sentence-transformers/all-mpnet-base-v2 \
  --search-type similarity --top-k 5
```

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

The `id`, `source`, `url`, `year` and `fallacy_label` fields become chunk metadata in the vector store, which is what
makes temporal filtering (`--temporal-rag`) and per-section indexing possible.

## 🗂️ Datasets

| Dataset           | Tasks                              | Role                                                                          |
|-------------------|------------------------------------|-------------------------------------------------------------------------------|
| **ElecDeb60to20** | detection, classification          | Main dataset (U.S. political debates, 1960–2020).                             |
| **propaganda**    | detection, relation-classification | Second detection dataset; relation predictions come from the LIARArg model.   |
| **LIARArg**       | relation-classification            | Training data for the relation classifier (`support / attack / no_relation`). |
| **DEBAGREEMENT**  | relation-classification            | Training data for the agreement/relation classifier.                          |

Each dataset lives under `datasets/<name>-dataset/` with raw `train.csv` / `dev.csv` / `test.csv` splits and an
`original/` folder holding the source files. RAG-enriched variants are written into per-task, per-config subfolders by
`create_db.py`.

> **Note.** The DEBAGREEMENT corpus is supported by the relation-classification scripts (which auto-detect its
> `support / attack / neutral` label space) but is **not shipped in this repository** — add it under
> `datasets/debagreeement-dataset/relation-classification/` to run those sweeps.

> **Git LFS.** Every CSV under `datasets/` is tracked with Git LFS (see [`.gitattributes`](.gitattributes)). Run
> `git lfs install && git lfs pull` after cloning, otherwise the splits will be pointer files rather than data.

### Configuration Folder Names

There is one folder per retrieval configuration, and its name *is* the configuration. Both the enriched datasets and
the results mirror this naming, so a results folder can always be traced back to the data that produced it:

```
temporal=true-retriever=ddore14-sentence-roosebert-search=hybrid-rerank=cross-encoder-ms-marco-minilm-l-6-v2-query=relations_labeled-summarizer=false
│             │                                  │             │                                          │                              └── passage summarization model, or false
│             │                                  │             │                                          └── query mode: fallacy_relations | relations_labeled (omitted in sentence mode)
│             │                                  │             └── cross-encoder reranker, or false
│             │                                  └── search strategy: similarity | mmr | bm25 | hybrid | similarity_score_threshold
│             └── dense embedding model, bm25, or false for no retrieval
└── temporal filter: only retrieve documents predating the debate
```

The no-RAG baseline is therefore `temporal=false-retriever=false-summarizer=false`.

## ⚙️ Setup

Create the conda environment (named `rags`):

```bash
conda env create -f environment.yaml
conda activate rags
```

A few things are needed on top of the environment file:

| Requirement                               | Why                                                                                    |
|-------------------------------------------|----------------------------------------------------------------------------------------|
| `git lfs`                                 | The dataset CSVs are LFS objects — `git lfs pull` after cloning.                       |
| `python -m spacy download en_core_web_sm` | Sentence splitting inside `create_db.py`.                                              |
| `poppler`, `tesseract`                    | OCR of scanned PDFs and newspaper scans while building the knowledge base.             |
| `HF_TOKEN`                                | Gated Hub models (Llama, Mistral). Or pass `--token`.                                  |
| `OPENAI_API_KEY`                          | Only for the LLM scripts run with `--backend openai`.                                  |
| `CONTACT_EMAIL`                           | Sent in the OpenAlex User-Agent when collecting fallacy papers for the knowledge base. |

The [`.project-root`](.project-root) marker file is what `rootutils` uses to resolve paths — **do not delete it**, and
run every script from the repository root so relative paths like `datasets/…` resolve correctly.

## ▶️ Running the Pipeline

The `script/` folder is numbered to reflect the pipeline order. Each `.sh` is an sbatch script, but nothing prevents
running them directly with `bash` on a single machine.

| Stage | Script                                                               | What it does                                                                               |
|-------|----------------------------------------------------------------------|--------------------------------------------------------------------------------------------|
| 00    | `00-create_knowledge_base.sh`                                        | Builds every knowledge-base section (`src/create_knowledge_base.py`).                      |
| 01    | `01-create_persistent_db.sh`                                         | Indexes the knowledge base into a persistent Chroma store, once per embedding model.       |
| 02    | `02-create_database.sh`, `02-create_database-propaganda.sh`          | Runs each dataset's `create_db.py` over the retrieval grid to produce RAG-enriched splits. |
| 03    | `03-fallacy_no_rag.sh`, `03-fallacy_no_rag_llm.sh`                   | No-RAG training sweep, encoder and LLM tracks.                                             |
| 04    | `04-fallacy_rag.sh`, `04-fallacy_rag_llm.sh`                         | RAG training sweep over the configs listed in the script's `DATASETS` array.               |
| 05    | `05-relation_classification.sh`, `05-relation_classification_llm.sh` | Relation-classification sweep on LIARArg / DEBAGREEMENT.                                   |
| 06    | `06-best_hparams_*.sh`                                               | **Generated**, not written by hand — see below.                                            |

Stages 00–01 only need to run once. Stage 02 is the expensive one: it performs all retrieval up front, so stages 03–05
never touch the vector store.

After a single-seed sweep, [`script/generate_best_hparam_script.py`](script/generate_best_hparam_script.py) picks the
best hyperparameter cell per model and emits a `06-best_hparams_*.sh` that re-runs only those, once per seed:

```bash
python script/generate_best_hparam_script.py \
  output/one-run/elecdeb60to20-detection/temporal=false-retriever=false-summarizer=false \
  --seeds 14 42 55 89 3
```

This writes `script/06-best_hparams_<data>_<task>_<config>.sh` for the encoders and a matching `..._llm_...` script for
the LLMs, whose results land under `output/five-runs/`.

### Switching Dataset or Task

The `03-*.sh` / `04-*.sh` scripts default to ElecDeb60to20 detection. Both are set by two variables near the top:

```bash
data="elecdeb60to20"  # elecdeb60to20 | propaganda
task=detection        # detection | classification
```

These feed the input paths (`datasets/$data-dataset/$task/…`), the script actually invoked
(`src/$task/run_$task.py`), and the output paths (`output/one-run/$data-$task/…`). The `04-*.sh` scripts additionally
take a `DATASETS=()` array listing which configuration folders to sweep.

### Experiment Grid

| Track    | Models                                                                                                                      | Grid                                                                                 |
|----------|-----------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------|
| Encoders | DeBERTa-v3, Longformer, ModernBERT, RoBERTa, LegalBERT, NeoBERT, and the four RooseBERT variants (cont/scr × cased/uncased) | batch ∈ {8, 16} × lr ∈ {2e-5, 3e-5, 5e-5} × epochs ∈ {3, 5} — 12 cells per model     |
| LLMs     | Llama-3.1-8B-Instruct, Qwen3.5-9B-Base, gpt-oss-20b, Mistral-7B-Instruct-v0.3                                               | zero-shot, few-shot, and fine-tuning at lr ∈ {1e-4, 3e-4, 5e-4} × epochs ∈ {1, 2, 3} |

Runs are named after their cell (`bs16_lr3e-5_ep5`, `mode-few_shot`, `mode-fine_tune_bs8_lr3e-4_ep2`), which is how the
aggregation scripts recover the hyperparameters.

## 📊 Results & Analysis

| Script                                                               | Purpose                                                                                                                     |
|----------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------|
| [`script/check_one_run.py`](script/check_one_run.py)                 | Verifies the single-seed grid is complete; reports gaps and exits non-zero.                                                 |
| [`script/check_five_runs.py`](script/check_five_runs.py)             | Same check for the multi-seed grid (all models × 5 seeds).                                                                  |
| [`script/extract_results.py`](script/extract_results.py)             | Aggregates `output/one-run/` into a `results.csv` per config plus a combined `all_results.csv`.                             |
| [`script/extract_results_5runs.py`](script/extract_results_5runs.py) | Aggregates `output/five-runs/` into `results_mean_std.csv` (mean, std, seed count) plus `all_results_mean_std.csv`.         |
| [`script/ablation.py`](script/ablation.py)                           | Leave-one-out ablation of the RAG components, with hyperparameters frozen at the anchor cell so deltas stay selection-free. |
| [`script/ttest_best_models.py`](script/ttest_best_models.py)         | Paired significance tests (t-test, Wilcoxon, permutation, Hedges' g) between the baseline and each config's best model.     |

A run is considered complete when it carries its **completion marker**: `predict_results.json` for encoders,
`predict_results_metrics.json` for LLMs. Every aggregation and check script keys off these two filenames.

The `notebooks/` folder holds the qualitative analyses: `error_analysis.ipynb` (confusion matrices comparing the
baseline against the best RAG config, exported to the `cm_*.pdf` files), `knowledge_base_analysis.ipynb` (KB
composition and document-length statistics), `topic_analysis.ipynb` and `topic_similarity_search_kb.ipynb` (debate
topic extraction and how well the knowledge base covers those topics).

## 📁 Project Structure

```
src/
  create_knowledge_base.py          # build the KB (bills, people, geography, history, papers, exemplars…)
  create_persistent_db.py           # index the KB into a persistent Chroma vector store
  perform_retrieval.py              # CLI to query the vector store (dense / BM25)
  detection/                        # binary fallacy detection (encoder + LLM) — see its README
  classification/                   # fallacy-type classification (encoder + LLM) — see its README
  relation-extraction/              # relation classification + relation prediction — see its README
  utils/
    fallacy_loader.py               # LangChain loader turning KB JSON records into Documents
    dataset_prompts.py              # dataset-aware system prompts and label spaces for the LLM scripts
    util.py                         # HTTP, Wikipedia and OCR helpers used by the KB builder

datasets/                           # source data + per-dataset RAG-enrichment scripts
  elecdeb60to20-dataset/            # main dataset
    original/                       # source files
    train.csv, dev.csv, test.csv    # raw splits
    detection/create_db.py          # builds RAG-enriched detection datasets
    classification/create_db.py     # builds RAG-enriched classification datasets
    <task>/<config>/                # generated RAG-enriched splits (one folder per config)
  propaganda-dataset/               # detection + relation-classification data
  LIARArg-dataset/                  # relation-classification training data

script/                             # sbatch / driver scripts (numbered by pipeline stage)
  00-create_knowledge_base.sh
  01-create_persistent_db.sh
  02-create_database*.sh            # runs create_db.py to build RAG-enriched datasets
  03-fallacy_no_rag*.sh             # no-RAG training sweep (encoder / LLM)
  04-fallacy_rag*.sh                # RAG training sweep (encoder / LLM)
  05-relation_classification*.sh    # relation-classification sweep (encoder / LLM)
  generate_best_hparam_script.py    # generates a 06-best_hparams_*.sh from a finished sweep
  extract_results.py                # aggregates output/one-run into per-config CSVs
  extract_results_5runs.py          # aggregates output/five-runs into per-config CSVs
  check_one_run.py, check_five_runs.py   # verify grid completeness
  ablation.py                       # leave-one-out ablation of the RAG components
  ttest_best_models.py              # paired significance tests across configs

notebooks/                          # topic / knowledge-base / error analysis
docs/pipeline.png                   # the pipeline figure shown above
misc/                               # auxiliary source files (congress roster, UDHR, source URLs)
output/
  one-run/<dataset>-<task>/<config>/<model>/<run>/            # single-seed grid-search results
  five-runs/<dataset>-<task>/<config>/<model>/seed_<n>/<run>/ # multi-seed best-hparam results
  trained-models/                   # checkpoints reused by the relation-prediction scripts
```

Each `create_db.py` reads its raw splits from `--dataset-dir` (default `datasets/<name>-dataset`) and writes the
RAG-enriched splits back under `<dataset-dir>/<task>/<config>/{train,dev,test}.csv`, where `<config>` encodes the
retrieval settings as described in [Configuration Folder Names](#configuration-folder-names).

## 📄 License

Inria holds the ownership rights on the code and data in this repository. Use is granted **free of charge to academic
users for research purposes only**; any other use requires prior consent from Inria. See [`LICENSE`](LICENSE) for the
full terms.
