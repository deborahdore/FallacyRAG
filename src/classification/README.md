# Fallacy Classification

Sentence-level classification of the **type** of fallacy in political debate transcripts. Unlike detection, this task
never sees non-fallacious sentences — `NoFallacy` sentences are dropped during dataset construction (see
[
`datasets/elecdeb60to20-dataset/classification/create_db.py`](../../datasets/elecdeb60to20-dataset/classification/create_db.py)) —
so every input is already known to
be fallacious and the model only has to decide *which* fallacy type(s) it exhibits.

The label inventory is **dataset-aware**: it is selected automatically from the training-file path (see
[`src/utils/dataset_prompts.py`](../utils/dataset_prompts.py)), and single- vs. multi-label is auto-detected from the
dataset columns. For **ElecDeb60to20** the task is single-label over six categories. Canonical labels (exactly as they
appear in the `fallacy_label` column) and their human-readable forms:

| Canonical label     | Human-readable      |
|---------------------|---------------------|
| `AdHominem`         | Ad Hominem          |
| `AppealtoAuthority` | Appeal to Authority |
| `AppealtoEmotion`   | Appeal to Emotion   |
| `FalseCause`        | False Cause         |
| `Slipperyslope`     | Slippery Slope      |
| `Slogans`           | Slogan              |

The **Reddit** dataset uses a different eight-class inventory plus an explicit "No Fallacy" option:

| Canonical label       | Human-readable          |
|-----------------------|-------------------------|
| `Authority`           | Appeal to Authority     |
| `Population`          | Appeal to Majority      |
| `Natural`             | Appeal to Nature        |
| `Tradition`           | Appeal to Tradition     |
| `WorseProblems`       | Appeal to Worse Problems |
| `Blackwhite`          | Black-or-White          |
| `HastyGeneralization` | Hasty Generalization    |
| `SlipperySlope`       | Slippery Slope          |

Two scripts are provided, covering encoder-based fine-tuning and LLM-based inference/fine-tuning:

| Script                                                   | Approach                                       | When to use                                                                                |
|----------------------------------------------------------|------------------------------------------------|--------------------------------------------------------------------------------------------|
| [`run_classification.py`](run_classification.py)         | Encoder (BERT-style) fine-tuning               | Fast, low-resource, reproducible baselines                                                 |
| [`run_classification_llm.py`](run_classification_llm.py) | LLM — zero-shot, few-shot, or LoRA fine-tuning | Larger models, no labelled data available (zero/few-shot), or pushing performance ceilings |

---

## Input format

Each example contains a short **sentence** extracted from a debate turn, the full **speech turn** for context, an
optional **retrieved context** field (when `--use_rag` is set), and a fallacy-type label.

| Column          | Description                                                   |
|-----------------|---------------------------------------------------------------|
| `sentence`      | The sentence to classify                                      |
| `speech_turn`   | The full debate turn the sentence was extracted from          |
| `context`       | Pre-built RAG context (only present in `rag/` split variants) |
| `fallacy_label` | One of the six canonical fallacy types (see the table above)  |

---

## Encoder script — `run_classification.py`

Fine-tunes a BERT-style encoder for multi-class classification. Each example is tokenised as:

```
[CLS] sentence [SEP] speech_turn [SEP]                              (without RAG)
[CLS] sentence [SEP] speech_turn context_1 … context_n [SEP]        (with RAG)
```

The `[CLS]` representation is passed to a linear head sized to the number of fallacy types. The label set is derived
from the data at run time. Single- vs. multi-label is auto-detected from the columns: single-label uses a softmax head
with **class-weighted cross-entropy**, while multi-label (several 0/1 indicator columns) uses a sigmoid head with
**pos_weight `BCEWithLogitsLoss`** — both offset the imbalance between fallacy types.

### Model

| Argument                    | Default                         | Description                                                  |
|-----------------------------|---------------------------------|--------------------------------------------------------------|
| `--model_name_or_path`      | `google-bert/bert-base-uncased` | Pretrained model identifier (Hub) or local path.             |
| `--config_name`             | `None`                          | Pretrained config name; defaults to `model_name_or_path`.    |
| `--tokenizer_name`          | `None`                          | Pretrained tokenizer name; defaults to `model_name_or_path`. |
| `--use_fast_tokenizer`      | `True`                          | Use a fast tokenizer (backed by the tokenizers library).     |
| `--cache_dir`               | `./cache/`                      | Directory for cached pretrained models.                      |
| `--model_revision`          | `main`                          | Branch / tag / commit id to use.                             |
| `--token`                   | `None`                          | HuggingFace auth token for private models.                   |
| `--trust_remote_code`       | `False`                         | Allow execution of remote code from Hub repos.               |
| `--ignore_mismatched_sizes` | `False`                         | Load a model whose head dimensions differ from checkpoint.   |

### Data

| Argument                      | Default                                                     | Description                                                                                                          |
|-------------------------------|-------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------|
| `--dataset_name`              | `None`                                                      | HuggingFace dataset name (via `datasets` library).                                                                   |
| `--dataset_config_name`       | `None`                                                      | Configuration name for the dataset.                                                                                  |
| `--train_file`                | `./datasets/elecdeb60to20-dataset/classification/train.csv` | Training data file (CSV or JSON). Point at the classification split; use the `rag/` variant when `--use_rag` is set. |
| `--validation_file`           | `./datasets/elecdeb60to20-dataset/classification/dev.csv`   | Validation data file (CSV or JSON). Use the `rag/` variant when `--use_rag` is set.                                  |
| `--test_file`                 | `./datasets/elecdeb60to20-dataset/classification/test.csv`  | Test / prediction data file (CSV or JSON). Use the `rag/` variant when `--use_rag` is set.                           |
| `--text_column_name`          | `sentence`                                                  | Column containing the input sentence text.                                                                           |
| `--speech_turn_column_name`   | `speech_turn`                                               | Column containing the full debate turn.                                                                              |
| `--label_column_name`         | `fallacy_label`                                             | Column containing the fallacy-type label.                                                                            |
| `--context_column_name`       | `None`                                                      | Column containing the RAG context text (used when `--use_rag` is set).                                               |
| `--overwrite_cache`           | `True`                                                      | Overwrite cached tokenised datasets.                                                                                 |
| `--preprocessing_num_workers` | `None`                                                      | Number of processes for `dataset.map()`.                                                                             |
| `--max_seq_length`            | `512`                                                       | Maximum total input length after tokenisation.                                                                       |
| `--pad_to_max_length`         | `False`                                                     | Pad all samples to `max_seq_length` (needed for TPU; otherwise dynamic padding is used).                             |
| `--max_train_samples`         | `None`                                                      | Truncate training set (for debugging).                                                                               |
| `--max_eval_samples`          | `None`                                                      | Truncate evaluation set (for debugging).                                                                             |
| `--max_predict_samples`       | `None`                                                      | Truncate prediction set (for debugging).                                                                             |
| `--shuffle_train_dataset`     | `False`                                                     | Shuffle the training dataset before training.                                                                        |
| `--shuffle_seed`              | `42`                                                        | Random seed used when `--shuffle_train_dataset` is set.                                                              |
| `--use_rag`                   | `False`                                                     | Append pre-built retrieved context to the BERT input.                                                                |

> When `--use_rag` is set and the default no-RAG file paths are still in place, the script automatically redirects
> `/no_rag/` paths to the corresponding `/rag/` subdirectory.

### RAG (Retrieval-Augmented Generation)

Retrieval is **not** performed by `run_classification.py`. The knowledge-base context is retrieved ahead of time by
[
`datasets/elecdeb60to20-dataset/classification/create_db.py`](../../datasets/elecdeb60to20-dataset/classification/create_db.py),
which writes a pre-built `context`
column into the dataset (the `rag/` split variants). At training time, `run_classification.py` simply reads that column
and appends it to `speech_turn` as the **second sequence** passed to the tokenizer (speech turn first, so truncation
trims the retrieved context rather than the local debate context).

`run_classification.py` therefore exposes only two RAG-related arguments:

| Argument                | Default | Description                                                                     |
|-------------------------|---------|---------------------------------------------------------------------------------|
| `--use_rag`             | `False` | Read the pre-built `context` field from the dataset and append it to the input. |
| `--context_column_name` | `None`  | Name of the column holding the retrieved context (typically `context`).         |

#### Building the RAG datasets — `datasets/elecdeb60to20-dataset/classification/create_db.py`

A `create_db.py` with the same interface (and dataset-specific defaults) exists per dataset; the ElecDeb60to20 one is
the canonical reference. The Reddit RAG datasets are built the same way via the `02-create_database-reddit.sh` driver
script.

The retrieval strategy is chosen when the datasets are built. The script sentence-splits the speech turns, drops
`NoFallacy` sentences, re-splits into train/dev/test, retrieves knowledge-base passages for each sentence, and writes
them into the `context` column. Key arguments:

| Argument                | Default                                | Description                                                                                                                                                                        |
|-------------------------|----------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `--mode`                | `sentence`                             | `sentence` — query with the bare sentence; `relations` — query with the fallacy text plus its related components.                                                                  |
| `--query-mode`          | `fallacy_relations`                    | (`relations` mode only) `fallacy_relations` — concatenate fallacy + relation text; `relations_labeled` — one `<relation> supports/attacks <fallacy>` clause per relation.          |
| `--retriever`           | `None`                                 | Dense embedding model: `sentence-transformers/all-mpnet-base-v2`, `ddore14/Sentence-RooseBERT`, `BAAI/bge-large-en-v1.5`, or `bm25`. Omit for a no-RAG dataset.                    |
| `--search-type`         | `similarity`                           | Retrieval strategy: `similarity`, `mmr`, `similarity_score_threshold`, `bm25`, `hybrid`.                                                                                           |
| `--persist-directory`   | `./chromadb`                           | ChromaDB persistence directory for dense search.                                                                                                                                   |
| `--chunk-size`          | `1000`                                 | Document chunk size for the BM25 leg.                                                                                                                                              |
| `--chunk-overlap`       | `200`                                  | Chunk overlap for the BM25 leg.                                                                                                                                                    |
| `--top-k`               | `3`                                    | Number of documents to keep per example.                                                                                                                                           |
| `--fetch-k`             | `20`                                   | Candidate pool size fetched before MMR / reranking narrows it to `top-k`.                                                                                                          |
| `--lambda-mult`         | `0.5`                                  | MMR relevance/diversity trade-off (`mmr` search only).                                                                                                                             |
| `--score-threshold`     | `0.5`                                  | Minimum similarity score (`similarity_score_threshold` search only).                                                                                                               |
| `--rrf-k`               | `60`                                   | Reciprocal Rank Fusion constant (`hybrid` search only).                                                                                                                            |
| `--dense-weight`        | `1.0`                                  | Weight of the dense leg in hybrid fusion.                                                                                                                                          |
| `--bm25-weight`         | `1.0`                                  | Weight of the BM25 leg in hybrid fusion.                                                                                                                                           |
| `--rerank`              | `False`                                | Rerank the candidate pool with a local cross-encoder before keeping `top-k`.                                                                                                       |
| `--reranker-model`      | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder used for reranking.                                                                                                                                                  |
| `--summarization-model` | `None`                                 | Optional LLM to summarise retrieved passages before writing them. Supported: `meta-llama/Llama-3.1-8B-Instruct`, `Qwen/Qwen3.5-9B-Base`. If `None`, passages are written verbatim. |
| `--temporal-rag`        | `False`                                | Only retrieve knowledge-base documents whose year precedes the debate year.                                                                                                        |

**Retrieval strategies at a glance:**

- **`similarity`** — standard cosine / dot-product search against the ChromaDB vector store.
- **`mmr`** — Maximal Marginal Relevance; balances relevance and diversity in the retrieved set.
- **`similarity_score_threshold`** — only returns documents above a minimum similarity score.
- **`bm25`** — sparse keyword-based retrieval; no embeddings or vector store required.
- **`hybrid`** — dense + BM25 fused with Reciprocal Rank Fusion.

### Training

| Argument                        | Default  | Description                                                                                                                                           |
|---------------------------------|----------|-------------------------------------------------------------------------------------------------------------------------------------------------------|
| `--do_train`                    | `False`  | Run training.                                                                                                                                         |
| `--output_dir`                  | `None`   | Directory for model predictions and checkpoints. **Required.**                                                                                        |
| `--per_device_train_batch_size` | `8`      | Batch size per device for training.                                                                                                                   |
| `--num_train_epochs`            | `3.0`    | Total training epochs.                                                                                                                                |
| `--max_steps`                   | `-1`     | If > 0, overrides `num_train_epochs`.                                                                                                                 |
| `--learning_rate`               | `5e-5`   | Initial learning rate for the optimizer.                                                                                                              |
| `--lr_scheduler_type`           | `linear` | Scheduler type: `linear`, `cosine`, `cosine_with_restarts`, `polynomial`, `constant`, `constant_with_warmup`, `inverse_sqrt`, `reduce_lr_on_plateau`. |
| `--warmup_steps`                | `0`      | Linear warmup steps (integer) or fraction of total steps (float in `[0, 1)`).                                                                         |
| `--weight_decay`                | `0.0`    | Weight decay coefficient.                                                                                                                             |
| `--gradient_accumulation_steps` | `1`      | Effective batch = `per_device_train_batch_size × devices × this`.                                                                                     |
| `--max_grad_norm`               | `1.0`    | Gradient clipping max norm. Set to `0` to disable.                                                                                                    |
| `--seed`                        | `42`     | Random seed.                                                                                                                                          |
| `--resume_from_checkpoint`      | `None`   | Path to a checkpoint folder to resume from.                                                                                                           |

### Evaluation

Metrics reported: **accuracy**, **precision** (macro), **recall** (macro), **F1 macro**, plus **per-class F1**
(`f1_<label>`) for error analysis.

| Argument                       | Default | Description                                                                      |
|--------------------------------|---------|----------------------------------------------------------------------------------|
| `--do_eval`                    | `False` | Run evaluation on the validation split.                                          |
| `--do_predict`                 | `False` | Run predictions on the test split and write `predict_results.txt`.               |
| `--eval_strategy`              | `no`    | When to evaluate during training: `no`, `steps`, `epoch`.                        |
| `--eval_steps`                 | `None`  | Steps between evaluations (when `eval_strategy=steps`).                          |
| `--per_device_eval_batch_size` | `8`     | Batch size per device for evaluation.                                            |
| `--load_best_model_at_end`     | `False` | Load the best checkpoint at end of training. Requires `eval_strategy` to be set. |
| `--metric_for_best_model`      | `None`  | Metric used with `load_best_model_at_end`.                                       |

### Example commands

```bash
# Fine-tune BERT (no RAG)
python src/classification/run_classification.py \
  --model_name_or_path google-bert/bert-base-uncased \
  --do_train --do_eval --do_predict \
  --num_train_epochs 5 \
  --learning_rate 2e-5 \
  --per_device_train_batch_size 16 \
  --output_dir ./results/bert_baseline

# Fine-tune DeBERTa with RAG context
python src/classification/run_classification.py \
  --model_name_or_path microsoft/deberta-v3-base \
  --use_rag --context_column_name context \
  --do_train --do_eval --do_predict \
  --output_dir ./results/deberta_rag
```

---

## LLM script — `run_classification_llm.py`

Runs fallacy-type classification with a causal language model using one of three modes. The **system prompt and label
inventory are selected automatically from the dataset** (ElecDeb60to20 vs. Reddit). Gold labels and predictions are
parsed as label *sets*, so both single-label datasets (ElecDeb60to20 — one type per sentence) and multi-label datasets
are handled transparently. Each example is formatted as a structured chat prompt:

```
[System]    You are an expert in argumentation analysis…
[User]      Sentence: "<sentence>"

            Debate turn context:
            <speech_turn>

            [Retrieved context: <context>]   (when --use_rag)

            Which fallacy types does this sentence contain?
            Answer with a comma-separated list drawn only from:
            Ad Hominem, Appeal to Authority, Appeal to Emotion,
            False Cause, Slippery Slope, Slogan.
[Assistant] Appeal to Emotion, Ad Hominem
```

### Post-processing

The LLM replies with a free-text, comma-separated list of human-readable fallacy names. After inference, each name is
parsed back into the canonical dataset labels via **fuzzy alias matching** before metrics and output files are written:

```
"Appeal to Emotion"  →  "AppealtoEmotion"
"Slippery Slope"     →  "Slipperyslope"
```

A dedicated token represents the empty label set (no fallacy type applies), so "none" answers are handled consistently.

### Modes

| Mode            | Flag               | Description                                                                                                                                                     |
|-----------------|--------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Zero-shot**   | `--mode zero_shot` | No examples in the prompt. The model relies entirely on its pre-trained knowledge.                                                                              |
| **Few-shot**    | `--mode few_shot`  | `--num_shots` balanced examples (per class) are prepended as prior conversation turns. Examples are sampled from the training split.                            |
| **Fine-tuning** | `--mode fine_tune` | LoRA / QLoRA supervised fine-tuning via TRL `SFTTrainer`, followed by evaluation / prediction on the fine-tuned adapter. Only supported with `--backend local`. |

### Backends

| Backend                 | Flag               | Requirements                                                                                                                                             |
|-------------------------|--------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Local (HuggingFace)** | `--backend local`  | Any causal LM accessible via `AutoModelForCausalLM`. Supports 4-bit and 8-bit quantization.                                                              |
| **OpenAI**              | `--backend openai` | `OPENAI_API_KEY` environment variable. Compatible with any OpenAI chat model (e.g. `gpt-4o-mini`, `gpt-4o`). Fine-tuning not supported via this backend. |

### Model

| Argument               | Default                            | Description                                                                                            |
|------------------------|------------------------------------|--------------------------------------------------------------------------------------------------------|
| `--model_name_or_path` | `meta-llama/Llama-3.1-8B-Instruct` | HuggingFace model identifier or local path (`local` backend), or OpenAI model name (`openai` backend). |
| `--cache_dir`          | `./cache/`                         | Directory for cached pretrained models.                                                                |
| `--model_revision`     | `main`                             | Branch / tag / commit id to use.                                                                       |
| `--token`              | `None`                             | HuggingFace auth token for private / gated models.                                                     |
| `--trust_remote_code`  | `False`                            | Allow execution of remote code from Hub repos.                                                         |

### Data

Same columns and defaults as the encoder script. Additional argument:

| Argument    | Default | Description                                                                         |
|-------------|---------|-------------------------------------------------------------------------------------|
| `--use_rag` | `False` | Append the pre-built `context` field to each prompt as a "Retrieved context" block. |

### LLM experiment arguments

| Argument                 | Default     | Description                                                                                                    |
|--------------------------|-------------|----------------------------------------------------------------------------------------------------------------|
| `--mode`                 | `zero_shot` | Experiment mode: `zero_shot`, `few_shot`, `fine_tune`.                                                         |
| `--backend`              | `local`     | Inference backend: `local` or `openai`.                                                                        |
| `--num_shots`            | `1`         | Examples **per class** for few-shot prompting.                                                                 |
| `--few_shot_seed`        | `42`        | Random seed for selecting few-shot examples from the training split.                                           |
| `--max_new_tokens`       | `32`        | Maximum new tokens to generate per example.                                                                    |
| `--temperature`          | `0.0`       | Sampling temperature. `0` = greedy decoding.                                                                   |
| `--inference_batch_size` | `8`         | Batch size for local inference (not used for `openai` backend).                                                |
| `--load_in_4bit`         | `False`     | Load model in 4-bit NF4 (QLoRA). Requires `bitsandbytes`.                                                      |
| `--load_in_8bit`         | `False`     | Load model in 8-bit. Requires `bitsandbytes`.                                                                  |
| `--lora_r`               | `16`        | LoRA rank (`fine_tune` mode only).                                                                             |
| `--lora_alpha`           | `32`        | LoRA scaling factor alpha (`fine_tune` mode only).                                                             |
| `--lora_dropout`         | `0.05`      | LoRA dropout (`fine_tune` mode only).                                                                          |
| `--lora_target_modules`  | `None`      | Comma-separated list of target modules for LoRA. If `None`, PEFT auto-detects suitable modules (`all-linear`). |
| `--merge_adapter`        | `False`     | After fine-tuning, merge LoRA weights into the base model and save the full model to `--output_dir/merged`.    |

### Training (fine_tune mode only)

Standard HuggingFace `TrainingArguments` are used. The most relevant:

| Argument                        | Default | Description                                                              |
|---------------------------------|---------|--------------------------------------------------------------------------|
| `--do_train`                    | `False` | Required for `fine_tune` mode.                                           |
| `--output_dir`                  | `None`  | Directory for adapter weights, checkpoints, and results. **Required.**   |
| `--num_train_epochs`            | `3.0`   | Total training epochs.                                                   |
| `--per_device_train_batch_size` | `8`     | Batch size per device for training.                                      |
| `--gradient_accumulation_steps` | `1`     | Gradient accumulation steps.                                             |
| `--learning_rate`               | `5e-5`  | Initial learning rate.                                                   |
| `--warmup_ratio`                | `0.0`   | Fraction of steps for linear warmup.                                     |
| `--bf16`                        | `False` | BF16 mixed precision (recommended for Ampere+ GPUs).                     |
| `--fp16`                        | `False` | FP16 mixed precision.                                                    |
| `--eval_strategy`               | `no`    | When to evaluate: `no`, `steps`, `epoch`.                                |
| `--save_strategy`               | `steps` | When to save checkpoints: `no`, `steps`, `epoch`.                        |
| `--load_best_model_at_end`      | `False` | Load best checkpoint after training. Requires `eval_strategy` to be set. |

### Evaluation

Same metrics as the encoder script: **accuracy**, **precision** (macro), **recall** (macro), **F1 macro**. Results are
written to `eval_results.json` and `predict_results_metrics.json`. Predictions are written to `predict_results.txt` (
`index`, `prediction`, `gold`, with labels parsed as sets).

### Example commands

```bash
# Zero-shot with GPT-4o-mini
python src/classification/run_classification_llm.py \
  --model_name_or_path gpt-4o-mini --backend openai \
  --mode zero_shot \
  --do_eval --do_predict \
  --output_dir ./results/zero_shot_gpt4o_mini

# Few-shot (3 per class) with a local Llama model
python src/classification/run_classification_llm.py \
  --model_name_or_path meta-llama/Llama-3.1-8B-Instruct --backend local \
  --mode few_shot --num_shots 3 \
  --do_eval --do_predict \
  --output_dir ./results/few_shot_llama_3

# Few-shot with RAG context
python src/classification/run_classification_llm.py \
  --model_name_or_path meta-llama/Llama-3.1-8B-Instruct --backend local \
  --mode few_shot --num_shots 2 \
  --use_rag --context_column_name context \
  --do_eval --do_predict \
  --output_dir ./results/few_shot_llama_rag

# Fine-tune with QLoRA (4-bit)
python src/classification/run_classification_llm.py \
  --model_name_or_path meta-llama/Llama-3.1-8B-Instruct --backend local \
  --mode fine_tune --load_in_4bit \
  --lora_r 16 --lora_alpha 32 \
  --do_train --do_eval --do_predict \
  --num_train_epochs 3 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --learning_rate 2e-4 \
  --bf16 \
  --output_dir ./results/finetune_llama_qlora

# Fine-tune and merge adapter into full model
python src/classification/run_classification_llm.py \
  --model_name_or_path meta-llama/Llama-3.1-8B-Instruct --backend local \
  --mode fine_tune --load_in_4bit --merge_adapter \
  --do_train --do_eval \
  --num_train_epochs 3 \
  --output_dir ./results/finetune_llama_merged
```

---

## Output files

Both scripts write the following to `--output_dir`:

| File                           | Description                                                    |
|--------------------------------|----------------------------------------------------------------|
| `predict_results.txt`          | Tab-separated file with columns `index`, `prediction`, `gold`. |
| `predict_results_metrics.json` | Accuracy, precision, recall, F1 macro on the test split.       |
| `eval_results.json`            | Same metrics on the validation split (LLM script).             |
| `train_results.json`           | Training metrics (encoder script / LLM fine-tuning).           |

The encoder script also saves model weights and HuggingFace model cards to `--output_dir`.
The LLM fine-tuning script saves LoRA adapter weights (and optionally the merged full model) to `--output_dir`.
