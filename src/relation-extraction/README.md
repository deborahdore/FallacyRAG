# Relation Classification

Classify the **argumentative relation** between a pair of components — a fallacy/claim sentence and a candidate
related sentence — into one of three classes:

```
support | attack | no_relation
```

This task supports the RAG pipeline. The predicted relations enrich the retrieval query used by the dataset builders
(`create_db.py`'s `relations` / `relations_labeled` query modes), so a fallacy is retrieved together with the
components that support or attack it.

The workflow has two phases:

1. **Train** an encoder relation classifier on a relation-labeled corpus (`run_relation_classification.py`).
2. **Predict** relations on an unlabeled corpus with a trained checkpoint (`predict_*.py`), writing the results back to
   CSV for downstream use.

| Script                                                               | Role                                                                                         |
|----------------------------------------------------------------------|----------------------------------------------------------------------------------------------|
| [`run_relation_classification.py`](run_relation_classification.py)   | Encoder (BERT-style) fine-tuning of the 3-class relation classifier.                         |
| [`predict_propaganda_relations.py`](predict_propaganda_relations.py) | Predict relations on the propaganda pairs with a LIARArg-trained model.                      |
| [`predict_agreement_relations.py`](predict_agreement_relations.py)   | Predict relations on the Reddit pairs with a DEBAGREEMENT-trained model.                     |
| [`utils/prepare_liarArg.py`](utils/prepare_liarArg.py)               | Build the LIARArg relation-classification train/dev/test splits.                             |
| [`utils/prepare_propaganda.py`](utils/prepare_propaganda.py)         | Build the propaganda **prediction** pairs (one fallacy × every other phrase in its article). |
| [`utils/prepare_reddit.py`](utils/prepare_reddit.py)                 | Build the Reddit **prediction** pairs (one fallacy × every other sentence in its comment).   |

---

## Input format

Each example is a **sentence pair** with a relation label. Training data uses `sentence1` / `sentence2`; the generated
prediction sets rename these to `fallacy` / `candidate_relation` and keep extra metadata columns for traceability.

| Column                             | Description                                              |
|------------------------------------|----------------------------------------------------------|
| `sentence1` / `fallacy`            | The first component (segment A) — the fallacy/claim.     |
| `sentence2` / `candidate_relation` | The second component (segment B) — the candidate.        |
| `relation_label`                   | `support`, `attack`, or `no_relation` (training only).   |
| `label` / `label_value`            | Numeric / string label pair mirroring the source scheme. |

Prediction sets carry `label = -1` (no gold relation) — they exist purely to run inference over.

---

## Encoder script — `run_relation_classification.py`

Fine-tunes a BERT-style encoder for 3-class (single-label) relation classification. Each pair is tokenised as:

```
[CLS] sentence1 [SEP] sentence2 [SEP]
```

The `[CLS]` representation is passed to a softmax head trained with **class-weighted cross-entropy** to offset the
imbalance between relation types. The label set is derived from the data at run time (sorted for determinism).

### Model

| Argument                    | Default                         | Description                                                  |
|-----------------------------|---------------------------------|--------------------------------------------------------------|
| `--model_name_or_path`      | `google-bert/bert-base-uncased` | Pretrained model identifier (Hub) or local path.             |
| `--config_name`             | `None`                          | Pretrained config name; defaults to `model_name_or_path`.    |
| `--tokenizer_name`          | `None`                          | Pretrained tokenizer name; defaults to `model_name_or_path`. |
| `--use_fast_tokenizer`      | `True`                          | Use a fast tokenizer (backed by the tokenizers library).     |
| `--cache_dir`               | `./cache/`                      | Directory for cached pretrained models.                      |
| `--trust_remote_code`       | `False`                         | Allow execution of remote code from Hub repos.               |
| `--ignore_mismatched_sizes` | `False`                         | Load a model whose head dimensions differ from checkpoint.   |

### Data

| Argument                    | Default          | Description                                                           |
|-----------------------------|------------------|-----------------------------------------------------------------------|
| `--train_file`              | *(see script)*   | Training pairs CSV/JSON.                                              |
| `--validation_file`         | *(see script)*   | Validation pairs CSV/JSON.                                            |
| `--test_file`               | *(see script)*   | Test / prediction pairs CSV/JSON.                                     |
| `--text_column_name`        | `sentence1`      | Column holding the first sentence of the pair.                        |
| `--speech_turn_column_name` | `sentence2`      | Column holding the second sentence of the pair.                       |
| `--label_column_name`       | `relation_label` | Column holding the relation label (`support`/`attack`/`no_relation`). |
| `--max_train_samples`       | `None`           | Truncate training set (for debugging).                                |
| `--shuffle_train_dataset`   | `False`          | Shuffle the training set before training.                             |

### Training & evaluation

Standard HuggingFace `TrainingArguments` (`--do_train`, `--do_eval`, `--do_predict`, `--output_dir`,
`--per_device_train_batch_size`, `--num_train_epochs`, `--learning_rate`, `--seed`, …). Metrics reported:
**accuracy**, **precision** (macro), **recall** (macro), **F1 macro**, plus **per-class F1** (`f1_<label>`).

Predictions are written to `predict_results.txt` (`index`, `prediction`, `gold`); metrics to the standard
`train`/`eval`/`predict` JSON files.

### Sweep driver — `run.sh`

`run.sh` is the sbatch driver: it sweeps the model list × batch sizes × learning rates × epochs over the datasets
listed in its `DATASETS` array (e.g. `change-my-view-dataset`), reading
`datasets/<dataset>/relation-classification/{train,dev,test}.csv` and writing to
`output/one-run/<dataset>-relation-classification/<model>/<run>/`.

### Example

```bash
python src/relation-extraction/run_relation_classification.py \
  --model_name_or_path microsoft/deberta-v3-base \
  --train_file datasets/LIARArg-dataset/relation-classification/train.csv \
  --validation_file datasets/LIARArg-dataset/relation-classification/dev.csv \
  --test_file datasets/LIARArg-dataset/relation-classification/test.csv \
  --label_column_name relation_label \
  --do_train --do_eval --do_predict \
  --num_train_epochs 3 --learning_rate 3e-5 \
  --per_device_train_batch_size 16 \
  --output_dir ./output/one-run/LIARArg-relation-classification/deberta-v3-base-bs16_lr3e-5_ep3
```

---

## Prediction scripts — `predict_propaganda_relations.py` / `predict_agreement_relations.py`

Pure inference: load a trained checkpoint, run it over a prediction-pairs CSV, and write back the predicted relation
plus per-class softmax probabilities alongside the original metadata columns.

The checkpoint's `config.id2label` is just `{0:"0", 1:"1", 2:"2"}`; the **semantic meaning** of each id comes from the
training corpus and is fixed in each script:

| Script                            | Trained on   | Predicts on | id → relation                                  |
|-----------------------------------|--------------|-------------|------------------------------------------------|
| `predict_propaganda_relations.py` | LIARArg      | propaganda  | `0 = attack`, `1 = support`, `2 = no_relation` |
| `predict_agreement_relations.py`  | DEBAGREEMENT | Reddit      | `0 = attack`, `1 = neutral`, `2 = support`     |

### Arguments

| Argument        | Default        | Description                                               |
|-----------------|----------------|-----------------------------------------------------------|
| `--model-dir`   | *(see script)* | Local fine-tuned checkpoint folder (loaded offline).      |
| `--data-file`   | *(see script)* | Prediction pairs CSV (from the `prepare_*.py` utilities). |
| `--output-file` | *(see script)* | Where to write the predictions CSV.                       |
| `--batch-size`  | `64`           | Inference batch size.                                     |
| `--max-length`  | `512`          | Max token sequence length.                                |
| `--device`      | `None`         | Force `cuda` / `mps` / `cpu` (auto-detected otherwise).   |

### Output columns

The input CSV is copied through, with these columns appended: `predicted_label` (id), `predicted_relation` (name),
`confidence` (max softmax probability), and one `prob_<relation>` column per class.

### Example

```bash
python src/relation-extraction/predict_propaganda_relations.py \
  --model-dir output/trained-models/LIARArg-deberta-v3-base-bs16_lr3e-5_ep3 \
  --data-file datasets/propaganda-dataset/relation-classification/dataset.csv \
  --output-file datasets/propaganda-dataset/relation-classification/predictions.csv \
  --batch-size 64
```

---

## Data preparation — `utils/prepare_*.py`

These build the pair datasets consumed above:

- **`prepare_liarArg.py`** — turns the LIARArg corpus into a labeled `support / attack / no_relation` train/dev/test
  set (partial support/attack are merged into support/attack, matching the paper's ablation).
- **`prepare_propaganda.py`** — builds a **prediction** set: every propaganda-labeled sentence paired with every other
  sentence of its own article. No gold relations (`label = -1`).
- **`prepare_reddit.py`** — builds a **prediction** set: every fallacy sentence paired with every other sentence of its
  own comment/turn. No gold relations (`label = -1`).
