# Fallacy Detection

Fine-tune and evaluate a token-classification model (NER, POS, …) on BIO-tagged data using the
script [run_detection_bio_tags.py](run_detection_bio_tags.py). Optionally augments each input with
retrieved knowledge-base context via a **Retrieval-Augmented Generation (RAG)** pipeline.

## Useful pointers

### Model

| Argument                    | Default                         | Description                                                  |
|-----------------------------|---------------------------------|--------------------------------------------------------------|
| `--model_name_or_path`      | `google-bert/bert-base-uncased` | Pretrained model identifier (Hub) or local path.             |
| `--config_name`             | `None`                          | Pretrained config name; defaults to `model_name_or_path`.    |
| `--tokenizer_name`          | `None`                          | Pretrained tokenizer name; defaults to `model_name_or_path`. |
| `--cache_dir`               | `./cache/`                      | Directory for cached pretrained models.                      |
| `--model_revision`          | `main`                          | Branch / tag / commit id to use.                             |
| `--token`                   | `None`                          | HuggingFace auth token for private models.                   |
| `--trust_remote_code`       | `False`                         | Allow execution of remote code from Hub repos.               |
| `--ignore_mismatched_sizes` | `False`                         | Load a model whose head dimensions differ from checkpoint.   |

---

### Data

| Argument                                       | Default                                   | Description                                           |
|------------------------------------------------|-------------------------------------------|-------------------------------------------------------|
| `--task_name`                                  | `ner`                                     | Task name (`ner`, `pos`, …).                          |
| `--dataset_name`                               | `None`                                    | HuggingFace dataset name (via `datasets` library).    |
| `--dataset_config_name`                        | `None`                                    | Configuration name for the dataset.                   |
| `--train_file`                                 | `./dataset/detection/bio_tags/train.json` | Training data file (CSV or JSON).                     |
| `--validation_file`                            | `./dataset/detection/bio_tags/dev.json`   | Validation data file (CSV or JSON).                   |
| `--test_file`                                  | `./dataset/detection/bio_tags/test.json`  | Test / prediction data file (CSV or JSON).            |
| `--text_column_name`                           | `tokens`                                  | Column containing the pre-tokenised words.            |
| `--label_column_name`                          | `ner_tags`                                | Column containing BIO label sequences.                |
| `--overwrite_cache`                            | `False`                                   | Overwrite cached tokenised datasets.                  |
| `--preprocessing_num_workers`                  | `None`                                    | Number of processes for `dataset.map()`.              |
| `--max_seq_length`                             | `512`                                     | Maximum total input length after tokenisation.        |
| `--pad_to_max_length`                          | `False`                                   | Pad all samples to `max_seq_length` (needed for TPU). |
| `--max_train_samples`                          | `None`                                    | Truncate training set (for debugging).                |
| `--max_eval_samples`                           | `None`                                    | Truncate evaluation set (for debugging).              |
| `--max_predict_samples`                        | `None`                                    | Truncate prediction set (for debugging).              |
| `--label_all_tokens` / `--no_label_all_tokens` | `True`                                    | Propagate word-level labels to all sub-word tokens.   |

---

### RAG (Retrieval-Augmented Generation)

These options control whether and how a knowledge base is used to inject additional context into each BERT input before
classification.

| Argument                   | Default                                   | Description                                                                                                                                                                                                                                           |
|----------------------------|-------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `--use_rag`                | `False`                                   | Enable RAG: augments BERT input with retrieved knowledge-base context.                                                                                                                                                                                |
| `--search_type`            | `similarity`                              | Retrieval strategy. One of: `similarity`, `mmr`, `similarity_score_threshold`, `bm25`.                                                                                                                                                                |
| `--embeddings_model`       | `sentence-transformers/all-mpnet-base-v2` | Encoder used to build dense retrieval embeddings.                                                                                                                                                                                                     |
| `--persist_directory_path` | `./chromadb`                              | ChromaDB persistence directory for dense search.                                                                                                                                                                                                      |
| `--chunk_size`             | `800`                                     | Document chunk size for the BM25 retriever.                                                                                                                                                                                                           |
| `--chunk_overlap`          | `80`                                      | Chunk overlap for the BM25 retriever.                                                                                                                                                                                                                 |
| `--top_k`                  | `3`                                       | Number of documents to retrieve per example.                                                                                                                                                                                                          |
| `--summarization_model`    | `None`                                    | LLM used to summarise retrieved passages before appending to BERT input. Supported: `meta-llama/Llama-3.1-8B-Instruct`, `mistralai/Mistral-7B-Instruct-v0.3`, `gpt-4o-mini`. If `None`, passages are appended verbatim (truncated to fit the budget). |
| `--context_budget`         | `128`                                     | Maximum tokens reserved for retrieved context per example.                                                                                                                                                                                            |

**Retrieval modes at a glance:**

- **`similarity`** – standard cosine / dot-product search against the ChromaDB vector store.
- **`mmr`** – Maximal Marginal Relevance; balances relevance and diversity in the retrieved set.
- **`similarity_score_threshold`** – only returns documents above a minimum similarity score.
- **`bm25`** – sparse keyword-based retrieval; uses `chunk_size` / `chunk_overlap` to split the corpus.

---

### Training

| Argument                        | Default             | Description                                                                                                                                                                                                                              |
|---------------------------------|---------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `--do_train`                    | `False`             | Run training.                                                                                                                                                                                                                            |
| `--output_dir`                  | `None`              | Directory for model predictions and checkpoints. **Required.**                                                                                                                                                                           |
| `--per_device_train_batch_size` | `8`                 | Batch size per device for training.                                                                                                                                                                                                      |
| `--num_train_epochs`            | `3.0`               | Total training epochs.                                                                                                                                                                                                                   |
| `--max_steps`                   | `-1`                | If > 0, overrides `num_train_epochs`.                                                                                                                                                                                                    |
| `--learning_rate`               | `5e-5`              | Initial learning rate for the optimizer.                                                                                                                                                                                                 |
| `--lr_scheduler_type`           | `linear`            | Scheduler type. Options: `linear`, `cosine`, `cosine_with_restarts`, `polynomial`, `constant`, `constant_with_warmup`, `inverse_sqrt`, `reduce_lr_on_plateau`, `cosine_with_min_lr`, `cosine_warmup_with_min_lr`, `warmup_stable_decay`. |
| `--lr_scheduler_kwargs`         | `None`              | Extra arguments for the scheduler.                                                                                                                                                                                                       |
| `--warmup_steps`                | `0`                 | Linear warmup steps (integer) or fraction of total steps (float in `[0, 1)`).                                                                                                                                                            |
| `--optim`                       | `adamw_torch_fused` | Optimizer. See `OptimizerNames` for the full list.                                                                                                                                                                                       |
| `--optim_args`                  | `None`              | Optional arguments for `AnyPrecisionAdamW`, `AdEMAMix`, `GaLore`, etc.                                                                                                                                                                   |
| `--weight_decay`                | `0.0`               | Weight decay coefficient (auto-excluded from bias and LayerNorm).                                                                                                                                                                        |
| `--adam_beta1`                  | `0.9`               | Exponential decay rate for Adam first moments.                                                                                                                                                                                           |
| `--adam_beta2`                  | `0.999`             | Exponential decay rate for Adam second moments.                                                                                                                                                                                          |
| `--adam_epsilon`                | `1e-8`              | Epsilon for numerical stability in Adam-based optimizers.                                                                                                                                                                                |
| `--optim_target_modules`        | `None`              | Target modules for GaLore / APOLLO.                                                                                                                                                                                                      |
| `--gradient_accumulation_steps` | `1`                 | Gradient accumulation steps. Effective batch = `per_device_train_batch_size × devices × this`.                                                                                                                                           |
| `--max_grad_norm`               | `1.0`               | Gradient clipping max norm. Set to `0` to disable.                                                                                                                                                                                       |
| `--label_smoothing_factor`      | `0.0`               | Label smoothing. `0` = no smoothing.                                                                                                                                                                                                     |
| `--resume_from_checkpoint`      | `None`              | Path to a checkpoint folder to resume from.                                                                                                                                                                                              |
| `--seed`                        | `42`                | Random seed.                                                                                                                                                                                                                             |
| `--data_seed`                   | `None`              | Seed for data samplers.                                                                                                                                                                                                                  |

---

### Mixed precision & compilation

| Argument                          | Default | Description                                                           |
|-----------------------------------|---------|-----------------------------------------------------------------------|
| `--bf16`                          | `False` | BF16 mixed precision training (preferred over FP16 for stability).    |
| `--fp16`                          | `False` | FP16 mixed precision training.                                        |
| `--bf16_full_eval`                | `False` | Full BF16 precision for evaluation.                                   |
| `--fp16_full_eval`                | `False` | Full FP16 precision for evaluation.                                   |
| `--tf32`                          | `None`  | TF32 mode on Ampere+ GPUs (up to 8× speedup).                         |
| `--gradient_checkpointing`        | `False` | Trade compute for memory (~20% slower training).                      |
| `--gradient_checkpointing_kwargs` | `None`  | Keyword arguments for `gradient_checkpointing_enable()`.              |
| `--torch_compile`                 | `False` | Compile model with `torch.compile()` for faster training.             |
| `--torch_compile_backend`         | `None`  | Backend for `torch.compile()`. Setting this auto-enables compilation. |
| `--torch_compile_mode`            | `None`  | Mode for `torch.compile()`. Setting this auto-enables compilation.    |
| `--use_liger_kernel`              | `False` | Enable Liger Kernel (~20% throughput, ~60% memory reduction).         |
| `--liger_kernel_config`           | `None`  | Liger Kernel configuration kwargs.                                    |

---

### Evaluation

| Argument                                                   | Default | Description                                                                      |
|------------------------------------------------------------|---------|----------------------------------------------------------------------------------|
| `--do_eval`                                                | `False` | Run evaluation.                                                                  |
| `--eval_strategy`                                          | `no`    | When to evaluate: `no`, `steps`, `epoch`.                                        |
| `--eval_steps`                                             | `None`  | Steps between evaluations (when `eval_strategy=steps`).                          |
| `--eval_delay`                                             | `0`     | Epochs / steps to wait before first evaluation.                                  |
| `--per_device_eval_batch_size`                             | `8`     | Batch size per device for evaluation.                                            |
| `--eval_on_start`                                          | `False` | Run a full evaluation pass before training starts (sanity check).                |
| `--eval_accumulation_steps`                                | `None`  | Accumulate prediction tensors for N steps before moving to CPU.                  |
| `--prediction_loss_only`                                   | `False` | Return only the loss during evaluation.                                          |
| `--eval_do_concat_batches` / `--no_eval_do_concat_batches` | `True`  | Concatenate batches during evaluation.                                           |
| `--eval_use_gather_object`                                 | `False` | Use recursive object gathering across devices.                                   |
| `--batch_eval_metrics`                                     | `False` | Compute eval metrics batch-by-batch to save memory.                              |
| `--include_for_metrics`                                    | `[]`    | Include `inputs` and/or `loss` in `compute_metrics`.                             |
| `--metric_for_best_model`                                  | `None`  | Metric used with `load_best_model_at_end`.                                       |
| `--greater_is_better`                                      | `None`  | Whether higher metric values are better.                                         |
| `--load_best_model_at_end`                                 | `False` | Load the best checkpoint at end of training. Requires `eval_strategy` to be set. |
| `--do_predict`                                             | `False` | Run predictions on the test set.                                                 |

---

### Logging

| Argument                                                   | Default       | Description                                                                                           |
|------------------------------------------------------------|---------------|-------------------------------------------------------------------------------------------------------|
| `--logging_strategy`                                       | `steps`       | Logging cadence: `no`, `steps`, `epoch`.                                                              |
| `--logging_steps`                                          | `500`         | Log every N update steps (or fraction of total steps).                                                |
| `--logging_first_step`                                     | `False`       | Log the very first step.                                                                              |
| `--log_on_each_node` / `--no_log_on_each_node`             | `True`        | Log on every node in multi-node training.                                                             |
| `--logging_nan_inf_filter` / `--no_logging_nan_inf_filter` | `True`        | Filter NaN / Inf losses from logs (does not affect gradients).                                        |
| `--log_level`                                              | `passive`     | Log level for the main process: `detail`, `debug`, `info`, `warning`, `error`, `critical`, `passive`. |
| `--log_level_replica`                                      | `warning`     | Log level for replica processes.                                                                      |
| `--disable_tqdm`                                           | `None`        | Disable tqdm progress bars.                                                                           |
| `--include_num_input_tokens_seen`                          | `no`          | Track input tokens seen: `all`, `non_padding`, or `no`.                                               |
| `--report_to`                                              | `none`        | Reporting integrations: `all`, `none`, or a specific integration name.                                |
| `--run_name`                                               | `None`        | Descriptor for the run (used by wandb, MLflow, comet, etc.).                                          |
| `--project`                                                | `huggingface` | Project name for logging (currently Trackio only).                                                    |
| `--trackio_space_id`                                       | `trackio`     | HuggingFace Space ID for Trackio deployment (`username/reponame`).                                    |
| `--skip_memory_metrics` / `--no_skip_memory_metrics`       | `True`        | Skip memory profiler reports in metrics (faster training).                                            |
| `--debug`                                                  | `None`        | Debug features: `underflow_overflow`, `tpu_metrics_debug`.                                            |

---

### Checkpointing & Hub

| Argument                                    | Default      | Description                                                                |
|---------------------------------------------|--------------|----------------------------------------------------------------------------|
| `--save_strategy`                           | `steps`      | Checkpoint save strategy: `no`, `steps`, `epoch`, `best`.                  |
| `--save_steps`                              | `500`        | Save checkpoint every N update steps.                                      |
| `--save_total_limit`                        | `None`       | Maximum number of checkpoints to keep.                                     |
| `--save_only_model`                         | `False`      | Save only model weights (no optimizer/scheduler state). Prevents resuming. |
| `--save_on_each_node`                       | `False`      | Save on every node in multi-node training.                                 |
| `--enable_jit_checkpoint`                   | `False`      | Enable JIT checkpointing on SIGTERM for preemptible workloads.             |
| `--push_to_hub`                             | `False`      | Push model to Hub on every save.                                           |
| `--hub_token`                               | `None`       | HuggingFace token for Hub upload (defaults to `hf auth login` token).      |
| `--hub_model_id`                            | `None`       | Repository name to sync with `output_dir`.                                 |
| `--hub_strategy`                            | `every_save` | What to push: `end`, `every_save`, `checkpoint`, `all_checkpoints`.        |
| `--hub_always_push`                         | `False`      | Push even if the previous push is not yet complete.                        |
| `--hub_private_repo`                        | `None`       | Make the Hub repo private.                                                 |
| `--hub_revision`                            | `None`       | Branch / tag / commit hash to push to.                                     |
| `--ignore_data_skip`                        | `False`      | On resume, skip fast-forwarding the dataset to the previous position.      |
| `--restore_callback_states_from_checkpoint` | `False`      | Restore callback states from checkpoint.                                   |

---

### Distributed training

| Argument                                                                 | Default | Description                                                                                                |
|--------------------------------------------------------------------------|---------|------------------------------------------------------------------------------------------------------------|
| `--average_tokens_across_devices` / `--no_average_tokens_across_devices` | `True`  | Average tokens across devices for precise loss calculation.                                                |
| `--use_cpu`                                                              | `False` | Force CPU training.                                                                                        |
| `--accelerator_config`                                                   | `None`  | Path to JSON config or dict for the Accelerate integration.                                                |
| `--parallelism_config`                                                   | `None`  | Parallelism config (requires Accelerate ≥ 1.10.1).                                                         |
| `--ddp_find_unused_parameters`                                           | `None`  | `find_unused_parameters` flag for `DistributedDataParallel`.                                               |
| `--ddp_bucket_cap_mb`                                                    | `None`  | `bucket_cap_mb` for `DistributedDataParallel`.                                                             |
| `--ddp_broadcast_buffers`                                                | `None`  | `broadcast_buffers` flag for `DistributedDataParallel`.                                                    |
| `--ddp_backend`                                                          | `None`  | DDP backend: `nccl`, `gloo`, `mpi`, `xccl`, `hccl`, `cncl`, `mccl`.                                        |
| `--ddp_timeout`                                                          | `1800`  | Timeout (seconds) for `torch.distributed.init_process_group`.                                              |
| `--fsdp`                                                                 | `None`  | FSDP options: `full_shard`, `shard_grad_op`, `hybrid_shard`, `hybrid_shard_zero2`, `offload`, `auto_wrap`. |
| `--fsdp_config`                                                          | `None`  | Path to FSDP JSON config or a dict.                                                                        |
| `--deepspeed`                                                            | `None`  | Path to DeepSpeed JSON config or a dict.                                                                   |
| `--full_determinism`                                                     | `False` | Full determinism across devices (debug only; hurts performance).                                           |

---

### DataLoader

| Argument                                                 | Default  | Description                                                      |
|----------------------------------------------------------|----------|------------------------------------------------------------------|
| `--dataloader_num_workers`                               | `0`      | Subprocesses for data loading (`0` = main process).              |
| `--dataloader_pin_memory` / `--no_dataloader_pin_memory` | `True`   | Pin memory in the DataLoader.                                    |
| `--dataloader_drop_last`                                 | `False`  | Drop the last incomplete batch.                                  |
| `--dataloader_persistent_workers`                        | `False`  | Keep worker processes alive between epochs (faster, higher RAM). |
| `--dataloader_prefetch_factor`                           | `None`   | Batches to prefetch per worker.                                  |
| `--remove_unused_columns` / `--no_remove_unused_columns` | `True`   | Auto-remove columns not used by the model forward method.        |
| `--label_names`                                          | `None`   | Keys in the input dict that correspond to labels.                |
| `--train_sampling_strategy`                              | `random` | Training sampler: `random`, `sequential`, `group_by_length`.     |
| `--length_column_name`                                   | `length` | Column with precomputed lengths (used with `group_by_length`).   |

---

### Miscellaneous / advanced

| Argument                    | Default | Description                                                                   |
|-----------------------------|---------|-------------------------------------------------------------------------------|
| `--neftune_noise_alpha`     | `None`  | NEFTune noise alpha for instruction fine-tuning. Typical range: `5.0`–`15.0`. |
| `--torch_empty_cache_steps` | `None`  | Call `torch.<device>.empty_cache()` every N steps to avoid OOM.               |
| `--auto_find_batch_size`    | `False` | Auto-find a batch size that fits in memory via exponential decay.             |
| `--use_cache`               | `False` | Enable KV cache in the model (usually not needed for training).               |
| `--local_rank`              | `-1`    | Local rank when launching with `torch.distributed.launch` (deprecated).       |

---

### Deprecated arguments

| Argument         | Replacement                                                     |
|------------------|-----------------------------------------------------------------|
| `--warmup_ratio` | Use `--warmup_steps` (also accepts floats).                     |
| `--logging_dir`  | Set the `TENSORBOARD_LOGGING_DIR` environment variable instead. |
| `--local_rank`   | Use the `local_process_index` property on the `Trainer`.        |
