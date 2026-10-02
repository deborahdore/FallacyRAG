"""Dataset-aware prompt configuration for fallacy classification and detection.
Maps each dataset (elecdeb60to20, propaganda) to its system prompt and label space, following
the papers and annotation schemes they use. The dataset is identified from the training file
path, so the right prompt is picked automatically without code changes."""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def identify_dataset(train_file: Optional[str]) -> str:
    """Identify the dataset name from the training file path.
    Returns "elecdeb60to20", "propaganda", or "unknown"."""
    if train_file is None:
        return "unknown"

    path = Path(train_file).as_posix().lower()

    if "elecdeb60to20" in path or "elecdeb" in path:
        return "elecdeb60to20"
    elif "propaganda" in path:
        return "propaganda"
    else:
        return "unknown"


CLASSIFICATION_PROMPTS = {
    "elecdeb60to20": (
        "You classify fallacies in political debate sentences. Types:\n"
        "- Ad Hominem: attacks the opponent's character or position instead of the argument "
        "(name-calling, tu quoque, bias).\n"
        "- Appeal to Authority: cites an authority or popular agreement without relevant evidence.\n"
        "- Appeal to Emotion: emotional language (pity, fear, loaded words, flag waving) "
        "instead of reasons.\n"
        "- False Cause: treats a correlation as causation.\n"
        "- Slippery Slope: claims an act leads to an exaggerated outcome, skipping the steps between.\n"
        "- Slogan: brief, striking phrase meant to provoke excitement.\n"
        "Each sentence contains exactly one type."
    ),
    "unknown": (
        "You classify fallacies in political debate sentences. Types:\n"
        "- Ad Hominem: attacks the opponent instead of the argument.\n"
        "- Appeal to Authority: relies on authority or consensus without evidence.\n"
        "- Appeal to Emotion: manipulates with emotional language.\n"
        "- False Cause: assumes causation from co-occurrence.\n"
        "- Slippery Slope: a first step supposedly leads to extreme consequences.\n"
        "- Slogan: short, striking phrase instead of an argument.\n"
        "Each sentence contains exactly one type."
    ),
}

DETECTION_PROMPTS = {
    "elecdeb60to20": (
        "You detect fallacies in political debate sentences. A fallacy is an argument that "
        "seems valid but violates correct reasoning."
    ),
    "propaganda": (
        "You detect propaganda in news article sentences. A propaganda technique is a persuasive "
        "device (e.g. loaded language, name calling, exaggeration, appeal to fear, flag waving) "
        "that sways the audience beyond sound reasoning."
    ),
    "unknown": (
        "You detect fallacies in political debate sentences. A fallacy is an argument that "
        "seems valid but violates correct reasoning."
    ),
}

_DETECTION_LABELS_BY_DATASET: dict[str, tuple[str, str]] = {
    "elecdeb60to20": ("Fallacy", "NoFallacy"),
    "propaganda": ("propaganda", "non-propaganda"),
    "unknown": ("Fallacy", "NoFallacy"),
}


def get_detection_label_config(train_file: Optional[str]) -> dict:
    """Get the dataset-specific positive/negative labels for binary detection.
    Returns a dict with ``dataset``, ``positive_label``, ``negative_label`` and ``labels``."""
    dataset = identify_dataset(train_file)
    positive, negative = _DETECTION_LABELS_BY_DATASET.get(
        dataset, _DETECTION_LABELS_BY_DATASET["unknown"]
    )
    logger.info(f"Using detection labels for dataset '{dataset}': {positive} / {negative}")
    return {
        "dataset": dataset,
        "positive_label": positive,
        "negative_label": negative,
        "labels": [positive, negative],
    }


_LABEL_TO_LLM_BY_DATASET: dict[str, dict[str, str]] = {
    "elecdeb60to20": {
        "AdHominem": "Ad Hominem",
        "AppealtoAuthority": "Appeal to Authority",
        "AppealtoEmotion": "Appeal to Emotion",
        "FalseCause": "False Cause",
        "Slipperyslope": "Slippery Slope",
        "Slogans": "Slogan",
    },
}
_LABEL_TO_LLM_BY_DATASET["unknown"] = _LABEL_TO_LLM_BY_DATASET["elecdeb60to20"]

_SINGLE_LABEL_NONE_BY_DATASET: dict[str, str] = {}

_EXTRA_ALIASES_BY_DATASET: dict[str, dict[str, list[str]]] = {
    "elecdeb60to20": {
        "AdHominem": ["ad-hominem", "personal attack", "name calling", "name-calling",
                      "labeling", "tu quoque"],
        "AppealtoAuthority": ["appeal-to-authority", "authority"],
        "AppealtoEmotion": ["appeal-to-emotion", "emotion", "emotional appeal",
                            "loaded language", "flag waving", "appeal to fear",
                            "appeal to pity"],
        "FalseCause": ["false-cause", "post hoc", "correlation causation"],
        "Slipperyslope": ["slippery-slope", "slippery", "slope"],
        "Slogans": ["slogan"],
    },
}
_EXTRA_ALIASES_BY_DATASET["unknown"] = _EXTRA_ALIASES_BY_DATASET["elecdeb60to20"]


def _normalize_label(text: str) -> str:
    """Lowercase and strip every non-alphanumeric character (for alias matching)."""
    import re
    return re.sub(r"[^a-z0-9]", "", text.lower())


def get_label_config(train_file: Optional[str]) -> dict:
    """Get the dataset-specific label configuration for single-label classification.
    Returns ``dataset``, ``labels``, ``label_to_llm``, ``alias_to_label``, ``label_menu``, ``none_label``."""
    dataset = identify_dataset(train_file)
    label_to_llm = dict(_LABEL_TO_LLM_BY_DATASET.get(dataset, _LABEL_TO_LLM_BY_DATASET["unknown"]))
    labels = list(label_to_llm.keys())

    alias_to_label: dict[str, str] = {}

    def _add(canon: str, *variants: str):
        for v in variants:
            n = _normalize_label(v)
            if n:
                alias_to_label[n] = canon

    for canon, llm in label_to_llm.items():
        _add(canon, canon, llm)
    for canon, variants in _EXTRA_ALIASES_BY_DATASET.get(dataset, {}).items():
        _add(canon, *variants)

    label_menu = ", ".join(label_to_llm[lab] for lab in labels)
    logger.info(f"Using label space for dataset '{dataset}' ({len(labels)}): {labels}")
    return {
        "dataset": dataset,
        "labels": labels,
        "label_to_llm": label_to_llm,
        "alias_to_label": alias_to_label,
        "label_menu": label_menu,
        "none_label": _SINGLE_LABEL_NONE_BY_DATASET.get(dataset),
    }


def get_classification_prompt(train_file: Optional[str]) -> str:
    """Get the classification system prompt for the dataset used in training.
    The dataset is identified from the training file path."""
    dataset_name = identify_dataset(train_file)
    prompt = CLASSIFICATION_PROMPTS.get(dataset_name, CLASSIFICATION_PROMPTS["unknown"])
    logger.info(f"Using classification prompt for dataset: {dataset_name}")
    return prompt


def get_detection_prompt(train_file: Optional[str]) -> str:
    """Get the detection system prompt for the dataset used in training.
    The dataset is identified from the training file path."""
    dataset_name = identify_dataset(train_file)
    prompt = DETECTION_PROMPTS.get(dataset_name, DETECTION_PROMPTS["unknown"])
    logger.info(f"Using detection prompt for dataset: {dataset_name}")
    return prompt
