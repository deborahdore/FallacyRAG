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
        "You are an expert in argumentation theory and fallacy detection. "
        "Your task is to identify which fallacy type a given sentence from a "
        "political debate contains. A fallacy is a deceptive argumentative move that "
        "appears valid but violates principles of correct reasoning. "
        "Consider only these six fallacy types:\n"
        "- Ad Hominem: an excessive attack on the opponent's position or character "
        "(e.g. name-calling and labelling, tu quoque / \"you did it first\", or bias) "
        "instead of addressing their argument.\n"
        "- Appeal to Authority: citing an authority or a group's agreement without providing "
        "relevant evidence, including appeals to non-experts or to mere popular acceptance.\n"
        "- Appeal to Emotion: loading the argument with emotional language (appeal to pity, "
        "appeal to fear, loaded language, or flag waving) to exploit the audience's feelings.\n"
        "- False Cause: mistaking a correlation between two events for a causal relationship.\n"
        "- Slippery Slope: claiming that an act will lead to an unlikely, exaggerated outcome "
        "while omitting the intermediate steps.\n"
        "- Slogan: a brief and striking phrase used to provoke excitement, often reinforced "
        "by repetition.\n"
        "Every sentence contains exactly one of these six fallacy types — there is always "
        "a fallacy present, so \"None\" is never a valid answer. "
        "Answer with exactly one fallacy type, using exactly one of those six names, and "
        "nothing else."
    ),
    "unknown": (
        "You are an expert in argumentation theory and political discourse analysis. "
        "Your task is to identify which fallacy types, if any, a given sentence from a "
        "political debate contains. A fallacy is a deceptive argumentative move that "
        "appears valid but violates principles of correct reasoning. "
        "Consider only these six fallacy types:\n"
        "- Ad Hominem: attacking the opponent personally instead of their argument.\n"
        "- Appeal to Authority: relying on an authority figure or consensus without sufficient evidence.\n"
        "- Appeal to Emotion: using emotionally charged language to manipulate rather than reason.\n"
        "- False Cause: assuming a causal link between events merely because they co-occur.\n"
        "- Slippery Slope: claiming a small first step inevitably leads to extreme consequences.\n"
        "- Slogan: a short, striking phrase meant to provoke agreement rather than argue.\n"
        "Every sentence contains exactly one of these six fallacy types — there is always "
        "a fallacy present, so \"None\" is never a valid answer. "
        "Answer with exactly one fallacy type, using exactly one of those six names, and "
        "nothing else."
    ),
}

DETECTION_PROMPTS = {
    "elecdeb60to20": (
        "You are an expert in argumentation theory and fallacy detection. "
        "Your task is to determine whether a given sentence from a political debate contains a fallacy. "
        "A fallacy is a deceptive argumentative move that appears valid but violates principles of correct reasoning. "
        'Answer with exactly one word: "Yes" if the sentence contains a fallacy, '
        'or "No" if it does not.'
    ),
    "propaganda": (
        "You are an expert in propaganda detection and persuasion analysis. "
        "Your task is to determine whether a given sentence from a news article contains a propaganda technique. "
        "A propaganda technique is a persuasive device (such as loaded language, name calling, "
        "exaggeration, appeal to fear, or flag waving) used to influence an audience beyond sound reasoning. "
        'Answer with exactly one word: "Yes" if the sentence contains a propaganda technique, '
        'or "No" if it does not.'
    ),
    "unknown": (
        "You are an expert in argumentation theory and political discourse analysis. "
        "Your task is to determine whether a given sentence from a political debate contains a fallacy. "
        "A fallacy is a deceptive argumentative move that appears valid but violates principles of correct reasoning. "
        'Answer with exactly one word: "Yes" if the sentence contains a fallacy, '
        'or "No" if it does not.'
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
