"""
Dataset-aware prompt configuration for fallacy classification and detection.

This module maps datasets (elecdeb60to20, reddit, propaganda) to their
corresponding system prompts, based on the papers and annotation schemes used.

The prompt is automatically selected based on the dataset path or name provided
to the LLM scripts, without requiring manual changes to the code.
"""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ── Dataset identification ────────────────────────────────────────────────────

def identify_dataset(train_file: Optional[str]) -> str:
    """
    Identify the dataset name from the training file path.

    Parameters
    ----------
    train_file : Optional[str]
        Path to the training file (CSV or JSON).

    Returns
    -------
    str
        One of: "elecdeb60to20", "reddit", "propaganda", or "unknown".

    Examples
    --------
    >>> identify_dataset("./datasets/elecdeb60to20-dataset/classification/train.csv")
    'elecdeb60to20'
    >>> identify_dataset("./datasets/reddit-dataset/detection/train.csv")
    'reddit'
    >>> identify_dataset("./datasets/propaganda-dataset/detection/train.csv")
    'propaganda'
    """
    if train_file is None:
        return "unknown"

    path = Path(train_file).as_posix().lower()

    if "elecdeb60to20" in path or "elecdeb" in path:
        return "elecdeb60to20"
    elif "reddit" in path:
        return "reddit"
    elif "propaganda" in path:
        return "propaganda"
    else:
        return "unknown"


# ── Classification prompts ────────────────────────────────────────────────────

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
    "reddit": (
        "You are an expert in argumentation theory and fallacy detection. "
        "Your task is to identify which fallacy type a given sentence from an online "
        "discussion (Reddit) contains. A fallacy is a deceptive argumentative move that "
        "appears valid but violates principles of correct reasoning. "
        "Consider only these eight fallacy types:\n"
        "- Appeal to Authority: holding a claim to be true because it is supported by the "
        "opinion of a person with authority (argumentum ad verecundiam).\n"
        "- Appeal to Majority: holding a claim to be true because many people believe it "
        "(bandwagon / appeal to the people / argumentum ad populum).\n"
        "- Appeal to Nature: justifying or rejecting an action because it does or does not "
        "occur in nature (naturalistic fallacy).\n"
        "- Appeal to Tradition: justifying or rejecting an action because it has always been "
        "considered that way in the past (argumentum ad antiquitatem).\n"
        "- Appeal to Worse Problems: dismissing a problem as unimportant because another "
        "problem is worse (relative privation / \"not as bad as\").\n"
        "- Black-or-White: presenting only two options as possible when at least one other "
        "option exists, in order to force one of them (false dilemma / false dichotomy).\n"
        "- Hasty Generalization: drawing a broad conclusion about a population from "
        "insufficient or unrepresentative evidence.\n"
        "- Slippery Slope: claiming a small first step will lead to a large unwanted "
        "consequence through a chain of causality.\n"
        "A sentence contains at most one of these eight fallacy types, but it may also "
        "contain no fallacy at all. "
        "Answer with exactly one of those eight type names, or \"No Fallacy\" if the "
        "sentence contains none of them, and nothing else."
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

# ── Detection prompts ─────────────────────────────────────────────────────────

DETECTION_PROMPTS = {
    "elecdeb60to20": (
        "You are an expert in argumentation theory and fallacy detection. "
        "Your task is to determine whether a given sentence from a political debate contains a fallacy. "
        "A fallacy is a deceptive argumentative move that appears valid but violates principles of correct reasoning. "
        'Answer with exactly one word: "Yes" if the sentence contains a fallacy, '
        'or "No" if it does not.'
    ),
    "reddit": (
        "You are an expert in argumentation theory and fallacy detection. "
        "Your task is to determine whether a given sentence from an online discussion (Reddit) contains a fallacy. "
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

# ── Detection label configuration (binary detection) ──────────────────────────
# Maps each dataset to its (positive, negative) CSV "fallacy_label" values.
# The positive label is translated to the LLM "Yes" token, the negative to "No".

_DETECTION_LABELS_BY_DATASET: dict[str, tuple[str, str]] = {
    "elecdeb60to20": ("Fallacy", "NoFallacy"),
    "reddit": ("Fallacy", "NoFallacy"),
    "propaganda": ("propaganda", "non-propaganda"),
    "unknown": ("Fallacy", "NoFallacy"),
}


def get_detection_label_config(train_file: Optional[str]) -> dict:
    """
    Get the dataset-specific positive/negative labels for binary detection.

    Parameters
    ----------
    train_file : Optional[str]
        Path to the training file (CSV or JSON).

    Returns
    -------
    dict with keys:
        dataset         : identified dataset name
        positive_label  : CSV label mapped to the LLM "Yes" token
        negative_label  : CSV label mapped to the LLM "No" token
        labels          : [positive_label, negative_label]

    Example
    -------
    >>> cfg = get_detection_label_config("./datasets/propaganda-dataset/detection/train.csv")
    >>> cfg["positive_label"], cfg["negative_label"]
    ('propaganda', 'non-propaganda')
    """
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


# ── Label configuration (single-label classification) ─────────────────────────
# Keys are canonical CSV "fallacy_label" values; values are the readable names the LLM emits.
# "No Fallacy" is handled as the empty label set elsewhere, so it is not listed here.

_LABEL_TO_LLM_BY_DATASET: dict[str, dict[str, str]] = {
    "elecdeb60to20": {
        "AdHominem": "Ad Hominem",
        "AppealtoAuthority": "Appeal to Authority",
        "AppealtoEmotion": "Appeal to Emotion",
        "FalseCause": "False Cause",
        "Slipperyslope": "Slippery Slope",
        "Slogans": "Slogan",
    },
    "reddit": {
        "Authority": "Appeal to Authority",
        "Population": "Appeal to Majority",
        "Natural": "Appeal to Nature",
        "Tradition": "Appeal to Tradition",
        "WorseProblems": "Appeal to Worse Problems",
        "Blackwhite": "Black-or-White",
        "HastyGeneralization": "Hasty Generalization",
        "SlipperySlope": "Slippery Slope",
        # Explicit negative class: reddit classification doubles as detection (see create_db).
        "NoFallacy": "No Fallacy",
    },
}
_LABEL_TO_LLM_BY_DATASET["unknown"] = _LABEL_TO_LLM_BY_DATASET["elecdeb60to20"]

# Single-label datasets that carry an explicit negative "No Fallacy" class (classification+detection).
# Its canonical value is a real, parseable, scored label — not the multi-label "None of the above".
_SINGLE_LABEL_NONE_BY_DATASET: dict[str, str] = {"reddit": "NoFallacy"}

# Extra alias variants a model tends to emit, mapped to the canonical label.
# Added on top of the canonical and readable names, which register automatically.
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
    "reddit": {
        "Authority": ["authority", "argument from authority", "ad verecundiam"],
        "Population": ["majority", "bandwagon", "ad populum", "appeal to the people",
                       "appeal to popularity", "popular opinion", "population"],
        "Natural": ["nature", "natural", "naturalistic"],
        "Tradition": ["tradition", "ad antiquitatem"],
        "WorseProblems": ["worse problems", "relative privation", "not as bad",
                          "worseproblems"],
        "Blackwhite": ["black or white", "black-or-white", "black and white",
                       "false dilemma", "false dichotomy", "bifurcation", "blackwhite"],
        "HastyGeneralization": ["hasty generalization", "hasty generalisation",
                                "generalization"],
        "SlipperySlope": ["slippery slope", "slippery", "slope",
                          "thin edge of the wedge", "camel's nose"],
        "NoFallacy": ["no fallacy", "nofallacy", "no-fallacy", "not a fallacy",
                      "none", "no fallacies", "no fallacy present"],
    },
}
_EXTRA_ALIASES_BY_DATASET["unknown"] = _EXTRA_ALIASES_BY_DATASET["elecdeb60to20"]


def _normalize_label(text: str) -> str:
    """Lowercase and strip every non-alphanumeric character (for alias matching)."""
    import re
    return re.sub(r"[^a-z0-9]", "", text.lower())


def get_label_config(train_file: Optional[str]) -> dict:
    """
    Get the dataset-specific label configuration for single-label classification.

    Parameters
    ----------
    train_file : Optional[str]
        Path to the training file (CSV or JSON).

    Returns
    -------
    dict with keys:
        dataset        : identified dataset name
        labels         : ordered list of canonical labels (CSV "fallacy_label" values)
        label_to_llm   : {canonical -> human-readable name shown to the LLM}
        alias_to_label : {normalized alias -> canonical label} for parsing responses
        label_menu     : comma-separated readable names, for the prompt
        none_label     : canonical negative "No Fallacy" class if the dataset has one, else None

    Example
    -------
    >>> cfg = get_label_config("./datasets/reddit-dataset/classification/train.csv")
    >>> cfg["labels"][0]
    'Authority'
    """
    dataset = identify_dataset(train_file)
    label_to_llm = dict(_LABEL_TO_LLM_BY_DATASET.get(dataset, _LABEL_TO_LLM_BY_DATASET["unknown"]))
    labels = list(label_to_llm.keys())

    alias_to_label: dict[str, str] = {}

    def _add(canon: str, *variants: str):
        for v in variants:
            n = _normalize_label(v)
            if n:
                alias_to_label[n] = canon

    # Always register the canonical label and its readable name.
    for canon, llm in label_to_llm.items():
        _add(canon, canon, llm)
    # Then the hand-written variants.
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
    """
    Get the classification system prompt for the dataset used in training.

    Parameters
    ----------
    train_file : Optional[str]
        Path to the training file (CSV or JSON).

    Returns
    -------
    str
        The dataset-specific system prompt for classification.

    Example
    -------
    >>> prompt = get_classification_prompt("./datasets/reddit-dataset/classification/train.csv")
    >>> "Reddit" in prompt
    True
    """
    dataset_name = identify_dataset(train_file)
    prompt = CLASSIFICATION_PROMPTS.get(dataset_name, CLASSIFICATION_PROMPTS["unknown"])
    logger.info(f"Using classification prompt for dataset: {dataset_name}")
    return prompt


def get_detection_prompt(train_file: Optional[str]) -> str:
    """
    Get the detection system prompt for the dataset used in training.

    Parameters
    ----------
    train_file : Optional[str]
        Path to the training file (CSV or JSON).

    Returns
    -------
    str
        The dataset-specific system prompt for detection.

    Example
    -------
    >>> prompt = get_detection_prompt("./datasets/reddit-dataset/detection/train.csv")
    >>> "Reddit" in prompt
    True
    """
    dataset_name = identify_dataset(train_file)
    prompt = DETECTION_PROMPTS.get(dataset_name, DETECTION_PROMPTS["unknown"])
    logger.info(f"Using detection prompt for dataset: {dataset_name}")
    return prompt
