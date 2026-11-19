"""Programmatic loader for the SDFT reproduction datasets.

We use the datasets released by the paper's authors (Science Q&A = SciKnowEval
Chemistry L-3 with GPT-4o demonstrations; Tool Use = ToolAlpaca with the
original train/test split). Only the *data* is reused; no code from the
official repository is copied.

Two hosted sources are supported, tried in this order:

1. ``hf``: the team's Hugging Face mirror (see ``scripts/mirror_official_data.py``),
   repo id taken from ``--hf_dataset`` or the ``SDFT_HF_DATASET`` env variable.
   This is the source the course rules ask for.
2. ``github``: the arrow files in the official GitHub repository, pinned to a
   fixed commit so that every run reads byte-identical data.

Every example is returned as a plain dict::

    {
        "id":            str,        # stable id, e.g. "science-train-00042"
        "system":        str | None, # system prompt (Science only)
        "question":      str,        # user turn x
        "demonstration": str | None, # expert demonstration c (train/val only)
        "answer":        Any,        # gold label: letter (Science) or list of
                                     # {"action", "action_input"} dicts (Tool Use)
    }
"""

import os
import random
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

from datasets import Dataset, load_dataset, load_from_disk

OFFICIAL_REPO = "idanshen/Self-Distillation"
# Pinned commit of the official repo (2026-04-07). Change only deliberately.
OFFICIAL_COMMIT = "d77573212fa0a3ae2eeb64b9b44db1c251f75e3e"
OFFICIAL_SUBDIRS = {
    ("science", "train"): "data/science_data/train_data",
    ("science", "test"): "data/science_data/eval_data",
    ("tool_use", "train"): "data/tooluse_data/train_data",
    ("tool_use", "test"): "data/tooluse_data/eval_data",
}
HF_CONFIG_NAMES = {"science": "science", "tool_use": "tooluse"}
TASKS = ("science", "tool_use")

Example = Dict[str, Any]


# --------------------------------------------------------------------------- #
# Raw download
# --------------------------------------------------------------------------- #
def _download_official_split(task: str, split: str, cache_dir: str) -> Dataset:
    """Download one split of the official data from GitHub and load it.

    Parameters
    ----------
    task : str
        ``"science"`` or ``"tool_use"``.
    split : str
        ``"train"`` or ``"test"``.
    cache_dir : str
        Local directory used to cache the downloaded files.

    Returns
    -------
    datasets.Dataset
        The raw split, exactly as saved by the authors with ``save_to_disk``.
    """
    subdir = OFFICIAL_SUBDIRS[(task, split)]
    local = os.path.join(cache_dir, OFFICIAL_COMMIT[:12], subdir)
    files = ["data-00000-of-00001.arrow", "dataset_info.json", "state.json"]
    if not all(os.path.exists(os.path.join(local, f)) for f in files):
        os.makedirs(local, exist_ok=True)
        base = f"https://raw.githubusercontent.com/{OFFICIAL_REPO}/{OFFICIAL_COMMIT}/{subdir}"
        for f in files:
            urllib.request.urlretrieve(f"{base}/{f}", os.path.join(local, f))
    return load_from_disk(local)


def load_raw_split(
    task: str,
    split: str,
    source: str = "auto",
    hf_dataset: Optional[str] = None,
    cache_dir: str = "./data_cache",
) -> Dataset:
    """Load one raw split from the team's HF mirror or the pinned GitHub copy.

    Parameters
    ----------
    task : str
        ``"science"`` or ``"tool_use"``.
    split : str
        ``"train"`` or ``"test"``.
    source : str
        ``"hf"``, ``"github"`` or ``"auto"`` (HF if a repo id is configured,
        otherwise GitHub).
    hf_dataset : str, optional
        Hugging Face dataset repo id; defaults to ``$SDFT_HF_DATASET``.
    cache_dir : str
        Cache directory for the GitHub download.

    Returns
    -------
    datasets.Dataset
        Raw split with the authors' column names.
    """
    if task not in TASKS:
        raise ValueError(f"Unsupported task {task!r}; choose one of {TASKS}.")
    hf_dataset = hf_dataset or os.environ.get("SDFT_HF_DATASET")
    if source == "auto":
        source = "hf" if hf_dataset else "github"
    if source == "hf":
        if not hf_dataset:
            raise ValueError("source='hf' needs --hf_dataset or $SDFT_HF_DATASET.")
        return load_dataset(hf_dataset, HF_CONFIG_NAMES[task], split=split)
    if source == "github":
        return _download_official_split(task, split, cache_dir)
    raise ValueError(f"Unknown source {source!r}.")


# --------------------------------------------------------------------------- #
# Formatting into the common example schema
# --------------------------------------------------------------------------- #
def _format_tool_demonstration(steps: List[str]) -> str:
    """Turn the authors' list of reasoning/action steps into one ReAct string.

    Each official step looks like ``" I need to ...\\nAction: X\\nAction Input: {...}"``
    (the leading ``Thought:`` keyword is missing because the prompt ends with
    the format instructions). We restore it so the demonstration follows the
    exact ``Thought / Action / Action Input`` format the prompt asks for.

    Parameters
    ----------
    steps : list of str
        ``golden_response`` field of one ToolAlpaca example.

    Returns
    -------
    str
        Demonstration text.
    """
    return "\n".join(f"Thought: {s.strip()}" for s in steps)


def _format_tool_answer(golden: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Normalise the gold tool calls to ``[{"action", "action_input"}]``.

    Parameters
    ----------
    golden : list of dict
        ``golden_answer`` field, each item with ``Action`` and ``Action_Input``.

    Returns
    -------
    list of dict
        Gold calls in call order; ``action_input`` is kept as the raw JSON string.
    """
    return [{"action": g["Action"], "action_input": g["Action_Input"]} for g in golden]


def format_split(task: str, split: str, raw: Dataset) -> List[Example]:
    """Convert a raw official split into the common example schema.

    Parameters
    ----------
    task : str
        ``"science"`` or ``"tool_use"``.
    split : str
        ``"train"`` or ``"test"`` (only used to build ids).
    raw : datasets.Dataset
        Output of :func:`load_raw_split`.

    Returns
    -------
    list of dict
        Examples with keys ``id, system, question, demonstration, answer``.
    """
    out: List[Example] = []
    for i, row in enumerate(raw):
        if task == "science":
            msgs = row["messages"] if split == "train" else row["prompt"]
            system = next((m["content"] for m in msgs if m["role"] == "system"), None)
            question = next(m["content"] for m in msgs if m["role"] == "user")
            if split == "train":
                demo = row["output_text"]
                answer = _science_answer_from_demo(demo)
            else:
                demo, answer = None, row["answer"].strip()
        else:
            system = None
            question = row["prompt"]
            demo = _format_tool_demonstration(row["golden_response"]) if split == "train" else None
            answer = _format_tool_answer(row["golden_answer"])
        out.append(
            {
                "id": f"{task}-{split}-{i:05d}",
                "system": system,
                "question": question,
                "demonstration": demo,
                "answer": answer,
            }
        )
    return out


def _science_answer_from_demo(demo: str) -> Optional[str]:
    """Read the option letter from a Science demonstration's ``<answer>`` tag.

    The official train split stores only the GPT-4o demonstration, so the gold
    letter of a *training/validation* example is recovered from it (the authors
    kept only demonstrations whose final answer matched the ground truth).

    Parameters
    ----------
    demo : str
        Demonstration text.

    Returns
    -------
    str or None
        ``"A"``-``"D"`` or ``None`` if the tag is missing.
    """
    from answer_parsing import parse_science_answer  # local import avoids a cycle

    return parse_science_answer(demo)


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def get_dataset(
    task: str,
    max_train: Optional[int] = None,
    val_fraction: float = 0.05,
    data_seed: int = 0,
    max_test: Optional[int] = None,
    source: str = "auto",
    hf_dataset: Optional[str] = None,
    cache_dir: str = "./data_cache",
) -> Tuple[List[Example], List[Example], List[Example]]:
    """Return deterministic (train, val, test) lists for one task.

    The validation set is carved out of the official training split *before*
    any sub-sampling, so it is identical for every run regardless of
    ``max_train``. The training subset (if ``max_train`` is set) is a prefix of
    one fixed permutation, so a 1 000-example subset is contained in a 2 000-
    example subset, and SFT and SDFT always see exactly the same examples.
    The test split is the authors' held-out split and is never sub-sampled
    unless ``max_test`` is given explicitly (for smoke tests only).

    Parameters
    ----------
    task : str
        ``"science"`` or ``"tool_use"``.
    max_train : int, optional
        Number of training examples to keep; ``None`` keeps all.
    val_fraction : float
        Fraction of the official train split held out for checkpoint selection.
    data_seed : int
        Seed of the data permutation. Kept separate from the training seed so
        that changing the training seed does not change the data.
    max_test : int, optional
        Truncate the test split (smoke tests only).
    source, hf_dataset, cache_dir
        Forwarded to :func:`load_raw_split`.

    Returns
    -------
    tuple of (train, val, test)
        Lists of example dicts.
    """
    train_pool = format_split(
        task, "train", load_raw_split(task, "train", source, hf_dataset, cache_dir)
    )
    test = format_split(task, "test", load_raw_split(task, "test", source, hf_dataset, cache_dir))

    rng = random.Random(data_seed)
    order = list(range(len(train_pool)))
    rng.shuffle(order)
    n_val = int(round(val_fraction * len(train_pool)))
    val = [train_pool[i] for i in order[:n_val]]
    train = [train_pool[i] for i in order[n_val:]]
    if max_train is not None:
        train = train[:max_train]
    if max_test is not None:
        test = test[:max_test]
    return train, val, test


def to_messages(system: Optional[str], user: str) -> List[Dict[str, str]]:
    """Build a chat message list from an optional system prompt and a user turn.

    Parameters
    ----------
    system : str or None
        System prompt.
    user : str
        User content.

    Returns
    -------
    list of dict
        Messages ready for ``tokenizer.apply_chat_template``.
    """
    msgs = [{"role": "system", "content": system}] if system else []
    msgs.append({"role": "user", "content": user})
    return msgs
