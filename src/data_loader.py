"""Programmatic dataset loader for SDFT reproduction experiments.

Loads datasets directly from Hugging Face Hub or public GitHub URLs,
applies deterministic splits, and formats examples into (prompt, demonstration, answer)
triplets required by SFT and SDFT.
"""

import json
import random
import urllib.request
from typing import Dict, List, Tuple

from datasets import load_dataset


def load_science_qa(
    seed: int = 42,
    max_train: int = 400,
    max_val: int = 50,
    max_test: int = 150,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
    """Load the Chemistry L3 subset of SciKnowEval from Hugging Face Hub.

    Parameters
    ----------
    seed : int
        Random seed for deterministic shuffling and splitting.
    max_train : int
        Maximum number of training examples to return (for free-tier GPU budgets).
    max_val : int
        Maximum number of validation examples.
    max_test : int
        Maximum number of test examples.

    Returns
    -------
    tuple of (train_data, val_data, test_data)
        Each item is a dict with keys: 'prompt', 'demonstration', 'answer'.
    """
    ds = load_dataset("hicai-zju/SciKnowEval", split="test")

    filtered = []
    for row in ds:
        domain = str(row.get("domain", "")).lower()
        level = str(row.get("level", "")).lower()
        if "chem" in domain and ("l3" in level or "3" in level):
            question = row.get("question", "")
            choices = row.get("choices", row.get("options", []))
            if isinstance(choices, dict) and "text" in choices:
                choices = choices["text"]
            if isinstance(choices, list) and choices:
                choice_str = "\n".join(
                    f"{chr(65 + i)}. {c}" for i, c in enumerate(choices)
                )
                prompt = f"{question}\n{choice_str}\nAnswer with the option letter."
            else:
                prompt = str(question)

            answer = str(row.get("answer", row.get("answerKey", ""))).strip()
            solution = str(row.get("solution", row.get("explanation", ""))).strip()
            demo = f"{solution}\nFinal Answer: {answer}" if solution else f"Final Answer: {answer}"

            if prompt and answer:
                filtered.append(
                    {
                        "prompt": prompt,
                        "demonstration": demo,
                        "answer": answer,
                    }
                )

    if not filtered:
        # Fallback to general chemistry subset if level tags differ in Hub revision
        for row in ds:
            if "chem" in str(row.get("domain", "")).lower():
                question = str(row.get("question", ""))
                answer = str(row.get("answer", row.get("answerKey", ""))).strip()
                if question and answer:
                    filtered.append(
                        {
                            "prompt": question,
                            "demonstration": f"Final Answer: {answer}",
                            "answer": answer,
                        }
                    )

    rng = random.Random(seed)
    rng.shuffle(filtered)

    n_total = len(filtered)
    n_train = min(int(0.75 * n_total), max_train)
    n_val = min(int(0.05 * n_total), max_val)
    n_test = min(n_total - n_train - n_val, max_test)

    train_data = filtered[:n_train]
    val_data = filtered[n_train : n_train + n_val]
    test_data = filtered[n_train + n_val : n_train + n_val + n_test]
    return train_data, val_data, test_data


def load_tool_alpaca(
    seed: int = 42,
    max_train: int = 400,
    max_test: int = 150,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
    """Load ToolAlpaca simulated tool-use demonstrations from the official GitHub repo.

    Parameters
    ----------
    seed : int
        Random seed for deterministic splitting.
    max_train : int
        Maximum number of training examples.
    max_test : int
        Maximum number of test examples.

    Returns
    -------
    tuple of (train_data, val_data, test_data)
        Each item is a dict with keys: 'prompt', 'demonstration', 'answer'.
    """
    url = "https://raw.githubusercontent.com/tangqiaoyu/ToolAlpaca/main/data/train_data.json"
    with urllib.request.urlopen(url, timeout=30) as resp:
        raw = json.loads(resp.read().decode("utf-8"))

    formatted = []
    for item in raw:
        nl_desc = item.get("NLDocumentation", "")
        instances = item.get("Instances", [])
        for inst in instances:
            raw_in = inst.get("input", "")
            raw_out = inst.get("output", "")
            user_input = (json.dumps(raw_in) if isinstance(raw_in, (dict, list)) else str(raw_in)).strip()
            output = (json.dumps(raw_out) if isinstance(raw_out, (dict, list)) else str(raw_out)).strip()
            if user_input and output:
                prompt = f"Tool Documentation:\n{nl_desc}\n\nUser Request: {user_input}"
                formatted.append(
                    {
                        "prompt": prompt,
                        "demonstration": output,
                        "answer": output,
                    }
                )

    rng = random.Random(seed)
    rng.shuffle(formatted)

    n_train = min(int(0.8 * len(formatted)), max_train)
    n_val = min(50, len(formatted) - n_train)
    train_data = formatted[:n_train]
    val_data = formatted[n_train : n_train + n_val]
    test_data = formatted[n_train + n_val : n_train + n_val + max_test]
    return train_data, val_data, test_data


def get_dataset(
    task: str,
    seed: int = 42,
    max_train: int = 400,
    max_test: int = 150,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
    """Dispatch dataset loading by task name.

    Parameters
    ----------
    task : str
        One of 'science_qa' or 'tool_use'.
    seed : int
        Random seed for deterministic splitting.
    max_train : int
        Maximum number of training examples.
    max_test : int
        Maximum number of test examples.

    Returns
    -------
    tuple of (train_data, val_data, test_data)
    """
    if task == "science_qa":
        return load_science_qa(seed=seed, max_train=max_train, max_test=max_test)
    if task == "tool_use":
        return load_tool_alpaca(seed=seed, max_train=max_train, max_test=max_test)
    raise ValueError(f"Unsupported task: {task}. Choose 'science_qa' or 'tool_use'.")
