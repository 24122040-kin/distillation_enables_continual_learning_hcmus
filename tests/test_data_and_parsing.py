"""Sanity tests for the data loader and the answer parsers (CPU only, no model).

Run with ``pytest tests/`` from the repository root. The first run downloads the
official data (~28 MB) into ./data_cache.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from answer_parsing import (  # noqa: E402
    parse_science_answer,
    parse_tool_calls,
    score,
    tool_match_merged,
    tool_match_strict,
)
from data_loader import get_dataset  # noqa: E402


@pytest.fixture(scope="module")
def science():
    return get_dataset("science", source="github")


@pytest.fixture(scope="module")
def tool():
    return get_dataset("tool_use", source="github")


def test_split_sizes(science, tool):
    s_train, s_val, s_test = science
    t_train, t_val, t_test = tool
    assert len(s_train) + len(s_val) == 2674 and len(s_test) == 507
    assert len(t_train) + len(t_val) == 4046 and len(t_test) == 97


def test_subset_is_prefix_and_val_is_fixed():
    full_train, full_val, _ = get_dataset("science", source="github")
    sub_train, sub_val, _ = get_dataset("science", max_train=500, source="github")
    assert [e["id"] for e in sub_val] == [e["id"] for e in full_val]
    assert [e["id"] for e in sub_train] == [e["id"] for e in full_train[:500]]
    assert not {e["id"] for e in full_val} & {e["id"] for e in full_train}


def test_science_fields(science):
    train, val, test = science
    assert all(e["system"] and "<answer>" in e["system"] for e in train + test)
    assert all("A:" in e["question"] and "D:" in e["question"] for e in test)
    # every demonstration carries a parseable answer letter
    assert all(e["answer"] in "ABCD" for e in train + val)
    assert all(e["answer"] in "ABCD" for e in test)


def test_science_parser():
    assert parse_science_answer("<reasoning>A molecule ...</reasoning>\n<answer>\nC\n</answer>") == "C"
    assert parse_science_answer("A molecule is small. The answer is B") is None  # no guessing
    assert parse_science_answer("<answer>A</answer> wait <answer>\nD\n</answer>") == "D"
    assert parse_science_answer("<reasoning>...</reasoning>\n<answer>\nB") == "B"  # cut before closing tag


def test_science_demos_score_correct(science):
    train, _, _ = science
    assert all(score("science", e["demonstration"], e["answer"])["acc"] == 1.0 for e in train)


def test_tool_parser_nested_json():
    text = 'Thought: x\nAction: send\nAction Input: {"url": "u", "headers": {"a": "b"}}\nObservation: ok'
    assert parse_tool_calls(text) == [("send", {"url": "u", "headers": {"a": "b"}})]


def test_tool_metrics_on_gold_demonstrations(tool):
    train, _, _ = tool
    merged = sum(tool_match_merged(e["demonstration"], e["answer"]) for e in train)
    strict = sum(tool_match_strict(e["demonstration"], e["answer"]) for e in train)
    # the gold demonstration must score (almost) perfectly under both metrics
    assert merged / len(train) > 0.99 and strict / len(train) > 0.99


def test_tool_metric_on_test_gold_echo(tool):
    _, _, test = tool
    echo = lambda gold: "\n".join(f"Action: {g['action']}\nAction Input: {g['action_input']}" for g in gold)
    assert all(tool_match_merged(echo(e["answer"]), e["answer"]) for e in test)
    assert all(tool_match_strict(echo(e["answer"]), e["answer"]) for e in test)


def test_tool_metric_rejects_wrong_args(tool):
    _, _, test = tool
    gold = test[0]["answer"]
    wrong = f"Action: {gold[0]['action']}\nAction Input: {{\"bogus\": 1}}"
    assert not tool_match_merged(wrong, gold)
