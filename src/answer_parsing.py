"""Answer extraction and scoring for Science Q&A and Tool Use.

Pure Python (no torch), so it can be unit-tested anywhere.

Science: the prompt asks for ``<reasoning>...</reasoning><answer>X</answer>``;
we read the letter from the *last* ``<answer>`` tag.

Tool Use: the paper scores "regex matching against the ground-truth API call,
accounting for variations in argument ordering" (Appendix B.3). We parse every
``Action:`` / ``Action Input:`` pair with a real JSON decoder (so nested
objects are handled) and report two metrics:

* ``tool_match_merged`` — same multiset of action names and the same merged
  argument dict. This mirrors how the paper's numbers were computed.
* ``tool_match_strict`` — the same ordered sequence of (action, arguments)
  calls. Stricter; reported as a secondary metric.
"""

import json
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

_SCIENCE_TAG = re.compile(r"<answer>\s*([A-Da-d])\s*(?:</answer>|$)")
# Any non-blank token: one gold test action is literally "[Optional]updateUserProfile".
_ACTION = re.compile(r"Action:[ \t]*(\S+)")
_ACTION_INPUT = re.compile(r"Action Input:\s*")
_DECODER = json.JSONDecoder()


def parse_science_answer(text: str) -> Optional[str]:
    """Extract the option letter from the last ``<answer>`` tag.

    Parameters
    ----------
    text : str
        Model response.

    Returns
    -------
    str or None
        Upper-case letter ``A``-``D``, or ``None`` when no well-formed tag exists.
        A missing tag counts as wrong; we deliberately do not guess from free
        text, because a bare letter search matches articles such as "A molecule".
    """
    matches = _SCIENCE_TAG.findall(text)
    return matches[-1].upper() if matches else None


def science_correct(response: str, gold: str) -> bool:
    """Return whether a Science response selects the gold option.

    Parameters
    ----------
    response : str
        Model response.
    gold : str
        Gold option letter.

    Returns
    -------
    bool
        ``True`` iff the parsed letter equals the gold letter.
    """
    return parse_science_answer(response) == gold.strip().upper()


def _decode_json_prefix(text: str) -> Optional[Any]:
    """Decode the first JSON value at the start of ``text`` (nested braces allowed).

    Parameters
    ----------
    text : str
        Text that should begin with a JSON value.

    Returns
    -------
    Any or None
        Decoded value, or ``None`` if the text does not start with valid JSON.
    """
    try:
        value, _ = _DECODER.raw_decode(text.lstrip())
        return value
    except json.JSONDecodeError:
        return None


def parse_tool_calls(text: str) -> List[Tuple[str, Optional[Dict[str, Any]]]]:
    """Extract ``(action, arguments)`` pairs in the order they appear.

    Parameters
    ----------
    text : str
        Model response in ReAct format.

    Returns
    -------
    list of (str, dict or None)
        One pair per ``Action:`` line. ``arguments`` is the JSON object that
        follows the next ``Action Input:`` (``None`` if missing or invalid).
    """
    calls: List[Tuple[str, Optional[Dict[str, Any]]]] = []
    for m in _ACTION.finditer(text):
        rest = text[m.end():]
        nxt = _ACTION.search(rest)
        segment = rest[: nxt.start()] if nxt else rest
        args = None
        mi = _ACTION_INPUT.search(segment)
        if mi:
            decoded = _decode_json_prefix(segment[mi.end():])
            args = decoded if isinstance(decoded, dict) else None
        calls.append((m.group(1), args))
    return calls


def _gold_calls(gold: List[Dict[str, str]]) -> List[Tuple[str, Dict[str, Any]]]:
    """Parse the gold call list into ``(action, arguments)`` pairs.

    Parameters
    ----------
    gold : list of dict
        Items with ``action`` and ``action_input`` (a JSON string).

    Returns
    -------
    list of (str, dict)
        Gold calls in order.
    """
    return [(g["action"], json.loads(g["action_input"] or "{}")) for g in gold]


def tool_match_merged(response: str, gold: List[Dict[str, str]]) -> bool:
    """Paper-style Tool Use score: same action multiset and same merged arguments.

    Parameters
    ----------
    response : str
        Model response.
    gold : list of dict
        Gold calls.

    Returns
    -------
    bool
        ``True`` iff ``Counter(actions)`` and the union of argument dicts match.
    """
    pred = parse_tool_calls(response)
    ref = _gold_calls(gold)
    if Counter(a for a, _ in pred) != Counter(a for a, _ in ref):
        return False
    merged_pred: Dict[str, Any] = {}
    for _, args in pred:
        merged_pred.update(args or {})
    merged_ref: Dict[str, Any] = {}
    for _, args in ref:
        merged_ref.update(args)
    return merged_pred == merged_ref


def tool_match_strict(response: str, gold: List[Dict[str, str]]) -> bool:
    """Strict Tool Use score: identical ordered list of (action, arguments).

    Parameters
    ----------
    response : str
        Model response.
    gold : list of dict
        Gold calls.

    Returns
    -------
    bool
        ``True`` iff every call matches the gold call at the same position.
    """
    pred = [(a, args or {}) for a, args in parse_tool_calls(response)]
    return pred == _gold_calls(gold)


def score(task: str, response: str, answer: Any) -> Dict[str, float]:
    """Score one response with every metric defined for ``task``.

    Parameters
    ----------
    task : str
        ``"science"`` or ``"tool_use"``.
    response : str
        Model response.
    answer : Any
        Gold label from the data loader.

    Returns
    -------
    dict
        ``{"acc": 0/1}`` for Science; ``{"acc", "acc_strict"}`` for Tool Use,
        where ``acc`` is the paper-style merged metric.
    """
    if task == "science":
        return {"acc": float(science_correct(response, answer))}
    return {
        "acc": float(tool_match_merged(response, answer)),
        "acc_strict": float(tool_match_strict(response, answer)),
    }
