"""Collect every evaluation result into the Markdown tables of the report (slides 22-24).

Reads, from one or more result folders (one per Kaggle account / notebook output):

- ``*.json`` written by ``src/evaluate.py`` (new-task accuracy, test/val, teacher);
- ``prior/<run>/**/results_*.json`` written by ``lm_eval`` (prior capabilities).

When the same evaluation appears in several folders, the file listed last wins.

Usage::

    python scripts/summarize_results.py results_kin/eval_results results_b/eval_results > docs/results.md
"""

import argparse
import glob
import json
import math
import os
import random
import re
from typing import Dict, List, Optional, Tuple

PAPER_NEW = {  # Table 5, Qwen2.5-7B-Instruct
    ("science", "base"): 32.1, ("science", "sft"): 66.2, ("science", "sdft"): 70.2,
    ("tool_use", "base"): 42.9, ("tool_use", "sft"): 63.2, ("tool_use", "sdft"): 70.6,
}
PAPER_PRIOR = {  # Table 5, per benchmark (Science rows) and average
    "base": {"hellaswag": 62.0, "humaneval": 65.8, "ifeval": 74.3, "mmlu": 71.7,
             "truthfulqa_mc2": 47.9, "winogrande": 71.1, "avg": 65.5},
    "sft_science": {"hellaswag": 55.0, "humaneval": 54.8, "ifeval": 35.3, "mmlu": 64.6,
                    "truthfulqa_mc2": 36.8, "winogrande": 73.7, "avg": 53.4},
    "sdft_science": {"hellaswag": 60.9, "humaneval": 68.9, "ifeval": 66.8, "mmlu": 70.7,
                     "truthfulqa_mc2": 46.5, "winogrande": 73.1, "avg": 64.5},
    "sft_tool_use": {"avg": 56.0},
    "sdft_tool_use": {"avg": 65.4},
}
PRIOR_METRIC = {  # lm_eval task -> metric key reported in the slides
    "hellaswag": "acc_norm,none",
    "humaneval": "pass@1,create_test",
    "ifeval": "prompt_level_strict_acc,none",
    "mmlu": "acc,none",
    "truthfulqa_mc2": "acc,none",
    "winogrande": "acc,none",
}
BENCHES = list(PRIOR_METRIC)


def size_of(model_name: str) -> str:
    """``Qwen/Qwen2.5-7B-Instruct`` -> ``7B``."""
    m = re.search(r"(\d+(?:\.\d+)?B)", model_name)
    return m.group(1) if m else model_name


def epoch_of(path: Optional[str]) -> str:
    """Epoch directory name of a checkpoint (``epoch1``), or ``-`` for the base model."""
    return os.path.basename(path.rstrip("/")) if path else "-"


def ci95(acc: float, n: int) -> float:
    """Half-width of the normal-approximation 95% interval, in percentage points."""
    p = acc / 100.0
    return 196.0 * math.sqrt(p * (1 - p) / n) if n else float("nan")


def load_new_task(dirs: List[str]) -> Dict[Tuple[str, str, str, str], Dict]:
    """Key ``(size, task, method_or_teacher, split)`` -> record; later folders override earlier ones."""
    out = {}
    for d in dirs:
        for f in sorted(glob.glob(os.path.join(d, "*.json"))):
            try:
                r = json.load(open(f))
                a, m = r["args"], r["metrics"]
            except (KeyError, ValueError, TypeError):
                continue
            tag = "teacher" if a.get("with_demo") else a["method"]
            out[(size_of(a["model_name"]), a["task"], tag, a["split"])] = {
                "acc": m["acc"], "n": m["n"], "truncated_pct": m.get("truncated_pct"),
                "epoch": epoch_of(a.get("checkpoint_dir")), "file": os.path.basename(f),
                "run": a.get("run_name") or os.path.splitext(os.path.basename(f))[0],
                "per_id": {r["id"]: r["acc"] for r in r.get("records", [])},
            }
    return out


def prior_key(run: str) -> str:
    """``sft_science_Qwen2.5-7B-..._s42`` -> ``sft_science``; ``base-7b`` -> ``base``."""
    if run.startswith("base"):
        return "base"
    m = re.match(r"(sft|sdft)_(science|tool_use)_", run)
    return f"{m.group(1)}_{m.group(2)}" if m else run


def load_prior(dirs: List[str]) -> Dict[str, Dict]:
    """Key ``prior_key`` (7B runs only) -> per-benchmark scores in % plus the average.

    Several lm-eval runs can feed one row (e.g. ``base-7b`` and ``base-7b-humaneval``); a later
    result overrides an earlier one. ``humaneval_instruct`` (chat variant) fills the HumanEval column
    and replaces plain ``humaneval``, which scores 0 with a chat template.
    """
    out: Dict[str, Dict] = {}
    for d in dirs:
        for run_dir in sorted(glob.glob(os.path.join(d, "prior", "*"))):
            run = os.path.basename(run_dir)
            if "3B" in run or "1.5B" in run:
                continue
            files = sorted(glob.glob(os.path.join(run_dir, "**", "results_*.json"), recursive=True))
            if not files:
                continue
            res = json.load(open(files[-1]))["results"]
            row = out.setdefault(prior_key(run), {"runs": [], "he_instruct": False})
            for b, key in PRIOR_METRIC.items():
                if b == "humaneval" and "humaneval_instruct" in res:
                    row["humaneval"] = 100.0 * res["humaneval_instruct"][key]
                    row["he_instruct"] = True
                elif b in res and res[b].get(key) is not None and not (b == "humaneval" and row["he_instruct"]):
                    row[b] = 100.0 * res[b][key]
            row["runs"].append(f"prior-{run}")
    for row in out.values():
        if all(b in row for b in BENCHES):
            row["avg"] = sum(row[b] for b in BENCHES) / len(BENCHES)
        if "humaneval" in row and not row["he_instruct"]:
            row["humaneval_note"] = "plain humaneval"
        row["run"] = ", ".join(row["runs"])
    return out


def paired(a: Dict, b: Dict, n_boot: int = 5000) -> Optional[str]:
    """Paired comparison of two evaluations on the same questions: difference ``b - a`` (pp), exact
    McNemar p-value and a paired-bootstrap 95% interval (both runs answered the same items)."""
    ids = sorted(set(a["per_id"]) & set(b["per_id"]))
    if not ids:
        return None
    x, y = a["per_id"], b["per_id"]
    only_b = sum(1 for i in ids if y[i] and not x[i])
    only_a = sum(1 for i in ids if x[i] and not y[i])
    n, k = only_a + only_b, min(only_a, only_b)
    p = min(1.0, 2 * sum(math.comb(n, j) for j in range(k + 1)) / 2 ** n) if n else 1.0
    rng = random.Random(0)
    diffs = sorted(100.0 * sum(y[i] - x[i] for i in s) / len(ids)
                   for s in ([rng.choice(ids) for _ in ids] for _ in range(n_boot)))
    d = 100.0 * sum(y[i] - x[i] for i in ids) / len(ids)
    lo, hi = diffs[int(0.025 * n_boot)], diffs[int(0.975 * n_boot) - 1]
    return f"{d:+.1f} | [{lo:+.1f}, {hi:+.1f}] | {only_b} / {only_a} | {p:.3f} | {len(ids)}"


def table_paired(new: Dict) -> List[str]:
    lines = ["## Paired tests on the same test questions", "",
             "| Comparison (B − A) | Δ (pp) | Paired bootstrap 95% CI | only B / only A correct | McNemar p | n |",
             "|---|---|---|---|---|---|"]
    for s in ("3B", "7B"):
        for task in ("science", "tool_use"):
            for a, b in (("base", "sft"), ("base", "sdft"), ("sft", "sdft")):
                ra, rb = new.get((s, task, a, "test")), new.get((s, task, b, "test"))
                res = paired(ra, rb) if ra and rb else None
                if res:
                    lines.append(f"| {s} {task}: {b.upper()} − {a.upper() if a != 'base' else 'Base'} | {res} |")
    return lines


def fmt(x: Optional[float], nd: int = 1) -> str:
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def table_new(new: Dict) -> List[str]:
    lines = ["## Slide 22: new-task accuracy, 7B (test split)", "",
             "| Task / Method | Paper (%) | Ours (%) | 95% CI (pp) | Δ (pp) | Rel. dev. (%) | Epoch | n | Truncated (%) | W&B run |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for task, label in (("science", "Science"), ("tool_use", "Tool Use")):
        for meth in ("base", "sft", "sdft"):
            paper = PAPER_NEW[(task, meth)]
            r = new.get(("7B", task, meth, "test"))
            if r is None:
                lines.append(f"| {label}: {meth.upper() if meth != 'base' else 'Base'} | {paper} | not run | | | | | | | |")
                continue
            d = r["acc"] - paper
            lines.append(
                f"| {label}: {meth.upper() if meth != 'base' else 'Base'} | {paper} | {fmt(r['acc'])} | ±{fmt(ci95(r['acc'], r['n']))} "
                f"| {d:+.1f} | {100 * abs(d) / paper:.1f} | {r['epoch']} | {r['n']} | {fmt(r['truncated_pct'])} | `{r['run']}` |")
    return lines


def table_prior(prior: Dict) -> List[str]:
    lines = ["## Slide 23: prior capabilities, 7B (lm-eval, limit 0.25; Paper → Ours)", "",
             "| Model | " + " | ".join(BENCHES) + " | Avg | W&B run |",
             "|---|" + "---|" * (len(BENCHES) + 2)]
    labels = [("base", "Base 7B"), ("sft_science", "SFT (Science)"), ("sdft_science", "SDFT (Science)"),
              ("sft_tool_use", "SFT (Tool Use)"), ("sdft_tool_use", "SDFT (Tool Use)")]
    for key, label in labels:
        p, o = PAPER_PRIOR.get(key, {}), prior.get(key)
        cells = []
        for b in BENCHES + ["avg"]:
            ours = fmt(o.get(b)) if o else "not run"
            cells.append(f"{fmt(p.get(b)) if b in p else 'T5'} → {ours}")
        lines.append(f"| {label} | " + " | ".join(cells) + f" | {'`' + o['run'] + '`' if o else ''} |")
    return lines


def table_scale(new: Dict) -> List[str]:
    lines = ["## Slide 24: model size on Science (E3)", "",
             "| Model | Base (test) | Teacher (val, with demo) | SFT (test) | SDFT (test) | SDFT − SFT (pp) | Paper (pp) |",
             "|---|---|---|---|---|---|---|"]
    paper_gap = {"3B": "−3.3", "7B": "+4.0"}
    sizes = sorted({k[0] for k in new if k[1] == "science"}, key=lambda s: float(s[:-1]))
    for s in sizes:
        g = lambda tag, split: new.get((s, "science", tag, split))
        base, teach, sft, sdft = g("base", "test"), g("teacher", "val"), g("sft", "test"), g("sdft", "test")
        cell = lambda r: f"{fmt(r['acc'])} ({r['epoch']})" if r and r["epoch"] != "-" else (fmt(r["acc"]) if r else "—")
        gap = f"{sdft['acc'] - sft['acc']:+.1f}" if sdft and sft else "—"
        lines.append(f"| Qwen2.5-{s} | {cell(base)} | {cell(teach)} | {cell(sft)} | {cell(sdft)} | {gap} | {paper_gap.get(s, '—')} |")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dirs", nargs="+", help="eval_results folders; later ones override earlier ones")
    args = ap.parse_args()
    new, prior = load_new_task(args.dirs), load_prior(args.dirs)
    out = table_new(new) + [""] + table_paired(new) + [""] + table_prior(prior) + [""] + table_scale(new)
    out += ["", "## Every new-task evaluation found", "", "| Model | Task | Method | Split | Epoch | Acc (%) | n | File |",
            "|---|---|---|---|---|---|---|---|"]
    for (s, t, m, sp), r in sorted(new.items()):
        out.append(f"| {s} | {t} | {m} | {sp} | {r['epoch']} | {fmt(r['acc'])} | {r['n']} | `{r['file']}` |")
    print("\n".join(out))


if __name__ == "__main__":
    main()
