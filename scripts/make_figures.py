"""Slide figures from the evaluation JSONs (same inputs as summarize_results.py).

Usage::

    python scripts/make_figures.py results/* --out presentation/assets

Writes PNGs (16:9-friendly, 200 dpi, Vietnamese labels). Evaluations that are not available yet are drawn as
hatched placeholders labelled "đang chạy", so re-running the script after new results only updates the bars.
"""

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from summarize_results import (BENCHES, PAPER_NEW, PAPER_PRIOR, ci95, load_new_task,  # noqa: E402
                               load_prior, paired)

# Reference palette (dataviz skill), light mode.
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
OURS, PAPER, SFT, SDFT, BASE, TEACH = "#2a78d6", "#b5b3ab", "#eb6834", "#2a78d6", "#8f8d85", "#1baf7a"
METHOD_VI = {"base": "Base", "sft": "SFT", "sdft": "SDFT"}

plt.rcParams.update({
    "font.family": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 13,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
})


def _bar(ax, x, h, w, color, **kw):
    return ax.bar(x, h, w, color=color, linewidth=0, zorder=3, **kw)


def _pending(ax, x, w, top=8):
    ax.bar(x, top, w, color="none", edgecolor=PAPER, hatch="///", linewidth=1, zorder=3)
    ax.text(x, top + 1.5, "đang chạy", ha="center", va="bottom", fontsize=10, color=INK2, style="italic")


def _style(ax, ymax, ylabel="Độ chính xác (%)"):
    ax.set_ylim(0, ymax)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_ylabel(ylabel)
    ax.tick_params(length=0)


def fig_new_task(new, out):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), gridspec_kw={"wspace": 0.18})
    for ax, (task, title, n) in zip(axes, (("science", "Science Q&A (n = 507)", 507), ("tool_use", "Tool Use (n = 97)", 97))):
        w = 0.36
        for i, m in enumerate(("base", "sft", "sdft")):
            p = PAPER_NEW[(task, m)]
            _bar(ax, i - w / 2, p, w, PAPER)
            ax.text(i - w / 2, p + 1.2, f"{p:.1f}", ha="center", va="bottom", fontsize=11, color=INK2)
            r = new.get(("7B", task, m, "test"))
            if r is None:
                _pending(ax, i + w / 2, w)
                continue
            _bar(ax, i + w / 2, r["acc"], w, OURS)
            ci = ci95(r["acc"], r["n"])
            ax.errorbar(i + w / 2, r["acc"], yerr=ci, fmt="none", ecolor=INK, elinewidth=1.2, capsize=4, zorder=4)
            ax.text(i + w / 2, r["acc"] + ci + 1.2, f"{r['acc']:.1f}", ha="center", va="bottom", fontsize=12,
                    color=INK, fontweight="bold")
        ax.set_xticks(range(3), [METHOD_VI[m] for m in ("base", "sft", "sdft")], fontsize=13, color=INK)
        ax.set_title(title, loc="left", fontsize=14, color=INK, fontweight="bold")
        _style(ax, 85)
    axes[1].set_ylabel("")
    fig.legend(handles=[Patch(color=PAPER, label="Paper (Table 5, full FT, H200)"),
                        Patch(color=OURS, label="Nhóm (QLoRA 4-bit, T4, ~13% dữ liệu) ± CI 95%")],
               loc="upper center", ncol=2, frameon=False, fontsize=12, bbox_to_anchor=(0.5, 1.02))
    fig.subplots_adjust(top=0.82, bottom=0.1, left=0.07, right=0.99)
    fig.savefig(os.path.join(out, "new_task_accuracy.png"), dpi=200)
    plt.close(fig)


def fig_paired(new, out):
    rows = [("Science 7B: SDFT − SFT", "science", "sft", "sdft", 70.2 - 66.2),
            ("Science 7B: SDFT − Base", "science", "base", "sdft", 70.2 - 32.1),
            ("Science 7B: SFT − Base", "science", "base", "sft", 66.2 - 32.1),
            ("Tool Use 7B: SFT − Base", "tool_use", "base", "sft", 63.2 - 42.9),
            ("Tool Use 7B: SDFT − SFT", "tool_use", "sft", "sdft", 70.6 - 63.2)]
    fig, ax = plt.subplots(figsize=(11, 4.6))
    labels = []
    for k, (label, task, a, b, paper_d) in enumerate(rows):
        y = len(rows) - 1 - k
        labels.append((y, label))
        ra, rb = new.get(("7B", task, a, "test")), new.get(("7B", task, b, "test"))
        ax.scatter([paper_d], [y], marker="D", s=70, color=PAPER, zorder=3)
        res = paired(ra, rb) if ra and rb else None
        if res is None:
            ax.text(0.5, y, "đang chạy", va="center", fontsize=11, color=INK2, style="italic")
            continue
        d, ci, _, p, _ = [s.strip() for s in res.split("|")]
        lo, hi = [float(v) for v in ci.strip("[]").split(",")]
        ax.plot([lo, hi], [y, y], color=OURS, linewidth=2.5, solid_capstyle="round", zorder=3)
        ax.scatter([float(d)], [y], s=90, color=OURS, edgecolor=SURFACE, linewidth=2, zorder=4)
        ptxt = "p < 0.001" if float(p) < 0.001 else f"p = {p}"
        ax.text(hi + 1, y, f"{float(d):+.1f} pp  ({ptxt})", va="center", fontsize=11, color=INK)
    ax.axvline(0, color=INK2, linewidth=1, zorder=2)
    ax.set_yticks([y for y, _ in labels], [l for _, l in labels], color=INK, fontsize=12)
    ax.set_xlabel("Chênh lệch độ chính xác (điểm %), cùng bộ câu hỏi test")
    ax.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_xlim(-5, 45)
    ax.tick_params(length=0)
    ax.legend(handles=[plt.Line2D([], [], color=OURS, marker="o", linewidth=2.5, label="Nhóm: Δ và CI 95% (bootstrap theo cặp)"),
                       plt.Line2D([], [], color=PAPER, marker="D", linestyle="none", markersize=8, label="Paper (Table 5)")],
              loc="lower right", frameon=False, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "paired_differences.png"), dpi=200)
    plt.close(fig)


def fig_model_size(new, out):
    fig, ax = plt.subplots(figsize=(10, 5))
    series = (("base", "Base (test)", BASE, "test"), ("sft", "SFT (test)", SFT, "test"),
              ("sdft", "SDFT (test)", SDFT, "test"), ("teacher", "Teacher có lời giải mẫu (val)", TEACH, "val"))
    w = 0.2
    for j, size in enumerate(("3B", "7B")):
        for i, (m, _, color, split) in enumerate(series):
            x = j + (i - 1.5) * (w + 0.02)
            r = new.get((size, "science", m, split))
            if r is None:
                _pending(ax, x, w)
                continue
            _bar(ax, x, r["acc"], w, color)
            ax.text(x, r["acc"] + 1.2, f"{r['acc']:.1f}", ha="center", va="bottom", fontsize=11, color=INK)
        sft, sd = new.get((size, "science", "sft", "test")), new.get((size, "science", "sdft", "test"))
        paper_gap = {"3B": "−3.3", "7B": "+4.0"}[size]
        ours_gap = f"{sd['acc'] - sft['acc']:+.1f}" if sft and sd else "?"
        ax.text(j, -13, f"SDFT − SFT: nhóm {ours_gap} pp | paper {paper_gap} pp", ha="center", fontsize=11, color=INK2)
    ax.set_xticks([0, 1], ["Qwen2.5-3B-Instruct", "Qwen2.5-7B-Instruct"], fontsize=13, color=INK)
    _style(ax, 100)
    ax.legend(handles=[Patch(color=c, label=l) for _, l, c, _ in series], loc="upper center", ncol=4, frameon=False,
              fontsize=11, bbox_to_anchor=(0.5, 1.12))
    fig.subplots_adjust(bottom=0.2, left=0.08, right=0.99, top=0.88)
    fig.savefig(os.path.join(out, "model_size_e3.png"), dpi=200)
    plt.close(fig)


def fig_prior(prior, out):
    rows = (("base", "Base 7B", BASE), ("sft_science", "SFT (Science)", SFT), ("sdft_science", "SDFT (Science)", SDFT))
    names = {"hellaswag": "HellaSwag", "humaneval": "HumanEval", "ifeval": "IFEval", "mmlu": "MMLU",
             "truthfulqa_mc2": "TruthfulQA", "winogrande": "Winogrande", "avg": "Trung bình"}
    cols = BENCHES + ["avg"]
    fig, ax = plt.subplots(figsize=(12, 5))
    w = 0.26
    for i, (key, label, color) in enumerate(rows):
        r = prior.get(key, {})
        for j, b in enumerate(cols):
            x = j + (i - 1) * (w + 0.02)
            if b not in r:
                ax.bar(x, 6, w, color="none", edgecolor=PAPER, hatch="///", linewidth=1, zorder=3)
                continue
            _bar(ax, x, r[b], w, color)
            ax.text(x, r[b] - 2, f"{r[b]:.0f}", ha="center", va="top", fontsize=9, color="white", fontweight="bold", zorder=6)
            pv = PAPER_PRIOR.get(key, {}).get(b)
            if pv is not None:
                ax.plot([x - w / 2, x + w / 2], [pv, pv], color=INK, linewidth=1.6, zorder=5)
    ax.set_xticks(range(len(cols)), [names[c] for c in cols], color=INK)
    _style(ax, 100, "Điểm (%)")
    ax.legend(handles=[Patch(color=c, label=l) for _, l, c in rows] +
              [plt.Line2D([], [], color=INK, linewidth=1.6, label="Paper (Table 5)"),
               Patch(facecolor="none", edgecolor=PAPER, hatch="///", label="đang chạy")],
              loc="upper center", ncol=5, frameon=False, fontsize=11, bbox_to_anchor=(0.5, 1.1))
    fig.subplots_adjust(top=0.88, bottom=0.08, left=0.07, right=0.99)
    fig.savefig(os.path.join(out, "prior_capabilities.png"), dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", default="presentation/assets")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    new, prior = load_new_task(args.dirs), load_prior(args.dirs)
    fig_new_task(new, args.out)
    fig_paired(new, args.out)
    fig_model_size(new, args.out)
    fig_prior(prior, args.out)
    print("wrote", sorted(os.listdir(args.out)))


if __name__ == "__main__":
    main()
