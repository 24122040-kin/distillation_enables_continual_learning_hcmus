"""New-task evaluation (greedy) and comparison with the paper's Table 5.

Usage examples::

    # base model on the official test split
    python src/evaluate.py --model_name Qwen/Qwen2.5-7B-Instruct --task science --method base

    # a trained LoRA checkpoint, validation split (for checkpoint selection)
    python src/evaluate.py --checkpoint_dir checkpoints/sdft_science_s42/epoch2 --method sdft --split val

    # teacher accuracy: the model with the demonstration in context (Section 3.2 / scaling analysis)
    python src/evaluate.py --task science --split val --with_demo --method base
"""

import argparse
import json
import os
from typing import Dict, List, Optional

import torch
from tqdm import tqdm
import wandb

from answer_parsing import score
from common import chat_ids, generate_batch, load_model_and_tokenizer, set_seed
from data_loader import get_dataset, resolve_source
from prompts import student_context, teacher_context

# Table 5 of the paper (Qwen2.5-7B-Instruct): new-task accuracy and average prior-task score.
PAPER_TABLE5 = {
    "science": {
        "base": {"new": 32.1, "prior_avg": 65.5},
        "sft": {"new": 66.2, "prior_avg": 53.4},
        "sdft": {"new": 70.2, "prior_avg": 64.5},
    },
    "tool_use": {
        "base": {"new": 42.9, "prior_avg": 65.5},
        "sft": {"new": 63.2, "prior_avg": 56.0},
        "sdft": {"new": 70.6, "prior_avg": 65.4},
    },
}


def evaluate_new_task(
    model,
    tokenizer,
    task: str,
    examples: List[Dict],
    max_new_tokens: int = 1024,
    batch_size: int = 16,
    with_demo: bool = False,
) -> Dict[str, object]:
    """Greedy-decode every example and score it.

    Parameters
    ----------
    model, tokenizer
        Model (base or PEFT) and tokenizer.
    task : str
        ``"science"`` or ``"tool_use"``.
    examples : list of dict
        Examples from :func:`data_loader.get_dataset`.
    max_new_tokens : int
        Generation budget (paper: 2048, official code: 1024).
    batch_size : int
        Number of prompts decoded together.
    with_demo : bool
        Use the teacher context (requires demonstrations, i.e. the val split).

    Returns
    -------
    dict
        ``metrics`` (mean of every score key, in %, plus truncation rate and
        mean response length) and ``records`` (per-example outputs).
    """
    records = []
    for start in tqdm(range(0, len(examples), batch_size), desc=f"eval {task}"):
        chunk = examples[start:start + batch_size]
        prompts = []
        for ex in chunk:
            if with_demo:
                system, user = teacher_context(ex["system"], ex["question"], ex["demonstration"])
            else:
                system, user = student_context(ex["system"], ex["question"])
            prompts.append(chat_ids(tokenizer, system, user))
        outs = generate_batch(model, tokenizer, prompts, max_new_tokens, do_sample=False)
        for ex, out in zip(chunk, outs):
            text = tokenizer.decode(out["ids"], skip_special_tokens=True)
            rec = {"id": ex["id"], "response": text, "n_tokens": len(out["ids"]), "truncated": out["truncated"]}
            rec.update(score(task, text, ex["answer"]))
            records.append(rec)

    keys = [k for k in records[0] if k.startswith("acc")]
    metrics = {k: 100.0 * sum(r[k] for r in records) / len(records) for k in keys}
    metrics["truncated_pct"] = 100.0 * sum(r["truncated"] for r in records) / len(records)
    metrics["mean_response_tokens"] = sum(r["n_tokens"] for r in records) / len(records)
    metrics["n"] = len(records)
    return {"metrics": metrics, "records": records}


def compare_with_paper(acc: float, task: str, method: str) -> Optional[Dict[str, float]]:
    """Absolute and relative deviation from Table 5 of the paper.

    Parameters
    ----------
    acc : float
        Reproduced accuracy (%).
    task : str
        Task name.
    method : str
        ``"base"``, ``"sft"`` or ``"sdft"``.

    Returns
    -------
    dict or None
        ``paper``, ``ours``, ``delta_pp`` and ``rel_dev_pct``; ``None`` if the
        paper has no matching number.
    """
    ref = PAPER_TABLE5.get(task, {}).get(method)
    if ref is None:
        return None
    paper = ref["new"]
    return {"paper": paper, "ours": acc, "delta_pp": acc - paper, "rel_dev_pct": 100.0 * (acc - paper) / paper}


def main() -> None:
    """CLI entry point: evaluate one model/checkpoint on one split and log to W&B."""
    p = argparse.ArgumentParser(description="Evaluate Base/SFT/SDFT on the new task")
    p.add_argument("--model_name", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--checkpoint_dir", default=None, help="LoRA adapter directory; omit for the base model")
    p.add_argument("--method", default="base", choices=["base", "sft", "sdft"])
    p.add_argument("--task", default="science", choices=["science", "tool_use"])
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--with_demo", action="store_true", help="Evaluate the demonstration-conditioned teacher")
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--quant_bits", type=int, default=16, choices=[4, 8, 16])
    p.add_argument("--max_examples", type=int, default=None, help="Smoke tests only")
    p.add_argument("--data_seed", type=int, default=0)
    p.add_argument("--data_source", default="auto", choices=["auto", "hf", "github"])
    p.add_argument("--hf_dataset", default=None)
    p.add_argument("--seed", type=int, default=42, help="Seed of the evaluated training run (for naming)")
    p.add_argument("--output_dir", default="./eval_results")
    p.add_argument("--run_name", default=None)
    args = p.parse_args()

    if args.with_demo and args.split != "val":
        p.error("--with_demo needs --split val (the test split has no demonstrations).")
    set_seed(args.seed)
    _, val, test = get_dataset(
        args.task, data_seed=args.data_seed, source=args.data_source, hf_dataset=args.hf_dataset
    )
    examples = val if args.split == "val" else test
    if args.max_examples:
        examples = examples[: args.max_examples]

    model, tokenizer = load_model_and_tokenizer(args.model_name, args.quant_bits, for_training=False)
    if args.checkpoint_dir:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.checkpoint_dir)
    model.eval()

    tag = "teacher" if args.with_demo else args.method
    name = args.run_name or f"eval-{tag}-{args.model_name.split('/')[-1]}-{args.task}-{args.split}-s{args.seed}"
    wandb.init(
        project=os.environ.get("WANDB_PROJECT", "sdft-continual-learning"),
        name=name,
        config={**vars(args), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                "data_source_resolved": resolve_source(args.data_source, args.hf_dataset)},
        tags=["lab1", "eval-newtask", tag, args.task, args.split],
    )
    result = evaluate_new_task(
        model, tokenizer, args.task, examples, args.max_new_tokens, args.batch_size, args.with_demo
    )
    metrics = result["metrics"]
    log = {f"eval/{k}": v for k, v in metrics.items()}
    cmp = compare_with_paper(metrics["acc"], args.task, args.method) if args.split == "test" and not args.with_demo else None
    if cmp:
        log.update({f"paper_cmp/{k}": v for k, v in cmp.items()})
    wandb.log(log)
    wandb.log({"eval/samples": wandb.Table(
        columns=["id", "acc", "n_tokens", "truncated", "response"],
        data=[[r["id"], r["acc"], r["n_tokens"], r["truncated"], r["response"]] for r in result["records"]],
    )})

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, f"{name}.json"), "w") as f:
        json.dump({"args": vars(args), "metrics": metrics, "paper_cmp": cmp, "records": result["records"]}, f, indent=2)

    print(json.dumps({"metrics": metrics, "paper_cmp": cmp}, indent=2))
    wandb.finish()


if __name__ == "__main__":
    main()
