"""Evaluation script for New Task Accuracy, Prior Capabilities, and Paper Deviation.

Computes greedy (temperature=0) accuracy on the target task, optionally invokes
EleutherAI lm-evaluation-harness on prior capability benchmarks, and reports
Absolute Difference (Delta) and Relative Error (%) vs. Table 5 of the paper.
"""

import argparse
import os
import re
from typing import Dict, Optional

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import wandb

from data_loader import get_dataset


PAPER_TABLE5 = {
    "science_qa": {"base": 32.1, "sft": 66.2, "sdft": 70.2},
    "tool_use": {"base": 42.9, "sft": 63.2, "sdft": 70.6},
}


def extract_choice(text: str) -> str:
    """Extract multiple-choice option letter (A-D) from model output.

    Parameters
    ----------
    text : str
        Generated model response.

    Returns
    -------
    str
        Extracted uppercase option letter, or empty string if not found.
    """
    match = re.search(r"(?:Final Answer:\s*|Answer:\s*|\b)([A-D])\b", text.strip(), re.IGNORECASE)
    return match.group(1).upper() if match else ""


def evaluate_new_task(
    model_name: str,
    checkpoint_dir: Optional[str],
    task: str,
    seed: int = 42,
    max_test: int = 100,
    max_new_tokens: int = 384,
    quant_bits: int = 4,
) -> float:
    """Evaluate a model or LoRA checkpoint on the held-out test set using greedy decoding.

    Parameters
    ----------
    model_name : str
        Base Hugging Face model ID.
    checkpoint_dir : str or None
        Path to saved PEFT LoRA checkpoint, or None to evaluate the Base model.
    task : str
        Task name ('science_qa' or 'tool_use').
    seed : int
        Random seed matching the dataset split.
    max_test : int
        Maximum number of test examples to evaluate.
    max_new_tokens : int
        Maximum generation length.

    Returns
    -------
    float
        Accuracy percentage in [0.0, 100.0].
    """
    if quant_bits == 4:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    elif quant_bits == 8:
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    else:
        bnb_config = None

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {"device_map": "auto", "trust_remote_code": True}
    if bnb_config is not None:
        model_kwargs["quantization_config"] = bnb_config
    else:
        model_kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    if checkpoint_dir:
        model = PeftModel.from_pretrained(model, checkpoint_dir)
    model.eval()

    _, _, test_data = get_dataset(task, seed=seed, max_test=max_test)
    correct = 0

    for ex in tqdm(test_data, desc=f"Evaluating {checkpoint_dir or 'Base'} on {task}"):
        prompt_str = tokenizer.apply_chat_template(
            [{"role": "user", "content": ex["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(prompt_str, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        pred_text = tokenizer.decode(out[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)

        if task == "science_qa":
            pred = extract_choice(pred_text)
            gold = ex["answer"].strip().upper()[:1]
            if pred == gold:
                correct += 1
        else:
            # Normalized substring / API call match for Tool Use
            gold_norm = re.sub(r"\s+", "", ex["answer"].lower())
            pred_norm = re.sub(r"\s+", "", pred_text.lower())
            if gold_norm and gold_norm in pred_norm:
                correct += 1

    acc = 100.0 * correct / max(1, len(test_data))
    return acc


def compare_with_paper(acc: float, task: str, method: str) -> Dict[str, float]:
    """Compute Absolute Difference and Relative Error against Table 5 of the paper.

    Parameters
    ----------
    acc : float
        Reproduced accuracy percentage.
    task : str
        Task name ('science_qa' or 'tool_use').
    method : str
        One of 'base', 'sft', 'sdft'.

    Returns
    -------
    dict
        Metrics dictionary with paper_acc, delta, rel_error_pct, and fidelity_pct.
    """
    paper_acc = PAPER_TABLE5[task][method]
    delta = acc - paper_acc
    rel_err = abs(delta) / paper_acc * 100.0
    fidelity = max(0.0, 100.0 - rel_err)
    return {
        "paper_acc": paper_acc,
        "reproduced_acc": acc,
        "delta": delta,
        "rel_error_pct": rel_err,
        "fidelity_pct": fidelity,
    }


def main() -> None:
    """CLI entrypoint for evaluating a model checkpoint and logging comparison to W&B."""
    parser = argparse.ArgumentParser(description="Evaluate SDFT/SFT/Base and compare with paper")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--method", type=str, default="base", choices=["base", "sft", "sdft"])
    parser.add_argument("--task", type=str, default="science_qa", choices=["science_qa", "tool_use"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_test", type=int, default=100)
    parser.add_argument("--quant_bits", type=int, default=4, choices=[4, 8, 16])
    args = parser.parse_args()

    wandb.init(
        project=os.environ.get("WANDB_PROJECT", "sdft-continual-learning"),
        name=f"eval-{args.method}-{args.task}-s{args.seed}",
        config=vars(args),
        tags=["evaluation", args.method, args.task],
    )

    acc = evaluate_new_task(
        model_name=args.model_name,
        checkpoint_dir=args.checkpoint_dir,
        task=args.task,
        seed=args.seed,
        max_test=args.max_test,
        quant_bits=args.quant_bits,
    )
    metrics = compare_with_paper(acc, args.task, args.method)
    wandb.log({f"eval/{k}": v for k, v in metrics.items()})

    print("\n=== Reproduction Comparison vs. Paper (Table 5) ===")
    print(f"Task: {args.task} | Method: {args.method}")
    print(f"Paper Accuracy     : {metrics['paper_acc']:.2f}%")
    print(f"Reproduced Accuracy: {metrics['reproduced_acc']:.2f}%")
    print(f"Delta (Reprod-Pap) : {metrics['delta']:+.2f} pp")
    print(f"Relative Error     : {metrics['rel_error_pct']:.2f}%")
    print(f"Fidelity Score     : {metrics['fidelity_pct']:.2f}%")
    wandb.finish()


if __name__ == "__main__":
    main()
