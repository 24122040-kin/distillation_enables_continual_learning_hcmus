"""Off-policy SFT baseline.

Same data, subset, LoRA configuration, batch size and logging as
``sdft_trainer.py``, so that the only difference between the two is the
training signal: SFT maximises log pi_theta(c | x) on the expert demonstration c,
whereas SDFT distils the demonstration-conditioned teacher on the student's own
rollouts.
"""

import argparse
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import get_cosine_schedule_with_warmup
import wandb

from common import chat_ids, load_model_and_tokenizer, set_seed
from data_loader import get_dataset, resolve_source
from prompts import student_context
from sdft_trainer import LORA_TARGETS


def sft_example_loss(model, tokenizer, example: dict, max_target_tokens: int) -> torch.Tensor:
    """Mean token negative log-likelihood of the demonstration given the task prompt.

    Parameters
    ----------
    model, tokenizer
        PEFT model and tokenizer.
    example : dict
        Training example with ``system``, ``question`` and ``demonstration``.
    max_target_tokens : int
        Truncate demonstrations longer than this (kept equal to SDFT's generation budget).

    Returns
    -------
    torch.Tensor
        Scalar loss with gradient.
    """
    device = next(model.parameters()).device
    prompt = chat_ids(tokenizer, *student_context(example["system"], example["question"]))
    target = tokenizer(example["demonstration"], add_special_tokens=False)["input_ids"][: max_target_tokens - 1]
    target = target + [tokenizer.eos_token_id]  # <|im_end|> for Qwen chat models
    ids = torch.tensor(prompt + target, device=device)[None]
    logits = model(input_ids=ids).logits[0, len(prompt) - 1:-1]
    return F.cross_entropy(logits.float(), torch.tensor(target, device=device))


def train_sft(args: argparse.Namespace) -> None:
    """Run SFT with the same loop structure as SDFT and log to W&B.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line configuration.
    """
    set_seed(args.seed)
    model_short = args.model_name.split("/")[-1]
    n = args.max_train if args.max_train else "all"
    out_dir = os.path.join(args.output_dir, f"sft_{args.task}_{model_short}_n{n}_s{args.seed}")

    train, _, _ = get_dataset(
        args.task, max_train=args.max_train, data_seed=args.data_seed,
        source=args.data_source, hf_dataset=args.hf_dataset,
    )
    model, tokenizer = load_model_and_tokenizer(args.model_name, args.quant_bits, for_training=True)
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM", target_modules=LORA_TARGETS,
    ))
    params = [p for p in model.parameters() if p.requires_grad]
    steps_per_epoch = math.ceil(len(train) / args.batch_size)
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, steps_per_epoch * args.epochs)

    wandb.init(
        project=os.environ.get("WANDB_PROJECT", "sdft-continual-learning"),
        name=args.run_name or os.path.basename(out_dir),
        config={**vars(args), "n_train": len(train),
                "data_source_resolved": resolve_source(args.data_source, args.hf_dataset),
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"},
        tags=["lab1", "sft", args.task, model_short] + args.tags,
    )
    model.train()
    global_step, start = 0, time.time()
    for epoch in range(args.epochs):
        order = list(range(len(train)))
        random.Random(args.seed * 1000 + epoch).shuffle(order)
        for step in range(steps_per_epoch):
            t0 = time.time()
            batch = [train[i] for i in order[step * args.batch_size:(step + 1) * args.batch_size]]
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for ex in batch:
                loss = sft_example_loss(model, tokenizer, ex, args.max_target_tokens)
                (loss / len(batch)).backward()
                losses.append(loss.item())
            grad_norm = torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            global_step += 1
            wandb.log({
                "train/sft_loss": float(np.mean(losses)),
                "train/grad_norm": float(grad_norm),
                "train/lr": scheduler.get_last_lr()[0],
                "time/step_sec": time.time() - t0,
                "time/elapsed_h": (time.time() - start) / 3600,
                "epoch": epoch + (step + 1) / steps_per_epoch,
            }, step=global_step)
            print(f"[ep {epoch + 1} step {step + 1}/{steps_per_epoch}] nll={np.mean(losses):.4f}", flush=True)
        model.save_pretrained(os.path.join(out_dir, f"epoch{epoch + 1}"))
        tokenizer.save_pretrained(os.path.join(out_dir, f"epoch{epoch + 1}"))
    wandb.finish()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="SFT baseline (LoRA / QLoRA)")
    p.add_argument("--model_name", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--task", default="science", choices=["science", "tool_use"])
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup_steps", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--max_target_tokens", type=int, default=1024)
    p.add_argument("--max_train", type=int, default=None)
    p.add_argument("--data_seed", type=int, default=0)
    p.add_argument("--data_source", default="auto", choices=["auto", "hf", "github"])
    p.add_argument("--hf_dataset", default=None)
    p.add_argument("--quant_bits", type=int, default=16, choices=[4, 8, 16])
    p.add_argument("--lora_r", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", default="./checkpoints")
    p.add_argument("--run_name", default=None)
    p.add_argument("--tags", nargs="*", default=[])
    train_sft(p.parse_args())
