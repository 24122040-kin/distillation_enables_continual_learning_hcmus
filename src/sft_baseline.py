"""Off-policy Supervised Fine-Tuning (SFT) baseline for SDFT comparison.

Uses the exact same 4-bit QLoRA configuration, programmatic dataset loader,
and W&B tracking as sdft_trainer.py for a controlled comparison.
"""

import argparse
import os
import random

import numpy as np
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)
import wandb

from data_loader import get_dataset


def set_seed(seed: int) -> None:
    """Fix random seeds for reproducibility.

    Parameters
    ----------
    seed : int
        Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_sft(args: argparse.Namespace) -> None:
    """Run standard off-policy SFT and log training metrics to W&B.

    Parameters
    ----------
    args : argparse.Namespace
        Training configuration arguments.
    """
    set_seed(args.seed)

    wandb.init(
        project=os.environ.get("WANDB_PROJECT", "sdft-continual-learning"),
        name=f"sft-{args.task}-s{args.seed}",
        config=vars(args),
        tags=["reproduction", "sft", args.task],
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    base_model = prepare_model_for_kbit_training(base_model)

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(base_model, lora_cfg)

    train_data, _, _ = get_dataset(args.task, seed=args.seed, max_train=args.max_train)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    total_steps = (len(train_data) // args.grad_accum) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=min(10, max(1, total_steps // 10)), num_training_steps=max(1, total_steps)
    )

    global_step = 0
    optimizer.zero_grad()
    model.train()

    for epoch in range(args.epochs):
        random.shuffle(train_data)
        pbar = tqdm(enumerate(train_data), total=len(train_data), desc=f"SFT Epoch {epoch + 1}/{args.epochs}")

        for idx, ex in pbar:
            prompt_str = tokenizer.apply_chat_template(
                [{"role": "user", "content": ex["prompt"]}],
                tokenize=False,
                add_generation_prompt=True,
            )
            full_str = prompt_str + ex["demonstration"] + tokenizer.eos_token

            prompt_ids = tokenizer(prompt_str, return_tensors="pt", truncation=True, max_length=1024)["input_ids"]
            full_inputs = tokenizer(full_str, return_tensors="pt", truncation=True, max_length=1280).to(model.device)

            labels = full_inputs["input_ids"].clone()
            prompt_len = min(prompt_ids.shape[1], labels.shape[1])
            labels[:, :prompt_len] = -100

            loss = model(**full_inputs, labels=labels).loss
            (loss / args.grad_accum).backward()

            if (idx + 1) % args.grad_accum == 0 or (idx + 1) == len(train_data):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                wandb.log(
                    {
                        "train/sft_loss": loss.item(),
                        "train/lr": scheduler.get_last_lr()[0],
                        "epoch": epoch + (idx + 1) / len(train_data),
                    },
                    step=global_step,
                )
                pbar.set_postfix({"sft_loss": f"{loss.item():.4f}"})

    save_dir = os.path.join(args.output_dir, f"sft_{args.task}_s{args.seed}")
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SFT Baseline Trainer (QLoRA)")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--task", type=str, default="science_qa", choices=["science_qa", "tool_use"])
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--max_train", type=int, default=200)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    train_sft(parser.parse_args())
