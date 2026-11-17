"""Independent re-implementation of Self-Distillation Fine-Tuning (SDFT).

Implements Algorithm 1, Section 3, and Appendix A.1 from:
'Self-Distillation Enables Continual Learning' (Shenfeld et al., 2026)
adapted for free-tier GPUs (16-32 GB VRAM) using 4-bit QLoRA and dual PEFT adapters
('student' and 'teacher' EMA).
"""

import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
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
    """Fix all random seeds for reproducibility.

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


def build_teacher_prompt(question: str, demonstration: str) -> str:
    """Construct the demonstration-conditioned teacher prompt from Section 3.

    Parameters
    ----------
    question : str
        Task prompt x.
    demonstration : str
        Expert demonstration c.

    Returns
    -------
    str
        Formatted teacher context Ctx_T(x, c).
    """
    return (
        f"{question}\n"
        "This is an example for a response to the question:\n"
        f"{demonstration}\n"
        "Now answer with a response of your own, including the thinking process:"
    )


def compute_analytic_kl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: torch.Tensor,
    kl_mode: str = "forward",
) -> torch.Tensor:
    """Compute the full analytic per-token KL divergence over the vocabulary (Appendix A.1).

    Parameters
    ----------
    student_logits : torch.Tensor
        Logits from the student policy, shape (B, T, V).
    teacher_logits : torch.Tensor
        Logits from the teacher policy (detached), shape (B, T, V).
    mask : torch.Tensor
        Binary mask over generated response tokens, shape (B, T).
    kl_mode : str
        'forward' for KL(teacher || student) as used in practice, or
        'reverse' for KL(student || teacher) as in Eq. 1.

    Returns
    -------
    torch.Tensor
        Scalar masked mean KL loss.
    """
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)

    if kl_mode == "forward":
        teacher_probs = teacher_log_probs.exp()
        token_kl = (teacher_probs * (teacher_log_probs - student_log_probs)).sum(dim=-1)
    elif kl_mode == "reverse":
        student_probs = student_log_probs.exp()
        token_kl = (student_probs * (student_log_probs - teacher_log_probs)).sum(dim=-1)
    else:
        raise ValueError(f"Unsupported kl_mode: {kl_mode}")

    denom = mask.sum().clamp(min=1.0)
    return (token_kl * mask).sum() / denom


@torch.no_grad()
def update_ema_adapter(model: torch.nn.Module, alpha: float) -> None:
    """Update the 'teacher' LoRA adapter as an EMA of the 'student' adapter (Algorithm 1, line 19).

    Parameters
    ----------
    model : torch.nn.Module
        PEFT model containing both 'default' (student) and 'teacher' adapters.
    alpha : float
        EMA decay rate: phi <- alpha * theta + (1 - alpha) * phi.
    """
    if alpha <= 0.0:
        return
    params = dict(model.named_parameters())
    for name, student_param in params.items():
        if ".default." in name:
            teacher_name = name.replace(".default.", ".teacher.")
            if teacher_name in params:
                teacher_param = params[teacher_name]
                if alpha >= 1.0:
                    teacher_param.data.copy_(student_param.data)
                else:
                    teacher_param.data.mul_(1.0 - alpha).add_(student_param.data, alpha=alpha)


def format_chat(tokenizer: AutoTokenizer, text: str) -> str:
    """Apply the model's chat template to a user prompt.

    Parameters
    ----------
    tokenizer : AutoTokenizer
        Model tokenizer.
    text : str
        Prompt content.

    Returns
    -------
    str
        Chat-formatted string ready for tokenization.
    """
    messages = [{"role": "user", "content": text}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def train_sdft(args: argparse.Namespace) -> None:
    """Run Self-Distillation Fine-Tuning (SDFT) and log metrics to Weights & Biases.

    Parameters
    ----------
    args : argparse.Namespace
        Training configuration arguments.
    """
    set_seed(args.seed)

    wandb.init(
        project=os.environ.get("WANDB_PROJECT", "sdft-continual-learning"),
        name=f"sdft-{args.task}-ema{args.ema_alpha}-{args.kl_mode}-s{args.seed}",
        config=vars(args),
        tags=["reproduction", "sdft", args.task],
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

    model = get_peft_model(base_model, lora_cfg, adapter_name="default")
    model.add_adapter("teacher", lora_cfg)
    update_ema_adapter(model, alpha=1.0)  # Initialize phi = theta (Algorithm 1, line 1)
    model.set_adapter("default")

    train_data, _, _ = get_dataset(args.task, seed=args.seed, max_train=args.max_train)

    trainable_params = [p for n, p in model.named_parameters() if ".default." in n and p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.0)
    total_steps = (len(train_data) // args.grad_accum) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=min(10, max(1, total_steps // 10)), num_training_steps=max(1, total_steps)
    )

    global_step = 0
    optimizer.zero_grad()

    for epoch in range(args.epochs):
        random.shuffle(train_data)
        pbar = tqdm(enumerate(train_data), total=len(train_data), desc=f"SDFT Epoch {epoch + 1}/{args.epochs}")

        for idx, ex in pbar:
            t0 = time.time()
            student_prompt = format_chat(tokenizer, ex["prompt"])
            teacher_prompt = format_chat(
                tokenizer, build_teacher_prompt(ex["prompt"], ex["demonstration"])
            )

            s_inputs = tokenizer(student_prompt, return_tensors="pt", truncation=True, max_length=1024).to(
                model.device
            )
            t_inputs = tokenizer(teacher_prompt, return_tensors="pt", truncation=True, max_length=1536).to(
                model.device
            )

            # 1. On-policy student rollout: y ~ pi_theta(. | x)
            model.eval()
            model.set_adapter("default")
            with torch.no_grad():
                gen_out = model.generate(
                    **s_inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=1.0,
                    top_p=0.95,
                    pad_token_id=tokenizer.pad_token_id,
                )
            prompt_len_s = s_inputs["input_ids"].shape[1]
            resp_ids = gen_out[:, prompt_len_s:]
            if resp_ids.shape[1] < 2:
                continue

            # 2. Teacher logits on the SAME student-generated trajectory
            teacher_full_ids = torch.cat([t_inputs["input_ids"], resp_ids], dim=1)
            prompt_len_t = t_inputs["input_ids"].shape[1]

            model.set_adapter("teacher")
            with torch.no_grad():
                t_logits_all = model(input_ids=teacher_full_ids).logits
            t_resp_logits = t_logits_all[:, prompt_len_t - 1 : -1, :].detach()

            # 3. Student logits on the student-generated trajectory
            model.train()
            model.set_adapter("default")
            for p in trainable_params:
                p.requires_grad_(True)
            s_logits_all = model(input_ids=gen_out).logits
            s_resp_logits = s_logits_all[:, prompt_len_s - 1 : -1, :]

            # Mask out first few tokens to suppress learned artifacts (Section 5)
            mask = torch.ones((1, resp_ids.shape[1]), device=model.device)
            if resp_ids.shape[1] > args.artifact_mask_tokens:
                mask[:, : args.artifact_mask_tokens] = 0.0

            loss = compute_analytic_kl_loss(s_resp_logits, t_resp_logits, mask, kl_mode=args.kl_mode)
            (loss / args.grad_accum).backward()

            if (idx + 1) % args.grad_accum == 0 or (idx + 1) == len(train_data):
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                update_ema_adapter(model, alpha=args.ema_alpha)
                global_step += 1

                dt = max(time.time() - t0, 1e-6)
                wandb.log(
                    {
                        "train/kl_loss": loss.item(),
                        "train/lr": scheduler.get_last_lr()[0],
                        "train/resp_tokens": resp_ids.shape[1],
                        "train/step_sec": dt,
                        "epoch": epoch + (idx + 1) / len(train_data),
                    },
                    step=global_step,
                )
                pbar.set_postfix({"kl_loss": f"{loss.item():.4f}"})

    save_dir = os.path.join(args.output_dir, f"sdft_{args.task}_ema{args.ema_alpha}_s{args.seed}")
    os.makedirs(save_dir, exist_ok=True)
    model.set_adapter("default")
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SDFT Trainer (QLoRA)")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--task", type=str, default="science_qa", choices=["science_qa", "tool_use"])
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--ema_alpha", type=float, default=0.02)
    parser.add_argument("--kl_mode", type=str, default="forward", choices=["forward", "reverse"])
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--artifact_mask_tokens", type=int, default=4)
    parser.add_argument("--max_train", type=int, default=200)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    train_sdft(parser.parse_args())
