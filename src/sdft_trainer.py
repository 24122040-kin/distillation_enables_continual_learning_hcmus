"""Independent re-implementation of Self-Distillation Fine-Tuning (SDFT).

Implements Algorithm 1, Section 3 and Appendix A.1 of Shenfeld et al. (2026),
"Self-Distillation Enables Continual Learning", from the paper's description.

Design (and deviations, see docs/reproduction_plan.md §7):

* Student pi_theta and teacher pi_phi share one frozen base model and differ
  only in their LoRA adapters ("default" = student, "teacher" = EMA copy).
  The EMA is applied to the LoRA factors A and B separately, which is only an
  approximation of an EMA over the weight update B @ A.
* One on-policy rollout per prompt (Appendix B.1); rollouts of ``gen_batch_size``
  prompts are generated together to amortise decoding cost.
* Loss: full analytic per-token KL over the vocabulary (Appendix A.1). Forward
  KL(teacher || student) is the default because the paper reports it works best
  in practice; reverse KL (Eq. 1) is available for the theory/practice analysis.
* The first ``artifact_mask_tokens`` response tokens are excluded from the loss
  (Section 5, "Learned Artifacts").
* Checkpoint/resume plus a wall-clock budget (``--max_hours``) so a long run can
  be split across several 12-hour Kaggle sessions.
"""

import argparse
import json
import math
import os
import random
import time
import uuid
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from peft import LoraConfig, get_peft_model
from transformers import get_cosine_schedule_with_warmup
import wandb

from common import chat_ids, generate_batch_split_on_oom, load_model_and_tokenizer, set_seed
from data_loader import get_dataset, resolve_source
from prompts import student_context, teacher_context

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


# --------------------------------------------------------------------------- #
# Core SDFT pieces
# --------------------------------------------------------------------------- #
def _chunk_kl(
    s_chunk: torch.Tensor,
    t_chunk: torch.Tensor,
    m_chunk: torch.Tensor,
    kl_mode: str,
) -> torch.Tensor:
    """Compute masked per-token KL sum over one sequence slice in float32."""
    s_logp = F.log_softmax(s_chunk.float(), dim=-1)
    t_logp = F.log_softmax(t_chunk.float(), dim=-1)
    if kl_mode == "forward":
        token_kl = (t_logp.exp() * (t_logp - s_logp)).sum(-1)
    else:
        token_kl = (s_logp.exp() * (s_logp - t_logp)).sum(-1)
    return (token_kl * m_chunk).sum()


def compute_analytic_kl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: torch.Tensor,
    kl_mode: str = "forward",
    chunk_size: int = 128,
) -> torch.Tensor:
    """Full analytic per-token KL over the vocabulary, averaged over unmasked tokens.

    Parameters
    ----------
    student_logits : torch.Tensor
        Student logits at the response positions, shape (T, V); carries gradient.
    teacher_logits : torch.Tensor
        Teacher logits at the same positions, shape (T, V); detached.
    mask : torch.Tensor
        Float mask of shape (T,), 1 for tokens that enter the loss.
    kl_mode : str
        ``"forward"``: sum_v p_T log(p_T / p_S) (used for the paper's results);
        ``"reverse"``: sum_v p_S log(p_S / p_T) (Eq. 1 / Eq. 2).
    chunk_size : int
        Time-axis chunk size for gradient-checkpointed KL to bound peak VRAM.

    Returns
    -------
    torch.Tensor
        Scalar loss.
    """
    if kl_mode not in ("forward", "reverse"):
        raise ValueError(f"Unsupported kl_mode: {kl_mode}")
    use_ckpt = student_logits.requires_grad and student_logits.size(0) > chunk_size
    total = student_logits.new_zeros((), dtype=torch.float32)
    for i in range(0, student_logits.size(0), chunk_size):
        s_c = student_logits[i: i + chunk_size]
        t_c = teacher_logits[i: i + chunk_size]
        m_c = mask[i: i + chunk_size]
        if use_ckpt:
            total = total + grad_checkpoint(_chunk_kl, s_c, t_c, m_c, kl_mode, use_reentrant=False)
        else:
            total = total + _chunk_kl(s_c, t_c, m_c, kl_mode)
    return total / mask.sum().clamp(min=1.0)


@torch.no_grad()
def update_ema_adapter(model: torch.nn.Module, alpha: float) -> None:
    """Teacher update phi <- alpha * theta + (1 - alpha) * phi (Algorithm 1, line 19).

    Parameters
    ----------
    model : torch.nn.Module
        PEFT model holding the ``default`` (student) and ``teacher`` adapters.
    alpha : float
        EMA rate. 0 keeps the teacher frozen at its initial value (= base model,
        because LoRA B is initialised to zero); 1 copies the student.
    """
    if alpha <= 0.0:
        return
    params = dict(model.named_parameters())
    for name, student in params.items():
        if ".default." in name:
            teacher = params.get(name.replace(".default.", ".teacher."))
            if teacher is not None:
                teacher.data.mul_(1.0 - alpha).add_(student.data.to(teacher.dtype), alpha=alpha)


def _set_active(model, adapter: str, student_params: List[torch.nn.Parameter]) -> None:
    """Activate an adapter while keeping only the student's parameters trainable.

    Parameters
    ----------
    model : PeftModel
        Model with both adapters.
    adapter : str
        ``"default"`` or ``"teacher"``.
    student_params : list of Parameter
        Parameters owned by the optimizer.
    """
    model.set_adapter(adapter)
    for n, p in model.named_parameters():
        p.requires_grad_(False)
    for p in student_params:
        p.requires_grad_(True)


def sdft_example_loss(
    model,
    tokenizer,
    example: Dict,
    student_prompt: List[int],
    response: List[int],
    student_params: List[torch.nn.Parameter],
    args: argparse.Namespace,
) -> Optional[torch.Tensor]:
    """KL loss of one on-policy rollout (Algorithm 1, lines 8-12 and 16).

    Parameters
    ----------
    model, tokenizer
        PEFT model and tokenizer.
    example : dict
        Training example with ``system``, ``question`` and ``demonstration``.
    student_prompt : list of int
        Token ids of Ctx_S(x).
    response : list of int
        Rollout y sampled from the student.
    student_params : list of Parameter
        Trainable student parameters.
    args : argparse.Namespace
        Needs ``artifact_mask_tokens`` and ``kl_mode``.

    Returns
    -------
    torch.Tensor or None
        Scalar loss with gradient, or ``None`` if the rollout is too short to train on.
    """
    n = len(response)
    if n <= args.artifact_mask_tokens:
        return None
    device = next(model.parameters()).device
    resp = torch.tensor(response, device=device)

    # Teacher log-probs on the student's own trajectory, conditioned on the demonstration.
    t_prompt = torch.tensor(
        chat_ids(tokenizer, *teacher_context(example["system"], example["question"], example["demonstration"])),
        device=device,
    )
    model.eval()
    _set_active(model, "teacher", student_params)
    with torch.no_grad():
        # Only the last n + 1 positions are projected to the vocabulary (logits_to_keep):
        # full-sequence logits of a long Tool Use prompt alone take GBs on a T4.
        t_logits = model(input_ids=torch.cat([t_prompt, resp])[None], logits_to_keep=n + 1).logits[0, :n]
    t_logits = t_logits.detach()

    # Student log-probs on the same trajectory, conditioned on the task prompt only.
    model.train()
    _set_active(model, "default", student_params)
    s_prompt = torch.tensor(student_prompt, device=device)
    s_logits = model(input_ids=torch.cat([s_prompt, resp])[None], logits_to_keep=n + 1).logits[0, :n]

    mask = torch.ones(n, device=device)
    mask[: args.artifact_mask_tokens] = 0.0
    return compute_analytic_kl_loss(s_logits, t_logits, mask, args.kl_mode, getattr(args, "kl_chunk", 128))


# --------------------------------------------------------------------------- #
# Checkpointing
# --------------------------------------------------------------------------- #
def _adapter_state(model) -> Dict[str, torch.Tensor]:
    """Return every LoRA tensor (student and teacher) keyed by parameter name."""
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters() if "lora_" in n}


def save_resume_checkpoint(path: str, model, optimizer, scheduler, state: Dict) -> None:
    """Save everything needed to continue training bit-for-bit in a new session.

    Parameters
    ----------
    path : str
        Directory to write.
    model, optimizer, scheduler
        Training objects.
    state : dict
        Loop counters, W&B run id, etc.
    """
    os.makedirs(path, exist_ok=True)
    tmp = os.path.join(path, "ckpt.pt.tmp")
    torch.save(
        {
            "adapters": _adapter_state(model),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
        },
        tmp,
    )
    os.replace(tmp, os.path.join(path, "ckpt.pt"))
    with open(os.path.join(path, "state.json"), "w") as f:
        json.dump(state, f, indent=2)


def load_resume_checkpoint(path: str, model, optimizer, scheduler) -> Dict:
    """Restore a checkpoint written by :func:`save_resume_checkpoint`.

    Parameters
    ----------
    path : str
        Checkpoint directory.
    model, optimizer, scheduler
        Freshly built training objects (same configuration as the saved run).

    Returns
    -------
    dict
        The saved loop state.
    """
    ckpt = torch.load(os.path.join(path, "ckpt.pt"), map_location="cpu", weights_only=False)
    params = dict(model.named_parameters())
    with torch.no_grad():
        for n, t in ckpt["adapters"].items():
            params[n].data.copy_(t.to(params[n].device, params[n].dtype))
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    random.setstate(ckpt["rng"]["python"])
    np.random.set_state(ckpt["rng"]["numpy"])
    torch.set_rng_state(ckpt["rng"]["torch"])
    if ckpt["rng"]["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(ckpt["rng"]["cuda"])
    with open(os.path.join(path, "state.json")) as f:
        return json.load(f)


def run_dir_name(args: argparse.Namespace) -> str:
    """Deterministic output directory name for a configuration."""
    model = args.model_name.split("/")[-1]
    n = args.max_train if args.max_train else "all"
    return f"sdft_{args.task}_{model}_n{n}_ema{args.ema_alpha}_{args.kl_mode}_s{args.seed}"


# --------------------------------------------------------------------------- #
# Training loop
# --------------------------------------------------------------------------- #
def train_sdft(args: argparse.Namespace) -> None:
    """Run SDFT (Algorithm 1) with checkpoint/resume and W&B logging.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line configuration.
    """
    if args.batch_size % args.gen_batch_size:
        raise ValueError("--batch_size must be a multiple of --gen_batch_size")
    set_seed(args.seed)
    out_dir = os.path.join(args.output_dir, run_dir_name(args))
    resume_dir = os.path.join(out_dir, "resume")

    train, _, _ = get_dataset(
        args.task, max_train=args.max_train, data_seed=args.data_seed,
        source=args.data_source, hf_dataset=args.hf_dataset,
    )
    model, tokenizer = load_model_and_tokenizer(args.model_name, args.quant_bits, for_training=True)
    lora = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM", target_modules=LORA_TARGETS,
    )
    model = get_peft_model(model, lora, adapter_name="default")
    model.add_adapter("teacher", lora)
    student_params = [p for n, p in model.named_parameters() if ".default." in n and "lora_" in n]
    update_ema_adapter(model, alpha=1.0)  # phi = theta (Algorithm 1, line 1)
    _set_active(model, "default", student_params)

    steps_per_epoch = math.ceil(len(train) / args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    optimizer = torch.optim.AdamW(student_params, lr=args.lr, weight_decay=0.0)
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, total_steps)

    state = {"epoch": 0, "step_in_epoch": 0, "global_step": 0, "wandb_id": None, "elapsed_h": 0.0}
    if os.path.exists(os.path.join(resume_dir, "ckpt.pt")):
        state = load_resume_checkpoint(resume_dir, model, optimizer, scheduler)
        print(f"Resumed from {resume_dir} at global step {state['global_step']}")

    wandb_id = state["wandb_id"] or uuid.uuid4().hex[:8]
    state["wandb_id"] = wandb_id
    wandb.init(
        project=os.environ.get("WANDB_PROJECT", "sdft-continual-learning"),
        id=wandb_id, resume="allow", name=args.run_name or run_dir_name(args),
        config={**vars(args), "n_train": len(train), "total_steps": total_steps,
                "data_source_resolved": resolve_source(args.data_source, args.hf_dataset),
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"},
        tags=["lab1", "sdft", args.task, args.model_name.split("/")[-1]] + args.tags,
    )

    session_start = time.time()
    base_elapsed = state["elapsed_h"]
    for epoch in range(state["epoch"], args.epochs):
        order = list(range(len(train)))
        random.Random(args.seed * 1000 + epoch).shuffle(order)  # same order after a resume
        first = state["step_in_epoch"] if epoch == state["epoch"] else 0
        for step in range(first, steps_per_epoch):
            t0 = time.time()
            batch = [train[i] for i in order[step * args.batch_size:(step + 1) * args.batch_size]]
            losses, lengths, truncs, skipped, gen_s, train_s = [], [], 0, 0, 0.0, 0.0
            optimizer.zero_grad(set_to_none=True)
            for g in range(0, len(batch), args.gen_batch_size):
                chunk = batch[g:g + args.gen_batch_size]
                prompts = [chat_ids(tokenizer, *student_context(ex["system"], ex["question"])) for ex in chunk]
                tg = time.time()
                model.eval()
                _set_active(model, "default", student_params)
                rollouts = generate_batch_split_on_oom(
                    model, tokenizer, prompts, args.max_new_tokens,
                    do_sample=True, temperature=args.temperature, top_p=args.top_p,
                )
                gen_s += time.time() - tg
                tt = time.time()
                for ex, prompt, ro in zip(chunk, prompts, rollouts):
                    loss = sdft_example_loss(model, tokenizer, ex, prompt, ro["ids"], student_params, args)
                    lengths.append(len(ro["ids"]))
                    truncs += int(ro["truncated"])
                    if loss is None:
                        skipped += 1
                        continue
                    (loss / len(batch)).backward()
                    losses.append(loss.item())
                train_s += time.time() - tt

            grad_norm = torch.nn.utils.clip_grad_norm_(student_params, args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            update_ema_adapter(model, args.ema_alpha)
            state["global_step"] += 1
            state["epoch"], state["step_in_epoch"] = epoch, step + 1
            state["elapsed_h"] = base_elapsed + (time.time() - session_start) / 3600

            wandb.log({
                "train/kl_loss": float(np.mean(losses)) if losses else float("nan"),
                "train/grad_norm": float(grad_norm),
                "train/lr": scheduler.get_last_lr()[0],
                "train/resp_tokens_mean": float(np.mean(lengths)),
                "train/truncated_frac": truncs / len(batch),
                "train/skipped": skipped,
                "time/step_sec": time.time() - t0,
                "time/gen_sec": gen_s,
                "time/train_sec": train_s,
                "time/elapsed_h": state["elapsed_h"],
                "epoch": epoch + (step + 1) / steps_per_epoch,
            }, step=state["global_step"])
            print(f"[ep {epoch + 1} step {step + 1}/{steps_per_epoch}] kl={np.mean(losses) if losses else float('nan'):.4f} "
                  f"len={np.mean(lengths):.0f} gen={gen_s:.0f}s train={train_s:.0f}s", flush=True)

            out_of_time = args.max_hours and (time.time() - session_start) / 3600 > args.max_hours
            if out_of_time or (args.save_every and state["global_step"] % args.save_every == 0):
                save_resume_checkpoint(resume_dir, model, optimizer, scheduler, state)
            if out_of_time:
                print(f"Time budget of {args.max_hours} h reached; re-run the same command to resume.")
                wandb.finish()
                return

        # End of epoch: save the student adapter for validation-based checkpoint selection.
        state["epoch"], state["step_in_epoch"] = epoch + 1, 0
        _set_active(model, "default", student_params)
        model.save_pretrained(os.path.join(out_dir, f"epoch{epoch + 1}"), selected_adapters=["default"])
        tokenizer.save_pretrained(os.path.join(out_dir, f"epoch{epoch + 1}"))
        save_resume_checkpoint(resume_dir, model, optimizer, scheduler, state)
        if args.stop_after_epoch and epoch + 1 >= args.stop_after_epoch < args.epochs:
            # Same schedule as the full run (lr still follows --epochs); re-run without the flag to continue.
            print(f"Stopping after epoch {epoch + 1} (--stop_after_epoch); re-run to continue.", flush=True)
            wandb.finish()
            return

    with open(os.path.join(out_dir, "DONE"), "w") as f:
        f.write(json.dumps(state))
    wandb.finish()


def build_parser() -> argparse.ArgumentParser:
    """Command-line options (defaults = the team's main configuration)."""
    p = argparse.ArgumentParser(description="SDFT trainer (LoRA / QLoRA)")
    p.add_argument("--model_name", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--task", default="science", choices=["science", "tool_use"])
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup_steps", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=32, help="Prompts per optimizer step")
    p.add_argument("--gen_batch_size", type=int, default=8, help="Rollouts generated together")
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--ema_alpha", type=float, default=0.01)
    p.add_argument("--kl_mode", default="forward", choices=["forward", "reverse"])
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--artifact_mask_tokens", type=int, default=3)
    p.add_argument("--max_train", type=int, default=None, help="Training subset size (None = all)")
    p.add_argument("--data_seed", type=int, default=0)
    p.add_argument("--data_source", default="auto", choices=["auto", "hf", "github"])
    p.add_argument("--hf_dataset", default=None)
    p.add_argument("--quant_bits", type=int, default=16, choices=[4, 8, 16])
    p.add_argument("--lora_r", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_hours", type=float, default=None, help="Stop and checkpoint after this many hours")
    p.add_argument("--kl_chunk", type=int, default=128, help="Tokens per chunk of the vocabulary KL (peak memory)")
    p.add_argument("--stop_after_epoch", type=int, default=None,
                   help="Save and exit after this epoch, keeping the lr schedule of --epochs (time budget)")
    p.add_argument("--save_every",type=int, default=10, help="Optimizer steps between resume checkpoints")
    p.add_argument("--output_dir", default="./checkpoints")
    p.add_argument("--run_name", default=None)
    p.add_argument("--tags", nargs="*", default=[])
    return p


if __name__ == "__main__":
    train_sdft(build_parser().parse_args())
