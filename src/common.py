"""Shared utilities: seeding, model loading, chat formatting, batched generation.

Everything here is standard tooling around the method (model loading through
``transformers``/``peft``/``bitsandbytes``); the SDFT-specific parts live in
``prompts.py`` and ``sdft_trainer.py``.
"""

import random
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from data_loader import to_messages


def set_seed(seed: int) -> None:
    """Fix Python, NumPy and PyTorch seeds.

    Parameters
    ----------
    seed : int
        Seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_dtype() -> torch.dtype:
    """Pick bf16 when the GPU supports it natively (A100/L4/H100), else fp16 (T4) or fp32 (CPU).

    Returns
    -------
    torch.dtype
        The dtype used for weights / compute.
    """
    if torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def load_model_and_tokenizer(model_name: str, quant_bits: int = 16, for_training: bool = True):
    """Load a causal LM (optionally 4/8-bit quantized) and its tokenizer.

    Parameters
    ----------
    model_name : str
        Hugging Face model id, e.g. ``Qwen/Qwen2.5-7B-Instruct``.
    quant_bits : int
        4 (NF4 QLoRA), 8, or 16 (no quantization).
    for_training : bool
        If True and quantized, run ``prepare_model_for_kbit_training``.

    Returns
    -------
    (model, tokenizer)
        The base model and a tokenizer with left padding and a pad token.
    """
    dtype = compute_dtype()
    kwargs = {"torch_dtype": dtype}
    if torch.cuda.is_available():
        kwargs["device_map"] = "auto"
    if quant_bits in (4, 8):
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = (
            BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
            )
            if quant_bits == 4
            else BitsAndBytesConfig(load_in_8bit=True)
        )
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    if quant_bits in (4, 8) and for_training:
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    elif for_training:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def chat_ids(tokenizer, system: Optional[str], user: str) -> List[int]:
    """Tokenize ``(system, user)`` with the chat template and a generation prompt.

    Parameters
    ----------
    tokenizer : PreTrainedTokenizer
        Model tokenizer.
    system : str or None
        System prompt.
    user : str
        User content.

    Returns
    -------
    list of int
        Prompt token ids ending with the assistant header.
    """
    text = tokenizer.apply_chat_template(
        to_messages(system, user), tokenize=False, add_generation_prompt=True
    )
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def stop_token_ids(model, tokenizer) -> List[int]:
    """Collect every token id that ends generation (e.g. ``<|im_end|>`` for Qwen).

    Parameters
    ----------
    model : PreTrainedModel
        Model whose ``generation_config`` lists EOS ids.
    tokenizer : PreTrainedTokenizer
        Tokenizer (fallback EOS id).

    Returns
    -------
    list of int
        Stop ids.
    """
    eos = model.generation_config.eos_token_id
    ids = list(eos) if isinstance(eos, (list, tuple)) else [eos]
    if tokenizer.eos_token_id is not None:
        ids.append(tokenizer.eos_token_id)
    return sorted({i for i in ids if i is not None})


@torch.no_grad()
def generate_batch(
    model,
    tokenizer,
    prompts: Sequence[List[int]],
    max_new_tokens: int,
    do_sample: bool,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> List[Dict[str, object]]:
    """Generate one response per prompt in a single left-padded batch.

    Parameters
    ----------
    model : PreTrainedModel
        Model (or PEFT model with the desired adapter active).
    tokenizer : PreTrainedTokenizer
        Tokenizer with a pad token.
    prompts : sequence of list of int
        Unpadded prompt ids.
    max_new_tokens : int
        Generation budget.
    do_sample : bool
        Sampling (training rollouts) or greedy (evaluation).
    temperature, top_p : float
        Sampling parameters (ignored when greedy).

    Returns
    -------
    list of dict
        For each prompt: ``ids`` (response ids up to and including the first
        stop token) and ``truncated`` (True if no stop token was produced).
    """
    stops = set(stop_token_ids(model, tokenizer))
    width = max(len(p) for p in prompts)
    pad = tokenizer.pad_token_id
    input_ids = torch.full((len(prompts), width), pad, dtype=torch.long)
    attn = torch.zeros((len(prompts), width), dtype=torch.long)
    for i, p in enumerate(prompts):
        input_ids[i, width - len(p):] = torch.tensor(p)
        attn[i, width - len(p):] = 1
    device = next(model.parameters()).device
    gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=do_sample, pad_token_id=pad, use_cache=True)
    if do_sample:
        gen_kwargs.update(temperature=temperature, top_p=top_p)
    else:
        gen_kwargs.update(temperature=None, top_p=None, top_k=None)
    out = model.generate(input_ids=input_ids.to(device), attention_mask=attn.to(device), **gen_kwargs)
    results = []
    for row in out[:, width:].tolist():
        cut = next((j for j, t in enumerate(row) if t in stops), None)
        ids = row[: cut + 1] if cut is not None else row
        results.append({"ids": ids, "truncated": cut is None})
    return results
