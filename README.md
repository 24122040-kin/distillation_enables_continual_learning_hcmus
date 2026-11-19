# Self-Distillation Enables Continual Learning (SDFT): Reproduction & Analysis

Course project for **Introduction to Machine Learning (HCMUS)**: an independent
re-implementation of **Self-Distillation Fine-Tuning (SDFT)** from
*"Self-Distillation Enables Continual Learning"* (Shenfeld et al., 2026, `arXiv:2601.19897v2`).

- Plan, scope, compute estimates and deviation list: [`docs/reproduction_plan.md`](docs/reproduction_plan.md)
- Slide-deck outline: [`docs/slide_outline.md`](docs/slide_outline.md)
- Kaggle runbook, person 2 (Vietnamese): [`docs/huong_dan_nguoi_2.md`](docs/huong_dan_nguoi_2.md)
- Kaggle runbook, account A / Kin (Vietnamese): [`docs/huong_dan_tai_khoan_A.md`](docs/huong_dan_tai_khoan_A.md)
- Hugging Face dataset guide (Vietnamese): [`docs/huong_dan_data_huggingface.md`](docs/huong_dan_data_huggingface.md)

---

## 1. Scope (Core scope)

| ID | Experiment | Paper reference | Model |
|---|---|---|---|
| E1 | Science Q&A: Base vs SFT vs SDFT, new-task accuracy + prior capabilities | Table 5, Fig. 4 | Qwen2.5-7B-Instruct |
| E2 | Tool Use: Base vs SFT vs SDFT | Table 5, Fig. 4 | Qwen2.5-7B-Instruct |
| E3 | Additional analysis: model size (1.5B / 3B / 7B) on Science, including teacher accuracy | Fig. 5 (left) | Qwen2.5 family |

Hardware adaptation: LoRA (r = 32, all linear layers) instead of full fine-tuning,
optionally 4-bit NF4 (QLoRA) on 16 GB GPUs. All deviations from the paper are listed in
`docs/reproduction_plan.md` §7.

## 2. Data

We use the **datasets released by the paper's authors**, unchanged:

| Task | Train | Test | Source |
|---|---|---|---|
| Science Q&A (SciKnowEval Chem L-3, GPT-4o demonstrations) | 2 674 | 507 | `idanshen/Self-Distillation` `data/science_data` |
| Tool Use (ToolAlpaca, original split) | 4 046 | 97 | `idanshen/Self-Distillation` `data/tooluse_data` |

`src/data_loader.py` reads them programmatically from the team's **public Hugging Face mirror**
([`Baokin/sdft-official-data`](https://huggingface.co/datasets/Baokin/sdft-official-data); repo id in `$SDFT_HF_DATASET`
or `--hf_dataset`, set by default in `scripts/kaggle_jobs.sh`), created unchanged from the official GitHub repository
pinned to commit `d775732`. `--data_source github` reads that pinned copy instead, and
`bash scripts/kaggle_jobs.sh check_data` verifies that both sources give identical train/val/test examples
(checked on 20/11: all six splits identical). Every run logs `data_source_resolved` to W&B; the first 7B SDFT
Science run (started 19/11) read the GitHub copy. 5 % of the official train split is held out as a fixed
validation set (data seed 0) for checkpoint selection. `--max_train N` keeps a fixed prefix of one
permutation, so SFT and SDFT always train on the same examples.
*(Note: on the HF mirror, split-specific missing columns are padded with typed `null` values so `train` and `test` share one Arrow schema per config without modifying any example values.)*

Load from the public HF mirror directly (no `HF_TOKEN` required):

```bash
export SDFT_HF_DATASET=Baokin/sdft-official-data
# or pass flags explicitly: --data_source hf --hf_dataset Baokin/sdft-official-data
```

```python
from datasets import load_dataset
sci_train = load_dataset("Baokin/sdft-official-data", "science", split="train")
tool_test = load_dataset("Baokin/sdft-official-data", "tooluse", split="test")
```

To recreate or update an HF mirror (needs a Hugging Face write token):

```bash
python scripts/mirror_official_data.py --hf_repo Baokin/sdft-official-data
```

## 3. Repository layout

| Path | Role |
|---|---|
| `src/data_loader.py` | Download + format the official data into `{system, question, demonstration, answer}` |
| `src/prompts.py` | Student context Ctx_S(x) and teacher context Ctx_T(x, c) (paper §3 template) |
| `src/sdft_trainer.py` | **SDFT** (Algorithm 1): batched on-policy rollouts, analytic per-token KL (forward / reverse), EMA teacher adapter, first-token masking, checkpoint/resume, `--max_hours` budget |
| `src/sft_baseline.py` | SFT baseline with the same data, LoRA config and batch size |
| `src/evaluate.py` | Greedy new-task evaluation, teacher evaluation (`--with_demo`), comparison with Table 5, W&B logging |
| `src/answer_parsing.py` | Science `<answer>` parser; Tool Use matcher (paper-style merged metric + strict metric) |
| `src/common.py` | Seeding, model loading (bf16 / fp16 / 4-bit / 8-bit), batched generation |
| `scripts/mirror_official_data.py` | Copy the official data to Hugging Face |
| `tests/` | Data and parser tests (`pytest tests/`) |

## 4. Quickstart

```bash
pip install -r requirements.txt
export WANDB_PROJECT=sdft-continual-learning   # public W&B project
export SDFT_HF_DATASET=Baokin/sdft-official-data
```

```bash
# Base model, new task (test split)
python src/evaluate.py --model_name Qwen/Qwen2.5-7B-Instruct --task science --method base

# SFT (1 epoch) and SDFT (2 epochs); add --quant_bits 4 on 16 GB GPUs (T4)
python src/sft_baseline.py --model_name Qwen/Qwen2.5-7B-Instruct --task science --seed 42
python src/sdft_trainer.py --model_name Qwen/Qwen2.5-7B-Instruct --task science --seed 42 --max_hours 11

# Select the epoch on the validation split, then evaluate it on test
python src/evaluate.py --checkpoint_dir checkpoints/<run>/epoch2 --method sdft --task science --split val
python src/evaluate.py --checkpoint_dir checkpoints/<run>/epoch2 --method sdft --task science --split test

# Teacher accuracy (demonstration in context), used in E3
python src/evaluate.py --model_name Qwen/Qwen2.5-3B-Instruct --task science --split val --with_demo
```

**Long runs on Kaggle (12-hour sessions).** `--max_hours 11` makes `sdft_trainer.py` save a resume checkpoint
(`checkpoints/<run>/resume/`, ~1.3 GB for 7B) and exit cleanly; re-running the identical command continues from it,
in the same W&B run. Persist `checkpoints/<run>/` between sessions (commit the notebook so `/kaggle/working`
becomes an output, then attach it as input in the next session, or upload the folder to a private HF repo).

Prior capabilities (lm-evaluation-harness, all six benchmarks of the paper):

```bash
HF_ALLOW_CODE_EVAL=1 lm_eval --model hf \
  --model_args pretrained=Qwen/Qwen2.5-7B-Instruct,peft=checkpoints/<run>/epoch2,dtype=bfloat16 \
  --tasks hellaswag,truthfulqa_mc2,mmlu,ifeval,winogrande,humaneval \
  --apply_chat_template --confirm_run_unsafe_code --batch_size auto \
  --wandb_args project=$WANDB_PROJECT,name=prior-<run> --output_path eval_results/prior/<run>
```

## 5. Experiment tracking (Weights & Biases)

| Experiment | Script | W&B run | Finding |
|---|---|---|---|
| E1 Base, Science | `src/evaluate.py` | _todo_ | |
| E1 SFT, Science | `src/sft_baseline.py` + `src/evaluate.py` | _todo_ | |
| E1 SDFT, Science | `src/sdft_trainer.py` + `src/evaluate.py` | _todo_ | |
| E1 prior capabilities | `lm_eval` | _todo_ | |
| E2 Base / SFT / SDFT, Tool Use | same as E1 | _todo_ | |
| E3 1.5B / 3B SFT, SDFT, teacher | same as E1 | _todo_ | |

## 6. Reproducibility

- Seeds: training seed `--seed` (42, 43, 44); data permutation `--data_seed 0` (fixed for all runs).
- Run names and W&B tags follow `docs/reproduction_plan.md` §6; every run logs its full config and GPU type.
- Smoke test on CPU: `python src/sdft_trainer.py --model_name Qwen/Qwen2.5-0.5B-Instruct --max_train 4 --batch_size 2 --gen_batch_size 2 --max_new_tokens 12 --epochs 1`.

## 7. Reused libraries and sources

The core SDFT algorithm (teacher/student contexts, on-policy rollout, analytic per-token KL, EMA teacher update,
first-token masking) is implemented by the team from the paper's description (Section 3, Algorithm 1, Appendix A).
The official implementation (`github.com/idanshen/Self-Distillation`) is used **only** for its released data and
for comparison; no code is copied from it.

- **PyTorch**: tensors and autograd.
- **Hugging Face Transformers, PEFT, bitsandbytes, Datasets, Accelerate**: model loading, LoRA adapters, 4-bit
  quantization, data loading. Outside the course's standard stack; required to run 7B LLMs on one GPU.
- **Weights & Biases**: experiment tracking (course requirement).
- **EleutherAI lm-evaluation-harness**: prior-capability benchmarks, as used by the paper.
- **Data**: SciKnowEval (Feng et al., 2024) and ToolAlpaca (Tang et al., 2023), as preprocessed by Shenfeld et al. (2026).
