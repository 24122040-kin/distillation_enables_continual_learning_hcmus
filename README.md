# Self-Distillation Enables Continual Learning (SDFT) — Reproduction & Analysis

Official course project repository for **Introduction to Machine Learning (HCMUS)**.
This repository contains an **independent re-implementation** of **Self-Distillation Fine-Tuning (SDFT)** from *"Self-Distillation Enables Continual Learning"* (Shenfeld et al., 2026, `arXiv:2601.19897v2`), adapted to run on **100% free GPU tiers** (Kaggle 2×T4 16GB / Google Colab T4 16GB) using 4-bit QLoRA.

---

## 1. Scope & Hardware Adaptation (Core Scope)

The original paper trains `Qwen2.5-7B-Instruct` with full fine-tuning on a single NVIDIA H200 (141 GB VRAM). Under free-tier GPU constraints (16–32 GB VRAM), we adopt **Core Scope** as defined in the project guidelines:
1. **Headline Comparison:** SDFT vs. SFT vs. Base Model on **Skill Learning** (New Task Accuracy + Prior Capabilities retention via `lm-eval`).
2. **Additional Self-Designed Analysis:** Sensitivity & ablation sweep over **EMA rate $\alpha \in \{0.0, 0.02, 1.0\}$** and **KL direction (`forward` vs `reverse`)** to test the paper's theoretical vs. practical claims.

*(Hardware adaptation: 4-bit QLoRA with `bfloat16`/`float16` compute and LoRA adapter EMA, documented in the Deviation Analysis.)*

---

## 2. Experiment Tracking (Weights & Biases)

All runs are tracked in a public W&B project.

| Experiment | Script / Notebook | W&B Run Link | Summary / Key Finding |
| :--- | :--- | :--- | :--- |
| Base Model Eval (`Qwen2.5-3B` / `7B`) | `src/evaluate.py` | `[Add W&B link]` | Baseline New Task & Prior Capabilities |
| SFT Baseline (Off-policy) | `src/sft_baseline.py` | `[Add W&B link]` | Higher new-task acc, degrades prior tasks |
| SDFT (On-policy, EMA $\alpha=0.02$) | `src/sdft_trainer.py` | `[Add W&B link]` | Improves new task while preserving prior tasks |
| Ablation: EMA $\alpha$ & Forward vs Reverse KL | `src/sdft_trainer.py` | `[Add W&B link]` | Tests stability & theory-practice gap |

---

## 3. Quickstart on Free Kaggle / Google Colab

```bash
git clone <your-public-repo-url>
cd sdft-repro
pip install -r requirements.txt
```

Set your W&B API key (from `https://forge.coreweave.com` or `https://wandb.ai`):
```bash
export WANDB_API_KEY="your_wandb_api_key"
export WANDB_PROJECT="sdft-continual-learning"
```

Run the pipeline:
```bash
# 1. Evaluate Base Model
python src/evaluate.py --model_name Qwen/Qwen2.5-3B-Instruct --task science_qa --seed 42

# 2. Train SFT Baseline (QLoRA)
python src/sft_baseline.py --model_name Qwen/Qwen2.5-3B-Instruct --task science_qa --epochs 1 --lr 1e-4 --seed 42

# 3. Train SDFT (QLoRA + EMA Teacher)
python src/sdft_trainer.py --model_name Qwen/Qwen2.5-3B-Instruct --task science_qa --epochs 2 --lr 1e-4 --ema_alpha 0.02 --kl_mode forward --seed 42

# 4. Evaluate Prior Capabilities (6 benchmarks via lm-eval)
lm_eval --model hf \
  --model_args pretrained=Qwen/Qwen2.5-3B-Instruct,peft=./checkpoints/sdft_science_qa_ema0.02_s42,load_in_4bit=True \
  --tasks hellaswag,truthfulqa_mc2,mmlu,winogrande \
  --batch_size auto \
  --limit 200 \
  --output_path ./eval_results/sdft
```

---

## 4. Reused Libraries & Citations

Per course integrity rules, the core SDFT algorithm (teacher conditioning, on-policy rollout, analytic per-token KL loss, and EMA adapter update) is implemented from scratch from the paper's equations (Section 3, Algorithm 1, Appendix A.1). Standard auxiliary tooling is reused from:
- **PyTorch** (`torch`) — tensor computation and autograd
- **Hugging Face Transformers, PEFT, BitsAndBytes, Datasets** — model loading, 4-bit quantization, LoRA adapters, and programmatic dataset loading
- **Weights & Biases** (`wandb`) — experiment tracking
- **EleutherAI `lm-evaluation-harness`** (`lm_eval`) — evaluation of prior capabilities (HellaSwag, MMLU, Winogrande, TruthfulQA, IFEval, HumanEval)
