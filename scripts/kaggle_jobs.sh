#!/usr/bin/env bash
# One entry point per Kaggle session.  Usage (from the repo root, in a Kaggle cell):
#
#   !bash scripts/kaggle_jobs.sh setup          # once per session
#   !bash scripts/kaggle_jobs.sh pilot          # ~30 min: measures step time, prints N_SCI / N_TOOL
#
# Schedule for 3 accounts (A, B, C); see docs/reproduction_plan.md §5:
#   B, now     : base                    (7B Base evals + teacher accuracy, ~2.5 h; needs no N)
#   A, evening : pilot, then e1_sdft     (7B SDFT Science, ~10 h)
#   C, evening : e2_sdft                 (7B SDFT Tool Use, ~10 h; after the pilot numbers)
#   B, evening : e1_sft ; e2_sft         (7B SFT, both tasks, ~7 h; after base finishes)
#   A, day 2   : eval_sdft e1            (~2-3 h)
#   C, day 2   : eval_sdft e2            (~2 h)
#   B, day 2   : e3_small                (1.5B + 3B Science, ~9 h)
# Current plan (2 accounts; docs/huong_dan_tai_khoan_A.md, docs/huong_dan_nguoi_2.md):
#   A night : e1_sdft            B night : night2 (GPU0 e2_sdft | GPU1 SFT x2 + E3 3B)
#   A day 2 : day2_a (SDFT Science eval)   B day 2 : day2_b (SDFT Tool + SFT x2 + Base evals)
#   B night 2 (optional): M_E3=Qwen/Qwen2.5-1.5B-Instruct e3_small
#   B re-run (20/11, after the night2 CUDA OOM): night3 (GPU0 e2_sdft from scratch | GPU1 rest of E3 3B)
#
# Pilot on Kaggle T4 (19/11): 50.6 s/example Science, 33.4 s/example Tool Use -> N_SCI=352, N_TOOL=528.
# Sub-sample sizes come from the pilot: N = 10 h * 3600 / (2 epochs * seconds per example).
# Override them without editing the file, e.g.  N_SCI=1400 bash scripts/kaggle_jobs.sh e1_sdft
# Run long jobs with "Save Version -> Save & Run All" so they continue with the browser closed
# and /kaggle/working (checkpoints, eval json) is kept as the notebook output.
# Every job is resumable: if a session dies, re-run the same command (SDFT saves every 10 steps).
set -euo pipefail

N_SCI=${N_SCI:-352}         # Science training subset (same for SFT and SDFT)
N_TOOL=${N_TOOL:-528}       # Tool Use training subset
BS=${BS:-16}                 # batch 16 keeps the step count of full data at batch 32
SEED=${SEED:-42}
M7=${M7:-Qwen/Qwen2.5-7B-Instruct}
M3=${M3:-Qwen/Qwen2.5-3B-Instruct}
M15=${M15:-Qwen/Qwen2.5-1.5B-Instruct}
CK=${CK:-/kaggle/working/checkpoints}
EV=${EV:-/kaggle/working/eval_results}
Q=(--quant_bits "${QUANT:-4}")  # training: 4-bit NF4 on one 16 GB T4; QUANT=16 on A100 / CPU smoke tests
QE=(--quant_bits "${QUANT_EVAL:-16}")  # evaluation: fp16, model split over both T4s (bnb 4-bit decoding is slow)
TRAIN_GPUS=${TRAIN_GPUS:-0}
EVAL_GPUS=${EVAL_GPUS:-0,1}
LIMIT=${LIMIT:-0.25}         # fraction of each lm-eval benchmark (T4 time budget)
export WANDB_PROJECT=${WANDB_PROJECT:-sdft-continual-learning}
# Team Hugging Face mirror of the authors' data (scripts/mirror_official_data.py).
# Empty -> fall back to the official GitHub repo pinned to the same commit (identical files).
HF_DATASET_DEFAULT="Baokin/sdft-official-data"
export SDFT_HF_DATASET=${SDFT_HF_DATASET:-$HF_DATASET_DEFAULT}
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

short() { basename "$1"; }
# Wait for two background jobs; on failure print the end of the job's log (traceback) right away.
wait_both() {  # pid0 log0 pid1 log1
  wait "$1" || { echo "!!! GPU0 job exited with an error, see $2" | tee -a "$EV/FAILED.txt"; tail -n 40 "$2"; }
  wait "$3" || { echo "!!! GPU1 job exited with an error or timed out, see $4" | tee -a "$EV/FAILED.txt"; tail -n 40 "$4"; }
  tail -n 5 "$2" "$4"
}
# Run a step; on failure log it and continue, so one broken evaluation does not stop a long job.
try() { "$@" || echo "!!! FAILED (exit $?): $*" | tee -a "$EV/FAILED.txt"; }

train_sdft() {  # model task n extra...
  local m=$1 t=$2 n=$3; shift 3
  CUDA_VISIBLE_DEVICES=$TRAIN_GPUS python src/sdft_trainer.py --model_name "$m" --task "$t" --max_train "$n" --batch_size "$BS" \
    --gen_batch_size "${GEN_BS:-8}" --seed "$SEED" --max_hours "${MAX_HOURS:-11}" --output_dir "$CK" "${Q[@]}" \
    ${MAX_NEW:+--max_new_tokens "$MAX_NEW"} "$@"
}
train_sft() {   # model task n
  CUDA_VISIBLE_DEVICES=$TRAIN_GPUS python src/sft_baseline.py --model_name "$1" --task "$2" --max_train "$3" --batch_size "$BS" \
    --seed "$SEED" --output_dir "$CK" "${Q[@]}"
}
eval_new() {    # model task method split [checkpoint] [extra...]
  local m=$1 t=$2 meth=$3 split=$4 ck=${5:-}; shift 4; [ $# -gt 0 ] && shift
  CUDA_VISIBLE_DEVICES=$EVAL_GPUS python src/evaluate.py --model_name "$m" --task "$t" --method "$meth" --split "$split" \
    ${ck:+--checkpoint_dir "$ck"} --seed "$SEED" --output_dir "$EV" "${QE[@]}" ${EVAL_MAX:+--max_examples "$EVAL_MAX"} \
    ${EVAL_BS:+--batch_size "$EVAL_BS"} ${MAX_NEW:+--max_new_tokens "$MAX_NEW"} "$@"
}
eval_prior() {  # model name [checkpoint]
  local m=$1 name=$2 ck=${3:-}
  local margs="pretrained=$m,dtype=float16,parallelize=True"
  [ "${QUANT_EVAL:-16}" = 4 ] && margs="pretrained=$m,load_in_4bit=True,dtype=float16"
  CUDA_VISIBLE_DEVICES=$EVAL_GPUS HF_ALLOW_CODE_EVAL=1 lm_eval --model hf \
    --model_args "$margs${ck:+,peft=$ck}" \
    --tasks hellaswag,truthfulqa_mc2,mmlu,ifeval,winogrande,humaneval \
    --apply_chat_template --confirm_run_unsafe_code --batch_size auto --limit "$LIMIT" \
    --wandb_args "project=$WANDB_PROJECT,name=prior-$name,job_type=eval-prior" \
    --output_path "$EV/prior/$name"
}
# Pick the epoch with the best validation accuracy, then evaluate it on test.
best_epoch_eval() {  # model task method run_dir
  local m=$1 t=$2 meth=$3 dir=$4 best="" bestacc=-1
  local eps=("$dir"/epoch*)
  if [ "${#eps[@]}" -eq 1 ]; then   # one epoch (SFT, or SDFT stopped early): nothing to select
    best=${eps[0]}; bestacc=n/a
  else
  for e in "${eps[@]}"; do
    eval_new "$m" "$t" "$meth" val "$e" --run_name "val-$(short "$dir")-$(short "$e")"
    acc=$(python -c "import json,sys;print(json.load(open(sys.argv[1]))['metrics']['acc'])" \
          "$EV/val-$(short "$dir")-$(short "$e").json")
    if python -c "import sys;sys.exit(0 if float(sys.argv[1])>float(sys.argv[2]) else 1)" "$acc" "$bestacc"; then
      best=$e; bestacc=$acc
    fi
  done
  fi
  echo "best checkpoint: $best (val acc $bestacc)"
  echo "$best" > "$EV/best-$(short "$dir").txt"   # read by day2_b's prior-capability pass
  eval_new "$m" "$t" "$meth" test "$best" --run_name "test-$(short "$dir")-$(short "$best")"
  [ -n "${SKIP_PRIOR:-}" ] || try eval_prior "$m" "$(short "$dir")" "$best"
}

mkdir -p "$EV"
job=${1:?job name}
case "$job" in
  setup)
    pip install -q "transformers==4.57.1" "peft==0.17.1" "datasets==4.3.0" "bitsandbytes>=0.44" \
      "lm-eval[ifeval]>=0.4.8" wandb accelerate
    python -c "import torch;print(torch.__version__, torch.cuda.get_device_name(0))" ;;
  pilot)
    # 3 optimizer steps (8 prompts each) of 7B SDFT per task; step 1 is warm-up and ignored.
    mkdir -p "$EV"
    for t in science tool_use; do
      BS=8 CK="$CK/pilot" train_sdft "$M7" "$t" 24 --epochs 1 --save_every 0 \
        --run_name "pilot-sdft-7b-$t" --tags pilot 2>&1 | tee "$EV/pilot_$t.log"
    done
    python - "$EV" <<'PY'
import re, sys, math
full = {"science": 2540, "tool_use": 3844}          # official train minus the 5 % validation split
for task, var in (("science", "N_SCI"), ("tool_use", "N_TOOL")):
    steps = [float(g) + float(t) for g, t in re.findall(r"gen=(\d+)s train=(\d+)s", open(f"{sys.argv[1]}/pilot_{task}.log").read())]
    sec = sum(steps[1:]) / max(1, len(steps[1:])) / 8          # seconds per training example
    n = min(full[task], int(10 * 3600 / (2 * sec)) // 16 * 16)
    print(f"{task}: {sec:.1f} s/example -> 2 epochs of full data = {2 * full[task] * sec / 3600:.1f} h ; use {var}={n}")
PY
    ;;
  e1_sdft)      train_sdft "$M7" science "$N_SCI" ;;
  e2_sdft)      train_sdft "$M7" tool_use "$N_TOOL" ;;
  base)
    # Base model: new-task accuracy (test), teacher accuracy with the demonstration in context
    # (val, paper Section 3.2), and prior capabilities. Independent of N_SCI / N_TOOL.
    try eval_new "$M7" science base test "" --run_name "test-base-7b-science"
    try eval_new "$M7" tool_use base test "" --run_name "test-base-7b-tool"
    try eval_new "$M7" science base val "" --with_demo --run_name "teacher-$(short $M7)-science"
    try eval_new "$M7" tool_use base val "" --with_demo --run_name "teacher-$(short $M7)-tool"
    [ -n "${SKIP_PRIOR:-}" ] || try eval_prior "$M7" base-7b ;;
  e1_sft_train) train_sft "$M7" science "$N_SCI" ;;
  e2_sft_train) train_sft "$M7" tool_use "$N_TOOL" ;;
  eval_sft)
    case "${2:?e1 or e2}" in
      e1) best_epoch_eval "$M7" science sft "$CK/sft_science_$(short $M7)_n${N_SCI}_s${SEED}" ;;
      e2) best_epoch_eval "$M7" tool_use sft "$CK/sft_tool_use_$(short $M7)_n${N_TOOL}_s${SEED}" ;;
    esac ;;
  night2)
    # Two accounts only: one session, two independent jobs, one per T4.
    #   GPU 0: 7B SDFT Tool Use.   GPU 1: 7B SFT (both tasks, training only), then E3 with 3B.
    # Everything must finish before Kaggle's 12 h limit, so GPU 1 gets a hard timeout.
    python -c "from huggingface_hub import snapshot_download as d; d('$M7')"   # download once, no race
    me=$(readlink -f "$0")
    MAX_HOURS=${MAX_HOURS:-10.5} TRAIN_GPUS=0 bash "$me" e2_sdft > "$EV/gpu0_e2_sdft.log" 2>&1 &
    p0=$!
    ${TIMEOUT_BIN:-timeout} "${GPU1_SECONDS:-39000}" bash -c "
      TRAIN_GPUS=1 EVAL_GPUS=1 bash '$me' e1_sft_train
      TRAIN_GPUS=1 EVAL_GPUS=1 bash '$me' e2_sft_train
      TRAIN_GPUS=1 EVAL_GPUS=1 MAX_HOURS=6 M_E3='$M3' bash '$me' e3_small" > "$EV/gpu1_chain.log" 2>&1 &
    p1=$!
    echo "started: GPU0 pid $p0 (e2_sdft), GPU1 pid $p1 (SFT + E3); logs in $EV/gpu*.log"
    wait_both "$p0" "$EV/gpu0_e2_sdft.log" "$p1" "$EV/gpu1_chain.log" ;;
  night3)
    # Re-run of 7B SDFT Tool Use after the CUDA OOM of night2 (fixed in sdft_trainer.py), plus the
    # unfinished part of E3 (3B) on the otherwise idle GPU 1. Needs the night2 checkpoints copied
    # to $CK first (3B SFT adapter). Resumes by itself if $CK already holds a resume checkpoint.
    python -c "from huggingface_hub import snapshot_download as d; d('$M7'); d('$M3')"
    me=$(readlink -f "$0")
    MAX_HOURS=${MAX_HOURS:-10.5} TRAIN_GPUS=0 bash "$me" e2_sdft > "$EV/gpu0_e2_sdft.log" 2>&1 &
    p0=$!
    ${TIMEOUT_BIN:-timeout} "${GPU1_SECONDS:-39000}" bash -c "
      TRAIN_GPUS=1 EVAL_GPUS=1 MAX_HOURS=6 M_E3='$M3' bash '$me' e3_rest" > "$EV/gpu1_e3_rest.log" 2>&1 &
    p1=$!
    echo "started: GPU0 pid $p0 (e2_sdft), GPU1 pid $p1 (E3 rest); logs in $EV/gpu*.log"
    wait_both "$p0" "$EV/gpu0_e2_sdft.log" "$p1" "$EV/gpu1_e3_rest.log" ;;
  e3_rest)
    # E3 after base / teacher / SFT training are done: SDFT training first (checkpoint is the
    # expensive part), then the two test evaluations. Skips evaluations whose json exists.
    for m in ${M_E3:-"$M3"}; do
      sd="$CK/sdft_science_$(short $m)_n${N_SCI}_ema0.01_forward_s${SEED}"
      [ -f "$sd/DONE" ] || try train_sdft "$m" science "$N_SCI"
      for d in "$CK/sft_science_$(short $m)_n${N_SCI}_s${SEED}/epoch1" "$sd/epoch2"; do
        meth=sft; case "$d" in *sdft_*) meth=sdft ;; esac
        r="test-$(short "$(dirname "$d")")-$(short "$d")"
        [ -f "$EV/$r.json" ] || try eval_new "$m" science "$meth" test "$d" --run_name "$r"
      done
    done ;;
  check_data)
    # HF mirror must be byte-identical (after formatting) to the pinned GitHub copy.
    python - <<'PY2'
import os, sys
sys.path.insert(0, "src")
from data_loader import get_dataset
repo = os.environ.get("SDFT_HF_DATASET")
if not repo:
    sys.exit("SDFT_HF_DATASET is empty: set HF_DATASET_DEFAULT in scripts/kaggle_jobs.sh")
ok = True
for task in ("science", "tool_use"):
    hf = get_dataset(task, source="hf", hf_dataset=repo)
    gh = get_dataset(task, source="github")
    for name, a, b in zip(("train", "val", "test"), hf, gh):
        same = a == b
        ok &= same
        print(f"{task:9s} {name:5s} hf={len(a):5d} github={len(b):5d} identical={same}")
print("DATA OK" if ok else "DATA MISMATCH")
sys.exit(0 if ok else 1)
PY2
    ;;
  day2_a)       # Kin, 20/11 morning (~1.5-2 h): only the SDFT Science evaluation, to save Kin's quota
    try best_epoch_eval "$M7" science sdft "$CK/sdft_science_$(short $M7)_n${N_SCI}_ema0.01_forward_s${SEED}" ;;
  day2_b)       # second person: SDFT Tool, SFT x2, Base. New-task accuracy for all four first (the
    # main tables), prior capabilities afterwards, so a quota cut only loses the least important part.
    export EVAL_BS=${EVAL_BS:-48}   # 3x the default batch: T4 decoding is overhead-bound; OOM halves it
    runs=("tool_use sdft $CK/sdft_tool_use_$(short $M7)_n${N_TOOL}_ema0.01_forward_s${SEED}"
          "science sft $CK/sft_science_$(short $M7)_n${N_SCI}_s${SEED}"
          "tool_use sft $CK/sft_tool_use_$(short $M7)_n${N_TOOL}_s${SEED}")
    for r in "${runs[@]}"; do set -- $r; SKIP_PRIOR=1 try best_epoch_eval "$M7" "$1" "$2" "$3"; done
    SKIP_PRIOR=1 bash "$(readlink -f "$0")" base
    [ -n "${SKIP_PRIOR:-}" ] && exit 0
    try eval_prior "$M7" base-7b
    for r in "${runs[@]}"; do set -- $r
      f="$EV/best-$(short "$3").txt"; if [ -f "$f" ]; then try eval_prior "$M7" "$(short "$3")" "$(cat "$f")"; fi
    done ;;
  final_b)
    # Last GPU session of account B (21/11), after 7B SDFT Tool Use hit CUDA OOM twice (rollout
    # generation at step 16/33, no epoch saved). Evaluations only, most important first; the
    # notebook wraps this in `timeout` below the remaining quota so outputs are always saved.
    export EVAL_BS=${EVAL_BS:-48}
    sft_s="$CK/sft_science_$(short $M7)_n${N_SCI}_s${SEED}/epoch1"
    sft_t="$CK/sft_tool_use_$(short $M7)_n${N_TOOL}_s${SEED}/epoch1"
    sd3="$CK/sdft_science_$(short $M3)_n${N_SCI}_ema0.01_forward_s${SEED}/epoch1"
    done_or() { [ -f "$EV/$1.json" ] || try "${@:2}"; }   # skip evaluations that already exist
    done_or "test-$(short "$(dirname "$sft_s")")-epoch1" eval_new "$M7" science sft test "$sft_s" --run_name "test-$(short "$(dirname "$sft_s")")-epoch1"
    done_or test-base-7b-science eval_new "$M7" science base test "" --run_name test-base-7b-science
    done_or "test-$(short "$(dirname "$sft_t")")-epoch1" eval_new "$M7" tool_use sft test "$sft_t" --run_name "test-$(short "$(dirname "$sft_t")")-epoch1"
    done_or test-base-7b-tool eval_new "$M7" tool_use base test "" --run_name test-base-7b-tool
    try eval_prior "$M7" base-7b
    try eval_prior "$M7" "$(short "$(dirname "$sft_s")")" "$sft_s"
    done_or "test-$(short "$(dirname "$sd3")")-epoch1" env EVAL_GPUS=0 bash "$(readlink -f "$0")" eval_one "$M3" science sdft test "$sd3" --run_name "test-$(short "$(dirname "$sd3")")-epoch1"
    done_or "teacher-$(short $M7)-science" eval_new "$M7" science base val "" --with_demo --run_name "teacher-$(short $M7)-science"
    try eval_prior "$M7" "$(short "$(dirname "$sft_t")")" "$sft_t"
    done_or "teacher-$(short $M7)-tool" eval_new "$M7" tool_use base val "" --with_demo --run_name "teacher-$(short $M7)-tool" ;;
  eval_one) shift; eval_new "$@" ;;   # one evaluation with the current EVAL_GPUS (used by final_b)
  e1_sft)
    train_sft "$M7" science "$N_SCI"
    best_epoch_eval "$M7" science sft "$CK/sft_science_$(short $M7)_n${N_SCI}_s${SEED}" ;;
  e2_sft)
    train_sft "$M7" tool_use "$N_TOOL"
    best_epoch_eval "$M7" tool_use sft "$CK/sft_tool_use_$(short $M7)_n${N_TOOL}_s${SEED}" ;;
  eval_sdft)
    case "${2:?e1 or e2}" in
      e1) best_epoch_eval "$M7" science sdft "$CK/sdft_science_$(short $M7)_n${N_SCI}_ema0.01_forward_s${SEED}" ;;
      e2) best_epoch_eval "$M7" tool_use sdft "$CK/sdft_tool_use_$(short $M7)_n${N_TOOL}_ema0.01_forward_s${SEED}" ;;
    esac ;;
  e3_small)
    # New-task accuracy only (prior capabilities are not needed for the scaling figure).
    for m in ${M_E3:-"$M15" "$M3"}; do
      try eval_new "$m" science base test "" --run_name "test-base-$(short $m)-science"
      try eval_new "$m" science base val "" --with_demo --run_name "teacher-$(short $m)-science"
      try train_sft "$m" science "$N_SCI"
      d="$CK/sft_science_$(short $m)_n${N_SCI}_s${SEED}"
      try eval_new "$m" science sft test "$d/epoch1" --run_name "test-$(short "$d")-epoch1"
      try train_sdft "$m" science "$N_SCI"
      d="$CK/sdft_science_$(short $m)_n${N_SCI}_ema0.01_forward_s${SEED}"
      try eval_new "$m" science sdft test "$d/epoch2" --run_name "test-$(short "$d")-epoch2"
    done ;;
  *) echo "unknown job: $job"; exit 1 ;;
esac
