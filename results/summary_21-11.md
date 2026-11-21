## Slide 22: new-task accuracy, 7B (test split)

| Task / Method | Paper (%) | Ours (%) | 95% CI (pp) | Δ (pp) | Rel. dev. (%) | Epoch | n | Truncated (%) | W&B run |
|---|---|---|---|---|---|---|---|---|---|
| Science: Base | 32.1 | 32.5 | ±4.1 | +0.4 | 1.4 | - | 507 | 0.2 | `test-base-7b-science` |
| Science: SFT | 66.2 | 36.5 | ±4.2 | -29.7 | 44.9 | epoch1 | 507 | 1.8 | `test-sft_science_Qwen2.5-7B-Instruct_n352_s42-epoch1` |
| Science: SDFT | 70.2 | 41.2 | ±4.3 | -29.0 | 41.3 | epoch2 | 507 | 0.6 | `test-sdft_science_Qwen2.5-7B-Instruct_n352_ema0.01_forward_s42-epoch2` |
| Tool Use: Base | 42.9 | 41.2 | ±9.8 | -1.7 | 3.9 | - | 97 | 0.0 | `test-base-7b-tool` |
| Tool Use: SFT | 63.2 | 62.9 | ±9.6 | -0.3 | 0.5 | epoch1 | 97 | 0.0 | `test-sft_tool_use_Qwen2.5-7B-Instruct_n528_s42-epoch1` |
| Tool Use: SDFT | 70.6 | not run | | | | | | | |

## Paired tests on the same test questions

| Comparison (B − A) | Δ (pp) | Paired bootstrap 95% CI | only B / only A correct | McNemar p | n |
|---|---|---|---|---|---|
| 3B science: SFT − Base | +6.9 | [+2.0, +11.8] | 97 / 62 | 0.007 | 507 |
| 7B science: SFT − Base | +3.9 | [-1.2, +8.9] | 92 / 72 | 0.138 | 507 |
| 7B science: SDFT − Base | +8.7 | [+3.6, +13.8] | 108 / 64 | 0.001 | 507 |
| 7B science: SDFT − SFT | +4.7 | [+0.2, +9.1] | 79 / 55 | 0.047 | 507 |
| 7B tool_use: SFT − Base | +21.6 | [+12.4, +32.0] | 25 / 4 | 0.000 | 97 |

## Slide 23: prior capabilities, 7B (lm-eval, limit 0.25; Paper → Ours)

| Model | hellaswag | humaneval | ifeval | mmlu | truthfulqa_mc2 | winogrande | Avg | W&B run |
|---|---|---|---|---|---|---|---|---|
| Base 7B | 62.0 → 52.9 | 65.8 → 90.2 | 74.3 → 68.4 | 71.7 → 69.1 | 47.9 → 59.8 | 71.1 → 63.4 | 65.5 → 67.3 | `prior-base-7b, prior-base-7b, prior-base-7b-humaneval` |
| SFT (Science) | 55.0 → 56.1 | 54.8 → 92.7 | 35.3 → 66.2 | 64.6 → 69.3 | 36.8 → 59.7 | 73.7 → 64.7 | 53.4 → 68.1 | `prior-sft_science_Qwen2.5-7B-Instruct_n352_s42` |
| SDFT (Science) | 60.9 → 55.8 | 68.9 → 92.7 | 66.8 → 71.3 | 70.7 → 68.7 | 46.5 → 60.7 | 73.1 → 65.0 | 64.5 → 69.0 | `prior-sdft_science_Qwen2.5-7B-Instruct_n352_ema0.01_forward_s42` |
| SFT (Tool Use) | T5 → 63.2 | T5 → 90.2 | T5 → 61.8 | T5 → 72.1 | T5 → 57.6 | T5 → 71.9 | 56.0 → 69.5 | `prior-sft_tool_use_Qwen2.5-7B-Instruct_n528_s42` |
| SDFT (Tool Use) | T5 → not run | T5 → not run | T5 → not run | T5 → not run | T5 → not run | T5 → not run | 65.4 → not run |  |

## Slide 24: model size on Science (E3)

| Model | Base (test) | Teacher (val, with demo) | SFT (test) | SDFT (test) | SDFT − SFT (pp) | Paper (pp) |
|---|---|---|---|---|---|---|
| Qwen2.5-3B | 27.8 | 89.6 | 34.7 (epoch1) | — | — | chỉ có trên hình (Fig. 5), không có số |
| Qwen2.5-7B | 32.5 | 92.5 | 36.5 (epoch1) | 41.2 (epoch2) | +4.7 | +4.0 (Table 5) |

## Every new-task evaluation found

| Model | Task | Method | Split | Epoch | Acc (%) | n | File |
|---|---|---|---|---|---|---|---|
| 3B | science | base | test | - | 27.8 | 507 | `test-base-Qwen2.5-3B-Instruct-science.json` |
| 3B | science | sft | test | epoch1 | 34.7 | 507 | `test-sft_science_Qwen2.5-3B-Instruct_n352_s42-epoch1.json` |
| 3B | science | teacher | val | - | 89.6 | 134 | `teacher-Qwen2.5-3B-Instruct-science.json` |
| 7B | science | base | test | - | 32.5 | 507 | `test-base-7b-science.json` |
| 7B | science | sdft | test | epoch2 | 41.2 | 507 | `test-sdft_science_Qwen2.5-7B-Instruct_n352_ema0.01_forward_s42-epoch2.json` |
| 7B | science | sdft | val | epoch2 | 41.8 | 134 | `val-sdft_science_Qwen2.5-7B-Instruct_n352_ema0.01_forward_s42-epoch2.json` |
| 7B | science | sft | test | epoch1 | 36.5 | 507 | `test-sft_science_Qwen2.5-7B-Instruct_n352_s42-epoch1.json` |
| 7B | science | teacher | val | - | 92.5 | 134 | `teacher-Qwen2.5-7B-Instruct-science.json` |
| 7B | tool_use | base | test | - | 41.2 | 97 | `test-base-7b-tool.json` |
| 7B | tool_use | sft | test | epoch1 | 62.9 | 97 | `test-sft_tool_use_Qwen2.5-7B-Instruct_n528_s42-epoch1.json` |
| 7B | tool_use | teacher | val | - | 55.9 | 202 | `teacher-Qwen2.5-7B-Instruct-tool.json` |
