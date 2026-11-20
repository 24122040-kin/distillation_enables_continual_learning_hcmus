# Cần thu thập gì khi một job bị lỗi

Tình huống hiện tại (20/11): notebook `sdft-B-night` báo
`!!! GPU0 job exited with an error, see /kaggle/working/eval_results/gpu0_e2_sdft.log`
sau khoảng 1h38 phút. Job SDFT Tool Use 7B (GPU0) đã chết. Chuỗi GPU1 (SFT + model 3B) vẫn chạy tiếp.

**Không bấm Stop / Cancel notebook đang chạy.** File trong `/kaggle/working` chỉ xuất hiện ở tab Output
**khi notebook chạy xong**. Cancel giữa chừng có thể làm mất toàn bộ checkpoint và log.

Thu thập theo thứ tự dưới đây, gửi lại cho nhóm. Mục A làm được **ngay bây giờ**. Mục B chỉ làm được
**sau khi notebook chạy xong**.

---

## A. Trên W&B (làm ngay)

Vào wandb.ai (giờ là CoreWeave Forge, đăng nhập bằng tài khoản cũ) → project `sdft-continual-learning`.

### A1. Run bị lỗi: `sdft_tool_use_Qwen2.5-7B-Instruct_n528_ema0.01_forward_s42`

| # | Lấy gì | Ở đâu | Để làm gì |
|---|---|---|---|
| 1 | **Trạng thái run** (`crashed` / `failed` / `running`) | Danh sách run, cột State | Biết process bị giết (crashed) hay báo lỗi Python (failed) |
| 2 | **30–50 dòng cuối của log** (copy text, không chỉ chụp ảnh) | Thanh trái → icon **Logs** (`>_`) → kéo xuống cuối | Thấy traceback, ví dụ `CUDA out of memory`, `Killed`, lỗi Python |
| 3 | **Step cuối cùng** đã log | Chart `train/kl_loss`, rê chuột vào điểm cuối | Biết checkpoint resume ở step 10 đã được lưu hay chưa (lưu ở step 10, 20, ...) |
| 4 | **Giá trị cuối** của `train/kl_loss`, `train/grad_norm`, `train/resp_tokens_mean`, `train/truncated_frac`, `time/step_sec` | Tab Overview → Summary, hoặc các chart | Loại trừ lỗi NaN; thấy response có quá dài không |
| 5 | **Ảnh chụp chart RAM**: `System Memory Utilization (%)` và `Process Memory In Use (non-swap) (MB)` | Thanh trái → icon **System** (hình con chip) | Nếu chạm gần 100% ngay trước khi chết → máy hết RAM, không phải lỗi code |
| 6 | **Ảnh chụp chart GPU**: `GPU Memory Allocated (%)` | Cùng tab System | Nếu chạm 100% → hết VRAM (`CUDA out of memory`) |

### A2. Các run trên GPU1 cùng notebook (để biết GPU1 còn khỏe không)

Với từng run dưới đây, ghi **trạng thái**, **step cuối / tổng số step**, và chụp chart
**System Memory Utilization (%)**:

- `sft_science_Qwen2.5-7B-Instruct_n352_s42`
- `sft_tool_use_Qwen2.5-7B-Instruct_n528_s42`
- các run có chữ `Qwen2.5-3B-Instruct` (nếu đã bắt đầu)

Quan trọng nhất là **thời điểm** (giờ:phút) mỗi run bắt đầu và kết thúc, xem ở tab Overview.
Nếu một run GPU1 bắt đầu đúng lúc run SDFT Tool chết, thì nguyên nhân rất có thể là hết RAM
do hai process cùng load model.

### A3. Run của Kin: `sdft_science_Qwen2.5-7B-Instruct_n352_ema0.01_forward_s42`

Ghi **trạng thái** và **step cuối / 44** (2 epoch × 22 step). Nếu nó cũng `crashed`, lấy đủ 6 mục như A1.

---

## B. Trên Kaggle, sau khi notebook chạy xong

Kaggle → notebook → tab **Output** (hoặc tab Versions → bấm version → Output).

### B1. Tải về thư mục `eval_results/`, trong đó quan trọng nhất là:

| File | Có gì |
|---|---|
| `gpu0_e2_sdft.log` | Toàn bộ log của SDFT Tool Use, có traceback đầy đủ |
| `gpu1_chain.log` | Log của SFT Science, SFT Tool Use, model 3B |
| `FAILED.txt` | Danh sách các bước bị lỗi (nếu có) |
| các file `*.json` | Kết quả đánh giá đã chạy xong |

### B2. Chụp màn hình cây thư mục `checkpoints/`

Mở từng thư mục và chụp, cần thấy:

- `sdft_tool_use_Qwen2.5-7B-Instruct_n528_ema0.01_forward_s42/` → có thư mục `resume/` không?
  Nếu có, **mở `resume/state.json` và copy nội dung** (cho biết `global_step` đã lưu).
- `sft_science_Qwen2.5-7B-Instruct_n352_s42/` → có `epoch1/` không?
- `sft_tool_use_Qwen2.5-7B-Instruct_n528_s42/` → có `epoch1/` không?
- các thư mục có chữ `3B` → có `epoch1/`, `epoch2/`, `DONE` không?

### B3. Log chính của notebook

Tab Versions → bấm version → phần **Log**: copy **20 dòng cuối** (dòng `tail -n 5` ở cuối in ra
5 dòng cuối của cả hai log GPU).

---

## C. Thông tin chung (ghi một lần)

- Giờ bắt đầu notebook (giờ Việt Nam) và giờ thấy lỗi.
- Quota GPU còn lại của tài khoản (avatar → Settings → GPU quota).
- Notebook đang dùng commit code nào: dòng in ra bởi `git log --oneline -1` ở ô 2.

---

## Mẫu tin nhắn gửi lại nhóm

```
[sdft_tool_use 7B]  State: ___   Step cuối: ___/66   kl_loss cuối: ___
Log cuối (A1.2):
<dán 30-50 dòng>
RAM cuối (A1.5): ___%   GPU mem cuối (A1.6): ___%
[GPU1] sft_science: ___ (___ / ___ step, xong lúc __:__)
       sft_tool_use: ___ (___ / ___ step, bắt đầu __:__)
       3B: ___
[Kin] sdft_science: ___ (___/44 step)
Quota còn: ___ h
```
