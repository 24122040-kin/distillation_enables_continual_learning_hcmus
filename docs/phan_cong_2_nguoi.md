# Phân công 2 người / 2 tài khoản Kaggle

> Cập nhật 20/11/2026, thay cho lịch 3 tài khoản. Deadline: **23:59 thứ Bảy 21/11**.
> Số từ pilot (19/11 22:12): **`N_SCI=352`, `N_TOOL=528`**. Mọi lệnh dùng đúng hai số này.
> Cách tạo notebook, gắn secret W&B, chạy nền và chép checkpoint vẫn như trong `docs/huong_dan_tai_khoan_A.md` (bước 0–2 và 6.2). File này chỉ ghi ai chạy gì, khi nào.

## Ý tưởng

Mỗi tài khoản Kaggle có **2 GPU T4**. Job train chỉ dùng 1 GPU, nên tài khoản của Người 2 chạy **2 job song song** trong cùng một phiên đêm nay:
- GPU 0: SDFT 7B Tool Use
- GPU 1: SFT 7B cho cả hai task, rồi đến E3 với model 3B

Phần đánh giá model 7B cần cả 2 GPU (fp16), nên làm vào sáng mai.

## Lịch

| Thời điểm | Người 1: Kin (tài khoản A) | Người 2 (tài khoản B) |
|---|---|---|
| **Đêm 19→20/11** | `e1_sdft` (đang chạy, xong khoảng 09:00) | **`night2`**: GPU 0 chạy `e2_sdft`, GPU 1 chạy SFT 7B ×2 rồi E3 3B (~11 h, xong khoảng 12:00) |
| **Sáng 20/11, ~09:00** | **`day2_a`**: đánh giá SDFT Science + đánh giá model gốc (~3–4 h) | (đang chạy `night2`) |
| **Trưa 20/11, ~12:00** | (xong khoảng 13:00) | **`day2_b`**: đánh giá SDFT Tool, SFT Science, SFT Tool (~4–5 h) |
| **Chiều 20/11** (tùy chọn) | E3 với 1.5B nếu còn quota (~3 h) | |
| **Trong lúc GPU chạy** | Viết slide 1–13 (bài toán, các hướng đi trước, đóng góp, ký hiệu, pipeline) | Viết slide 14–26 (thành phần, lý thuyết, độ phức tạp, dữ liệu, kết quả của paper, setup) |
| **21/11** | Slide 27–30 (kết quả E1, E2, E3) | Slide 31–33 (phân tích sai lệch, so sánh với code gốc, kết luận) + README + checklist |
| **21/11 20:00** | **Nộp** (pptx + link repo), còn 4 giờ dự phòng | ← |

**Quota GPU cần:** Người 1 khoảng 16 giờ (19 giờ nếu chạy thêm 1.5B), Người 2 khoảng 17 giờ.

---

## Người 2: notebook `sdft-B-night` (chạy ngay)

Thiết lập: `GPU T4 x2`, Internet `On`, secret `WANDB_API_KEY` (key của Kin) đã tick **Attach**.

**Ô 1, ô 2:** giống hệt hướng dẫn A, bước 2. Dòng `CUDA_VISIBLE_DEVICES` trong ô 1 không ảnh hưởng gì, vì script tự chia GPU.

**Ô 3:**

```python
!N_SCI=352 N_TOOL=528 bash scripts/kaggle_jobs.sh night2
```

Sau đó bấm **Save Version → Save & Run All (Commit) → Save**, rồi **tắt phiên tương tác**.

**Job này làm gì:**
1. Tải model 7B một lần (~10 phút).
2. Chạy song song:
   - **GPU 0:** SDFT Tool Use. Run W&B tên `sdft_tool_use_Qwen2.5-7B-Instruct_n528_ema0.01_forward_s42`. Tự dừng sau tối đa 10,5 giờ.
   - **GPU 1:** lần lượt
     - SFT Science (`sft_science_..._n352_s42`)
     - SFT Tool Use (`sft_tool_use_..._n528_s42`)
     - E3 với 3B: Base trên test, teacher trên val, SFT rồi test, SDFT rồi test (`sdft_science_Qwen2.5-3B-Instruct_...`)

   GPU 1 có giới hạn cứng 10,8 giờ để cả phiên kết thúc trước mốc 12 giờ của Kaggle.
3. Log chi tiết nằm trong output: `eval_results/gpu0_e2_sdft.log` và `eval_results/gpu1_chain.log`.

**Theo dõi:** trong lúc chạy, log của Kaggle chỉ hiện dòng `started: GPU0 ... GPU1 ...`. Muốn xem tiến độ thì mở **W&B**. Khoảng 30 phút sau khi bắt đầu, phải thấy **cùng lúc** hai run `sdft_tool_use_...` và `sft_science_...` đang ghi số liệu.

## Người 2: notebook `sdft-B-day2` (trưa 20/11)

1. Mở output của `sdft-B-night` và kiểm tra `checkpoints/` có đủ:
   - `sdft_tool_use_..._n528_.../`: có `epoch1/`, `epoch2/`, file `DONE`
   - `sft_science_..._n352_s42/epoch1/`
   - `sft_tool_use_..._n528_s42/epoch1/`
2. Tạo notebook mới, thiết lập như trên. **Add Input → Your Work → Notebooks →** `sdft-B-night`.
3. **Ô 1, ô 2** như cũ. **Ô 3: chép checkpoint**

   ```python
   !mkdir -p /kaggle/working/checkpoints
   !cp -r /kaggle/input/*/checkpoints/* /kaggle/working/checkpoints/
   !ls /kaggle/working/checkpoints/*
   ```

4. **Ô 4:**

   ```python
   # Nếu sdft_tool_use chưa có DONE (hết giờ giữa epoch 2), bỏ dấu # dòng dưới để train tiếp trước (~1 h):
   # !N_TOOL=528 bash scripts/kaggle_jobs.sh e2_sdft
   !N_SCI=352 N_TOOL=528 bash scripts/kaggle_jobs.sh day2_b
   ```

5. Bấm **Save & Run All**, tắt phiên tương tác.

## Người 1 (Kin): notebook `sdft-A-day2` (sáng 20/11)

Làm như bước 6 trong hướng dẫn A. Chỉ đổi **ô 4** thành:

```python
# Nếu e1_sdft chưa có DONE, bỏ dấu # dòng dưới để train tiếp trước:
# !N_SCI=352 bash scripts/kaggle_jobs.sh e1_sdft
!N_SCI=352 bash scripts/kaggle_jobs.sh day2_a
```

`day2_a` đánh giá SDFT Science (chọn epoch trên val, rồi test, rồi lm-eval), sau đó chạy toàn bộ đánh giá model gốc: test Science và Tool Use, teacher có demo, lm-eval.

**Tùy chọn, chiều 20/11:** nếu còn quota, chạy E3 với 1.5B trong một notebook mới (không cần input):

```python
!N_SCI=352 M_E3=Qwen/Qwen2.5-1.5B-Instruct bash scripts/kaggle_jobs.sh e3_small
```

---

## Mỗi kết quả dùng cho slide nào

| Kết quả | Từ đâu | Slide |
|---|---|---|
| Base / SFT / SDFT trên task mới, Science và Tool Use | `test-*.json`, W&B `test-...` | 27, 29 |
| Khả năng cũ (6 benchmark) | `eval_results/prior/`, W&B `prior-...` | 28, 29 |
| Teacher có demo (7B, 3B) | `teacher-*.json` | 19, 30 |
| 3B (và 1.5B) SFT / SDFT | `test-*-3B-*.json` | 30 |
| Thời gian chạy thật | W&B `time/*` | 20, 31 |

## Nếu có sự cố

- **Một bước lỗi:** các bước khác vẫn chạy tiếp; lỗi ghi vào `eval_results/FAILED.txt`. Gửi file đó cùng log cho nhóm.
- **Hết bộ nhớ GPU (`CUDA out of memory`) ở job SDFT:** thêm `GEN_BS=4` vào đầu lệnh rồi chạy lại.
- **Thiếu quota:** ưu tiên giữ E1 và E2, bỏ 1.5B trước, sau đó bỏ 3B.
