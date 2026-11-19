# Hướng dẫn chạy Tài khoản B và C (Kaggle)

> Viết ngày 19/11/2026. Deadline nộp bài: **23:59 thứ Bảy 21/11**.
> Tài khoản A (SDFT 7B Science) có hướng dẫn riêng: `docs/huong_dan_tai_khoan_A.md`. Các bước tạo notebook, gắn secret W&B, chạy nền và chép checkpoint giống hệt, file này chỉ ghi phần khác.

## Ai chạy gì

| Tài khoản | Thí nghiệm | Job |
|---|---|---|
| **B** | Model gốc 7B (độ chính xác task mới, độ chính xác của teacher, khả năng cũ) · SFT 7B cả hai task · thí nghiệm kích thước model 1.5B / 3B (E3) | `base`, `e1_sft`, `e2_sft`, `e3_small` |
| **C** | SDFT 7B trên Tool Use (E2) | `e2_sdft`, `eval_sdft e2` |

## Lịch

| Thời điểm | B | C |
|---|---|---|
| **Ngay bây giờ (19/11)** | **Notebook B1:** chạy nền `base` (~1–2 h) | **Notebook C1:** chạy nền `e2_sdft` với `N_TOOL=528` (~10 h) |
| **Khi B1 xong** | **Notebook B2:** chạy nền `e1_sft` rồi `e2_sft` (~5–6 h) | (đang chạy) |
| **Sáng 20/11, ~07:00** | **Notebook B3:** chạy nền `e3_small` (~8–9 h) | **Notebook C2:** chép checkpoint, chạy nền `eval_sdft e2` (~1–2 h) |
| **Chiều 20/11** | Gửi kết quả | Gửi kết quả |

**Quota GPU cần:** B khoảng **16 giờ**, C khoảng **12 giờ**. Kiểm tra ở avatar → Settings trước khi bắt đầu.

**Hai số từ pilot của A:**

```text
N_SCI  = 352     (pilot của A, 19/11 22:12: 50,6 s/mẫu)
N_TOOL = 528     (33,4 s/mẫu)
```

Phải dùng **đúng** hai số này ở mọi lệnh của B và C, nếu không SFT và SDFT sẽ train trên hai tập con khác nhau.

---

## Phần chung cho mọi notebook

Mỗi notebook (B1, B2, B3, C1, C2) đều thiết lập như sau. Chi tiết từng thao tác xem hướng dẫn A, bước 0–1:

- **Accelerator:** `GPU T4 x2`
- **Internet:** `On`
- **Secret:** `WANDB_API_KEY` (key W&B của bạn có tài khoản), đã tick **Attach**

**Ô 1: cấu hình** (giống hệt A)

```python
import os, wandb
from kaggle_secrets import UserSecretsClient

os.environ["WANDB_API_KEY"] = UserSecretsClient().get_secret("WANDB_API_KEY")
os.environ["WANDB_PROJECT"] = "sdft-continual-learning"
# os.environ["WANDB_ENTITY"] = "<tên-team>"   # chỉ khi A cũng dùng dòng này
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
wandb.login(key=os.environ["WANDB_API_KEY"], verify=True)
```

**Ô 2: tải code và cài thư viện** (giống hệt A)

```python
!git clone https://github.com/24122040-kin/distillation_enables_continual_learning_hcmus.git
%cd distillation_enables_continual_learning_hcmus
!git log --oneline -1
!bash scripts/kaggle_jobs.sh setup
```

**Ô 3: job.** Khác nhau ở từng notebook, xem bên dưới.

**Cách chạy nền:** bấm **Save Version → Save & Run All (Commit) → Save**, rồi **tắt phiên tương tác** (nút ⏻) để không tốn quota song song. Đóng trình duyệt được.

---

## Tài khoản B

### B1: `base` (chạy ngay, ~1–2 giờ)

Notebook tên `sdft-B1-base`. **Ô 3:**

```python
!bash scripts/kaggle_jobs.sh base
```

Job này chạy theo thứ tự:

| Bước | Tên run trên W&B | Dùng cho slide |
|---|---|---|
| Base 7B trên test Science (507 câu) | `test-base-7b-science` | Table 5, cột Base (paper: 32,1) |
| Base 7B trên test Tool Use (97 câu) | `test-base-7b-tool` | Table 5, cột Base (paper: 42,9) |
| Teacher 7B (có demo trong ngữ cảnh), val Science | `teacher-Qwen2.5-7B-Instruct-science` | Slide 19 (kiểm chứng giả định ICL), E3 |
| Teacher 7B, val Tool Use | `teacher-Qwen2.5-7B-Instruct-tool` | Slide 19 (paper: teacher đạt 100%) |
| 6 benchmark khả năng cũ (lm-eval, 25% số câu) | `prior-base-7b` | Table 5, cột "Previous tasks" của Base (paper: 65,5) |

**Cần kiểm tra:** đây là lần **đầu tiên phần đánh giá chạy fp16 trên 2 GPU** (thay cho 4-bit, vì 4-bit sinh token rất chậm). Khoảng 30 phút sau khi bắt đầu, mở log của version:
- Thanh tiến độ `eval science` nên chạy khoảng vài giây đến 30 giây mỗi batch. Ghi lại tổng thời gian của bước test Science (507 câu) và báo nhóm: con số này quyết định lịch đánh giá ngày mai.
- Thấy các dòng `Building contexts for mmlu_...` rồi thanh tiến độ: ✅ ổn.
- Thấy `!!! FAILED ... eval_prior`: copy khoảng 40 dòng phía trên dòng đó gửi mình. Các bước đánh giá trước đó vẫn giữ kết quả.

Lỗi ở một bước **không** làm dừng cả job; mọi lỗi được ghi vào `eval_results/FAILED.txt`.

### B2: `e1_sft` rồi `e2_sft` (sau khi B1 xong, ~5–6 giờ)

Notebook mới tên `sdft-B2-sft`. **Ô 3**, thay số bằng số pilot:

```python
!N_SCI=352 bash scripts/kaggle_jobs.sh e1_sft
!N_TOOL=528 bash scripts/kaggle_jobs.sh e2_sft
```

Hai dòng nằm chung một ô: nếu dòng đầu lỗi, dòng sau vẫn chạy.

- Nếu Kaggle cho chạy song song với B1 thì bấm luôn. Nếu báo đã đạt giới hạn phiên GPU thì đợi B1 xong.
- Log bình thường trông như:
  ```text
  [ep 1 step 12/86] nll=0.4127
  ```
  `nll` giảm dần trong vài chục bước đầu là đúng.
- Mỗi job tự chạy: train 1 epoch → đánh giá trên val → test → lm-eval.
  - Run Science: `sft_science_Qwen2.5-7B-Instruct_n<N_SCI>_s42`, `test-sft_science_...-epoch1`, `prior-sft_science_...`
  - Run Tool Use: `sft_tool_use_...`, `test-sft_tool_use_...-epoch1`, `prior-sft_tool_use_...`

### B3: `e3_small` (sáng 20/11, ~8–9 giờ)

Notebook mới tên `sdft-B3-e3`. **Ô 3:**

```python
!N_SCI=352 bash scripts/kaggle_jobs.sh e3_small
```

- Chạy lần lượt cho **1.5B** rồi **3B**: Base trên test, teacher trên val, SFT rồi test, SDFT rồi test. Không chạy lm-eval (hình về kích thước model chỉ cần độ chính xác task mới).
- **Nếu bắt đầu B3 muộn hơn 12:00 ngày 20/11**, chỉ chạy 1.5B để kịp xong trước tối:
  ```python
  !N_SCI=352 M_E3=Qwen/Qwen2.5-1.5B-Instruct bash scripts/kaggle_jobs.sh e3_small
  ```
- Kết quả dùng cho slide 30: SFT so với SDFT theo kích thước model (paper: chênh −3,3 ở 3B, +4,0 ở 7B).

---

## Tài khoản C

### C1: `e2_sdft` (tối 19/11, ~10 giờ)

1. Tạo notebook `sdft-C1-e2-sdft`. Chạy thử **ô 1 và ô 2** để kiểm tra GPU và W&B (ô 2 in ra `Tesla T4`).
2. Viết **ô 3**:
   ```python
   !N_TOOL=528 bash scripts/kaggle_jobs.sh e2_sdft
   ```
3. Bấm **Save & Run All**, rồi tắt phiên tương tác.
4. Theo dõi trên W&B, run `sdft_tool_use_Qwen2.5-7B-Instruct_n<N_TOOL>_ema0.01_forward_s42`, giống bảng ở bước 5 của hướng dẫn A:
   - `train/kl_loss` phải là số, không phải NaN;
   - `time/step_sec` ổn định;
   - `train/truncated_frac` nhỏ.
5. Hết bộ nhớ GPU (`CUDA out of memory`) thì thay ô 3 bằng `!GEN_BS=4 N_TOOL=528 bash scripts/kaggle_jobs.sh e2_sdft` rồi chạy lại.

### C2: `eval_sdft e2` (sáng 20/11, ~1–2 giờ)

Làm giống bước 6 của hướng dẫn A, chỉ thay tên. Phần đánh giá giờ chạy fp16 trên 2 GPU nên nhanh hơn trước.

1. Kiểm tra output của C1: có `checkpoints/sdft_tool_use_..._s42/epoch1/` và `epoch2/`, có file `DONE`.
2. Tạo notebook `sdft-C2-e2-eval`. **Add Input → Your Work → Notebooks →** `sdft-C1-e2-sdft`.
3. Ô 1, ô 2 như phần chung. **Ô 3: chép checkpoint**
   ```python
   !mkdir -p /kaggle/working/checkpoints
   !cp -r /kaggle/input/*/checkpoints/* /kaggle/working/checkpoints/
   !ls /kaggle/working/checkpoints/*
   ```
4. **Ô 4:**
   ```python
   # Nếu C1 chưa có DONE (hết giờ giữa epoch 2), bỏ dấu # dòng dưới để train tiếp trước:
   # !N_TOOL=528 bash scripts/kaggle_jobs.sh e2_sdft
   !N_TOOL=528 bash scripts/kaggle_jobs.sh eval_sdft e2
   ```
5. Bấm **Save & Run All**, tắt phiên tương tác.

---

## Gửi kết quả (B và C, khi mỗi notebook xong)

1. Tải file từ tab **Output**:
   - `eval_results/*.json`: mỗi file là một lần đánh giá, có `acc`, `paper_cmp` và từng câu trả lời;
   - `eval_results/prior/`: kết quả lm-eval;
   - `eval_results/FAILED.txt` nếu có.
2. Gửi vào nhóm:
   - link W&B của các run;
   - số giờ chạy (`time/elapsed_h` cho run train);
   - mọi sự cố đã gặp, để ghi vào slide phân tích sai lệch.

## Gặp lỗi

| Triệu chứng | Cách xử lý |
|---|---|
| `CUDA out of memory` khi chạy SDFT | Thêm `GEN_BS=4` vào đầu lệnh (vẫn lỗi thì `GEN_BS=2`) |
| `kl=nan` trong log SDFT | Dừng lại, gửi log, không để chạy tiếp |
| `!!! FAILED ... eval_prior` | Gửi khoảng 40 dòng log phía trên. Các kết quả khác vẫn còn |
| Ô 1 báo key W&B sai | Kiểm tra secret tên `WANDB_API_KEY` và đã tick **Attach** |
| Không có GPU / không tải được model | Kiểm tra Accelerator `GPU T4 x2`, Internet `On`, tài khoản đã xác minh số điện thoại |
| Run chết giữa chừng | SDFT: chép checkpoint sang notebook mới như C2 rồi chạy lại đúng lệnh cũ, nó tự train tiếp. SFT: chạy lại từ đầu (~1–2,5 h) |
| Quota sắp hết | Báo nhóm để chuyển job (ưu tiên giữ E1 và E2, bỏ 3B ở E3 trước) |

## Quy tắc

- Dùng **đúng** `N_SCI` / `N_TOOL` từ pilot của A cho mọi lệnh.
- Không sửa code trên Kaggle. Ô 2 in commit đang dùng; mọi tài khoản phải cùng một commit.
- Seed giữ mặc định `42`.
