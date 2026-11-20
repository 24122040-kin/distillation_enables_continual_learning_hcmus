# Hướng dẫn cho Người 2 (tài khoản Kaggle thứ hai)

Bạn chạy **2 notebook**, thêm 1 notebook tùy chọn:

| Notebook | Khi nào | Chạy bao lâu |
|---|---|---|
| **Notebook 1: train** | Ngay bây giờ | ~11–12 tiếng (chạy nền, đóng máy được) |
| **Notebook 2: đánh giá** | Khi notebook 1 xong (khoảng trưa 20/11) | ~6–7 tiếng (chạy nền) |
| Notebook 3: model 1.5B (tùy chọn) | Tối 20/11, nếu còn quota | ~3 tiếng |

**Quota:** notebook 1 + 2 dùng khoảng **19 tiếng** trong 30 tiếng của bạn; thêm notebook 3 thì khoảng 22 tiếng. Kin chỉ còn 16 tiếng, nên bạn nhận luôn phần đánh giá model gốc.

Dữ liệu được tải tự động từ dataset Hugging Face của nhóm (tên đã ghi sẵn trong script), không cần cấu hình gì thêm.

Kin chạy phần của Kin, không liên quan tới bạn. Bạn chỉ cần làm theo file này.

---

## Việc làm một lần trước khi bắt đầu

1. Tài khoản Kaggle đã **xác minh số điện thoại** (avatar → Settings → Phone verification).
2. Quota GPU còn **ít nhất 19 tiếng** (avatar → Settings).
3. Xin Kin **API key W&B**.

---

## NOTEBOOK 1: train (làm ngay)

### Bước 1. Tạo notebook

1. kaggle.com → **Create → New Notebook**. Đặt tên `sdft-B-night`.
2. Panel bên phải → **Session options**:
   - **Accelerator:** chọn `GPU T4 x2`
   - **Internet:** bật `On`
3. Panel bên phải → **Add-ons → Secrets → Add Secret**:
   - Label: `WANDB_API_KEY`
   - Value: key W&B của Kin
   - **Tick ô Attach**
4. Xóa ô code mẫu có sẵn.

### Bước 2. Tạo 3 ô code, copy nguyên văn

**Ô 1**

```python
import os, wandb
from kaggle_secrets import UserSecretsClient
os.environ["WANDB_API_KEY"] = UserSecretsClient().get_secret("WANDB_API_KEY")
os.environ["WANDB_PROJECT"] = "sdft-continual-learning"
wandb.login(key=os.environ["WANDB_API_KEY"], verify=True)
```

**Ô 2**

```python
!rm -rf /kaggle/working/distillation_enables_continual_learning_hcmus
!git clone --depth 1 https://github.com/24122040-kin/distillation_enables_continual_learning_hcmus.git /kaggle/working/distillation_enables_continual_learning_hcmus
%cd /kaggle/working/distillation_enables_continual_learning_hcmus
!git log --oneline -1
!bash scripts/kaggle_jobs.sh setup
```

**Ô 3**

```python
!N_SCI=352 N_TOOL=528 bash scripts/kaggle_jobs.sh night2
```

### Bước 3. Thử trước ô 1 và ô 2 (khoảng 5 phút)

Bấm **Run** ô 1, rồi ô 2. **Không chạy ô 3.**

- Ô 1 in ra `True` hoặc dòng có chữ `Logged in`. Nếu báo lỗi, kiểm tra lại secret đã tick **Attach** chưa.
- Ô 2, dòng cuối có chữ `Tesla T4`. Nếu không có, kiểm tra lại Accelerator đã chọn `GPU T4 x2` chưa.

### Bước 4. Cho chạy nền

1. Bấm **Save Version** (góc trên bên phải) → chọn **Save & Run All (Commit)** → **Save**.
2. Bấm nút **⏻ (Stop session)** ở góc trên để tắt phiên đang mở, cho đỡ tốn quota.
3. Giờ có thể đóng trình duyệt.

### Bước 5. Kiểm tra sau khoảng 30 phút

Mở W&B, project `sdft-continual-learning`. Phải thấy **2 run đang chạy cùng lúc**:

- `sdft_tool_use_Qwen2.5-7B-Instruct_n528_ema0.01_forward_s42`: ô `train/kl_loss` phải là số, không phải `NaN`.
- `sft_science_Qwen2.5-7B-Instruct_n352_s42`: ô `train/sft_loss` phải là số và giảm dần.

Nếu chỉ thấy 1 run, hoặc thấy `NaN`: vào Kaggle → notebook → tab **Versions** → bấm version đang chạy → copy phần log gửi nhóm.

Trong đêm, notebook này tự chạy lần lượt:
- **GPU 0:** SDFT Tool Use 7B
- **GPU 1:** SFT Science 7B → SFT Tool Use 7B → model 3B (SFT và SDFT)

Không cần làm gì thêm.

---

## NOTEBOOK 1b: chạy lại SDFT Tool Use (20/11, sau lỗi CUDA OOM)

Notebook 1 đã chạy xong SFT Science 7B, SFT Tool Use 7B và một phần model 3B, nhưng SDFT Tool Use 7B
chết ở khoảng step 10 vì `CUDA out of memory` (đã sửa trong code) và chưa kịp lưu checkpoint.
Notebook này chạy lại SDFT Tool Use từ đầu trên GPU 0 (khoảng 10–11 tiếng), đồng thời GPU 1 làm nốt
model 3B (SDFT 3B và 2 lần đánh giá test).

1. **Create → New Notebook**, đặt tên `sdft-B-tool`. Accelerator `GPU T4 x2`, Internet `On`,
   tick **Attach** cho `WANDB_API_KEY`.
2. **Add Input** → **Your Work** → **Notebooks** → chọn `sdft-B-night` → **Add**.
3. **Ô 1** và **ô 2** giống hệt notebook 1. Dòng `git log` ở ô 2 phải là commit có chữ `CUDA OOM` hoặc mới hơn.
4. **Ô 3**

   ```python
   !mkdir -p /kaggle/working/checkpoints
   !cp -r /kaggle/input/*/checkpoints/* /kaggle/working/checkpoints/
   !ls /kaggle/working/checkpoints/*
   ```

5. **Ô 4**

   ```python
   !N_SCI=352 N_TOOL=528 bash scripts/kaggle_jobs.sh night3
   ```

6. Chạy thử ô 1, 2, 3. Rồi **Save Version → Save & Run All (Commit) → Save**, bấm **⏻**.

Nếu một GPU lỗi, log chính của notebook sẽ in luôn 40 dòng cuối của log lỗi (không cần chờ notebook xong).

---

## NOTEBOOK 2: đánh giá (khi notebook 1b xong, sáng 21/11)

### Bước 1. Kiểm tra notebook 1 đã xong

Mở tab **Output** → thư mục `checkpoints` của `sdft-B-tool` (thư mục `sdft_tool_use_...`) và của `sdft-B-night` (hai thư mục `sft_...`). Phải có:

- `sdft_tool_use_Qwen2.5-7B-Instruct_n528_ema0.01_forward_s42`, bên trong có `epoch1`, `epoch2` và file `DONE`
- `sft_science_Qwen2.5-7B-Instruct_n352_s42`, bên trong có `epoch1`
- `sft_tool_use_Qwen2.5-7B-Instruct_n528_s42`, bên trong có `epoch1`

Nếu thư mục `sdft_tool_use_...` **không có `DONE`**, xem dòng chú thích trong ô 4 bên dưới.

### Bước 2. Tạo notebook

1. **Create → New Notebook**, đặt tên `sdft-B-day2`.
2. **Session options:** Accelerator `GPU T4 x2`, Internet `On`.
3. **Add-ons → Secrets:** tick **Attach** cho `WANDB_API_KEY` (secret đã có sẵn từ notebook 1).
4. Panel bên phải → **Add Input** → **Your Work** → **Notebooks** → chọn `sdft-B-night` → **Add**; làm lại với `sdft-B-tool` (checkpoint SDFT Tool Use nằm ở notebook 1b).
5. Xóa ô code mẫu.

### Bước 3. Tạo 4 ô code, copy nguyên văn

**Ô 1**

```python
import os, wandb
from kaggle_secrets import UserSecretsClient
os.environ["WANDB_API_KEY"] = UserSecretsClient().get_secret("WANDB_API_KEY")
os.environ["WANDB_PROJECT"] = "sdft-continual-learning"
wandb.login(key=os.environ["WANDB_API_KEY"], verify=True)
```

**Ô 2**

```python
!rm -rf /kaggle/working/distillation_enables_continual_learning_hcmus
!git clone --depth 1 https://github.com/24122040-kin/distillation_enables_continual_learning_hcmus.git /kaggle/working/distillation_enables_continual_learning_hcmus
%cd /kaggle/working/distillation_enables_continual_learning_hcmus
!git log --oneline -1
!bash scripts/kaggle_jobs.sh setup
```

**Ô 3**

```python
!mkdir -p /kaggle/working/checkpoints
!cp -r /kaggle/input/*/checkpoints/* /kaggle/working/checkpoints/
!ls /kaggle/working/checkpoints/*
```

**Ô 4**

```python
# Chỉ khi sdft_tool_use ở bước 1 KHÔNG có file DONE: xóa dấu # ở dòng dưới (train nốt khoảng 1 tiếng)
# !N_TOOL=528 bash scripts/kaggle_jobs.sh e2_sdft
!N_SCI=352 N_TOOL=528 bash scripts/kaggle_jobs.sh day2_b
```

### Bước 4. Thử ô 1, 2, 3 rồi cho chạy nền

1. Bấm **Run** ô 1, ô 2, ô 3. Ô 3 phải in ra danh sách các thư mục ở bước 1.
2. **Save Version → Save & Run All (Commit) → Save**.
3. Bấm **⏻** để tắt phiên đang mở.

Notebook này lần lượt đánh giá:
1. SDFT Tool Use, SFT Science, SFT Tool Use: task mới + 6 benchmark khả năng cũ.
2. Model gốc 7B: task mới cả hai task, teacher (model có ví dụ mẫu trong prompt), 6 benchmark khả năng cũ.

---

## NOTEBOOK 3: model 1.5B (tùy chọn, tối 20/11)

Chỉ làm nếu notebook 2 đã xong và quota còn **ít nhất 4 tiếng**. Không cần Add Input.

1. **Create → New Notebook**, đặt tên `sdft-B-e3-15b`. Accelerator `GPU T4 x2`, Internet `On`, tick **Attach** cho `WANDB_API_KEY`.
2. **Ô 1** và **ô 2** giống hệt notebook 1.
3. **Ô 3**

   ```python
   !N_SCI=352 M_E3=Qwen/Qwen2.5-1.5B-Instruct bash scripts/kaggle_jobs.sh e3_small
   ```

4. **Save Version → Save & Run All (Commit) → Save**, rồi bấm **⏻**.

---

## Khi notebook 2 xong: gửi cho nhóm

1. Kaggle → `sdft-B-day2` → tab **Output** → tải về:
   - thư mục `eval_results`;
   - file `eval_results/FAILED.txt` nếu có (là danh sách các bước bị lỗi).
2. Thêm cả thư mục `eval_results` của notebook 1 (`sdft-B-night` → Output), vì kết quả model 3B nằm ở đó, và của notebook 3 nếu có chạy.
3. Gửi link W&B của project `sdft-continual-learning`.

---

## Gặp lỗi

| Thấy gì | Làm gì |
|---|---|
| `CUDA out of memory` | Sửa ô 3 của notebook 1 thành `!GEN_BS=4 N_SCI=352 N_TOOL=528 bash scripts/kaggle_jobs.sh night2` rồi chạy lại |
| `kl=nan` hoặc `NaN` trên W&B | Dừng notebook, gửi log cho nhóm |
| Ô 1 báo lỗi key | Secret phải tên đúng `WANDB_API_KEY` và đã tick **Attach** |
| Ô 2 không có `Tesla T4` | Accelerator chưa chọn `GPU T4 x2`, hoặc tài khoản chưa xác minh số điện thoại |
| Lỗi tải model / `Connection error` | Internet chưa bật |

**Quy tắc:** luôn dùng đúng `N_SCI=352` và `N_TOOL=528`. Không sửa code trên Kaggle.
