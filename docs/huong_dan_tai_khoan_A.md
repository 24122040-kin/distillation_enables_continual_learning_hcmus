# Hướng dẫn chạy Tài khoản A (Kaggle)

> Viết ngày 19/11/2026. Deadline nộp bài: **23:59 thứ Bảy 21/11**.
> Tài khoản A chạy thí nghiệm quan trọng nhất của deck: **SDFT trên Qwen2.5-7B, task Science Q&A (E1)**.
> Lịch tổng của cả nhóm nằm trong `docs/reproduction_plan.md` §5.

## Tóm tắt việc của A

| Khi nào | Việc | Thời gian | Chạy kiểu |
|---|---|---|---|
| **Tối 19/11** | Bước 1–3: tạo notebook, chạy `pilot`, gửi số `N_SCI` / `N_TOOL` cho nhóm | ~1 giờ | Tương tác (ngồi xem) |
| **Tối 19/11** | Bước 4: chạy `e1_sdft` qua đêm | ~10 giờ | Chạy nền, đóng trình duyệt được |
| **Sáng 20/11** | Bước 6: chạy `eval_sdft e1` | ~2–3 giờ | Chạy nền |
| **Trưa 20/11** | Bước 7: gửi kết quả cho người làm slide | 5 phút | |

Quota GPU cần cho A: khoảng **14 giờ** trong tuần này.

---

## Bước 0. Chuẩn bị (làm một lần)

1. **Xác minh số điện thoại cho tài khoản Kaggle:** avatar → **Settings → Phone verification**. Chưa xác minh thì không bật được GPU và Internet.
2. **Kiểm tra quota GPU:** avatar → **Settings**, xem mục GPU quota. Còn dưới 14 giờ thì báo nhóm ngay để đổi tài khoản.
3. **Lấy API key W&B** (người có tài khoản W&B làm): đăng nhập W&B, vào `wandb.ai/authorize` hoặc **User Settings → API Keys**, rồi copy key.
   - Key là mật khẩu, **không dán vào nhóm chat**. Người có key tự nhập ở bước 1.4, hoặc gửi tin nhắn riêng.

## Bước 1. Tạo notebook

1. Vào kaggle.com → **Create → New Notebook**. Đặt tên: `sdft-A-e1-sdft`.
2. Ở panel bên phải, mục **Session options** (hoặc **Settings**):
   - **Accelerator:** `GPU T4 x2`
   - **Internet:** `On`
3. Bấm **Add-ons → Secrets → Add Secret**:
   - Label: `WANDB_API_KEY`
   - Value: key W&B ở bước 0.3
   - **Tick ô "Attach"** để notebook dùng được secret này.
4. Xóa ô code mẫu Kaggle tạo sẵn.

## Bước 2. Tạo 4 ô code

Mỗi khối dưới đây là **một ô riêng**, giữ đúng thứ tự.

**Ô 1: cấu hình**

```python
import os, wandb
from kaggle_secrets import UserSecretsClient

os.environ["WANDB_API_KEY"] = UserSecretsClient().get_secret("WANDB_API_KEY")
os.environ["WANDB_PROJECT"] = "sdft-continual-learning"
# os.environ["WANDB_ENTITY"] = "<tên-team>"   # chỉ khi W&B dùng team; mặc định là tài khoản cá nhân
os.environ["CUDA_VISIBLE_DEVICES"] = "0"       # mỗi job dùng 1 GPU để thời gian đo khớp ước tính
wandb.login(key=os.environ["WANDB_API_KEY"], verify=True)
```

**Ô 2: tải code và cài thư viện** (~5 phút)

```python
!git clone https://github.com/24122040-kin/distillation_enables_continual_learning_hcmus.git
%cd distillation_enables_continual_learning_hcmus
!git log --oneline -1
!bash scripts/kaggle_jobs.sh setup
```

**Ô 3: pilot** (chỉ chạy tương tác, **xóa trước khi chạy nền**)

```python
!bash scripts/kaggle_jobs.sh pilot
```

**Ô 4: job chính** (tạm để trống, điền ở bước 4)

```python
!N_SCI=____ bash scripts/kaggle_jobs.sh e1_sdft
```

## Bước 3. Chạy pilot (tối 19/11, ~30–40 phút)

1. Bấm **Run** lần lượt **ô 1 → ô 2 → ô 3**. Không chạy ô 4.
2. Kiểm tra từng ô:
   - **Ô 1** in ra dòng có `Logged in` hoặc `True`. Nếu báo lỗi key thì kiểm tra lại secret đã được tick "Attach" chưa.
   - **Ô 2**, dòng cuối in ra dạng `2.x.x Tesla T4`. Không thấy `Tesla T4` thì kiểm tra lại Accelerator.
   - **Ô 3** chạy lâu nhất. Khoảng 10 phút đầu là tải model 7B (~15 GB). Sau đó mỗi task chạy 3 bước train, in ra các dòng:
     ```text
     [ep 1 step 2/3] kl=0.0312 len=412 gen=78s train=31s
     ```
     **`kl=` phải là một con số.** Nếu thấy `kl=nan`, xem mục "Gặp lỗi" ở cuối file.
3. Cuối ô 3 có 2 dòng quan trọng, ví dụ:
   ```text
   science: 13.0 s/example -> 2 epochs of full data = 18.3 h ; use N_SCI=1376
   tool_use: 6.9 s/example -> 2 epochs of full data = 14.7 h ; use N_TOOL=2608
   ```
   - **Chụp màn hình hai dòng này, gửi ngay vào nhóm.** B và C phải dùng **đúng hai số này**.
   - Nếu dòng `science` ghi `full data = ... h` **≤ 10 h**: dùng `N_SCI=2540` (chạy đủ dữ liệu).

## Bước 4. Chạy `e1_sdft` qua đêm

1. **Xóa ô 3** (pilot). Nếu để lại, lần chạy nền sẽ chạy lại pilot và mất thêm 40 phút.
2. Sửa ô 4, thay `____` bằng số `N_SCI` của bước 3, ví dụ:
   ```python
   !N_SCI=1376 bash scripts/kaggle_jobs.sh e1_sdft
   ```
3. Bấm **Save Version** (góc trên bên phải) → chọn **Save & Run All (Commit)** → **Save**.
   - Kaggle sẽ chạy lại toàn bộ notebook từ ô 1 trên máy riêng. **Đóng trình duyệt cũng không sao.**
4. **Tắt phiên tương tác:** bấm nút nguồn ⏻ ở góc trên, hoặc **Run → Stop session**. Nếu để mở, phiên này vẫn ăn quota song song với lần chạy nền.
5. Ghi lại giờ bắt đầu. Job tự dừng sau tối đa **11 giờ train** và lưu checkpoint, không bị Kaggle cắt ngang.

## Bước 5. Theo dõi (thỉnh thoảng mở xem)

**Trên W&B**, project `sdft-continual-learning`, run tên `sdft_science_Qwen2.5-7B-Instruct_n<N_SCI>_ema0.01_forward_s42`:

| Biểu đồ | Bình thường | Bất thường |
|---|---|---|
| `train/kl_loss` | Một con số, dao động hoặc giảm dần | `NaN`, hoặc tăng vọt liên tục |
| `time/step_sec` | Ổn định, xấp xỉ `16 × s/example` của pilot | Tăng dần theo thời gian |
| `train/truncated_frac` | Nhỏ (< 0,2) | Gần 1 (câu trả lời luôn bị cắt) |
| `epoch` | Tăng đều lên 2.0 | Đứng yên |

**Trên Kaggle:** mở notebook → tab **Versions** → bấm vào version đang chạy để xem log.

## Bước 6. Sáng 20/11: đánh giá (`eval_sdft e1`)

### 6.1 Kiểm tra run đêm qua đã xong chưa

Trên Kaggle, mở version đêm qua → tab **Output** → thư mục `checkpoints/sdft_science_..._s42/`:

| Thấy gì | Nghĩa là | Làm gì |
|---|---|---|
| Có `DONE`, `epoch1/`, `epoch2/` | Train xong | Sang 6.2 |
| Chỉ có `epoch1/` + `resume/`, không có `DONE` | Hết 11 giờ khi đang ở epoch 2 | Sang 6.2, nhưng ở 6.3 chạy **`e1_sdft` trước** (nó tự train tiếp), rồi mới chạy `eval_sdft e1` |
| Không có thư mục `checkpoints` | Run lỗi | Mở log của version, copy ~30 dòng cuối gửi nhóm |

### 6.2 Tạo notebook đánh giá

1. **Create → New Notebook**, đặt tên `sdft-A-e1-eval`. Thiết lập giống bước 1: `GPU T4 x2`, Internet `On`, gắn secret `WANDB_API_KEY`.
2. **Add Input** (panel phải) → **Your Work** → **Notebooks** → chọn `sdft-A-e1-sdft` → **Add**. Output của run đêm qua sẽ nằm trong `/kaggle/input/...`.
3. Copy **ô 1** và **ô 2** từ bước 2 sang.
4. Thêm **ô 3: chép checkpoint về**:
   ```python
   !mkdir -p /kaggle/working/checkpoints
   !cp -r /kaggle/input/*/checkpoints/* /kaggle/working/checkpoints/
   !ls /kaggle/working/checkpoints/*
   ```
   Phải thấy thư mục `sdft_science_Qwen2.5-7B-Instruct_n<N_SCI>_ema0.01_forward_s42` và bên trong có `epoch1`, `epoch2`.

### 6.3 Chạy đánh giá

Thêm **ô 4**, dùng đúng số `N_SCI` của đêm qua:

```python
# Nếu 6.1 báo train CHƯA xong, bỏ dấu # ở dòng dưới để train tiếp trước:
# !N_SCI=1376 bash scripts/kaggle_jobs.sh e1_sdft
!N_SCI=1376 bash scripts/kaggle_jobs.sh eval_sdft e1
```

Bấm **Save Version → Save & Run All**, rồi tắt phiên tương tác. Thời gian chạy:
- đánh giá trên tập val để chọn epoch: ~20 phút;
- tập test 507 câu: ~30 phút;
- 6 benchmark khả năng cũ (lm-eval, 25% số câu): ~1–2 giờ.

Trong log sẽ có dòng `best checkpoint: .../epochX (val acc ...)`.

## Bước 7. Gửi kết quả cho nhóm

Khi version đánh giá chạy xong, gửi vào nhóm:

1. **Link W&B** của các run:
   - run train;
   - `val-...-epoch1`, `val-...-epoch2`;
   - `test-...`;
   - `prior-...`.
2. Từ tab **Output** của notebook đánh giá, tải về:
   - `eval_results/test-*.json`: có `acc` (độ chính xác task mới), `paper_cmp` (so với 70,2 của paper) và từng câu trả lời;
   - `eval_results/prior/`: kết quả 6 benchmark.
3. Thông tin để ghi vào phần deviation của slide: số `N_SCI` đã dùng, epoch được chọn, tổng giờ chạy (`time/elapsed_h` trên W&B), và mọi sự cố đã gặp.

---

## Gặp lỗi

| Triệu chứng | Cách xử lý |
|---|---|
| `CUDA out of memory` khi pilot hoặc train | Thêm `GEN_BS=4` vào đầu lệnh, ví dụ `!GEN_BS=4 bash scripts/kaggle_jobs.sh pilot` (và giữ `GEN_BS=4` cho `e1_sdft`). Vẫn OOM thì dùng `GEN_BS=2` và báo nhóm |
| `kl=nan` | fp16 trên T4 bị tràn số. **Dừng lại**, gửi log cho nhóm, không chạy qua đêm |
| Ô 1 báo key W&B sai | Kiểm tra secret đã tick "Attach" chưa, tên phải đúng `WANDB_API_KEY` |
| Ô 2 không có GPU / `CUDA not available` | Accelerator chưa chọn `GPU T4 x2`, hoặc tài khoản chưa xác minh số điện thoại |
| Lỗi tải model / `Connection error` | Internet chưa bật. Bật lên rồi chạy lại ô 2 |
| Run chết giữa chừng (Kaggle lỗi, hết quota) | Checkpoint được lưu mỗi 10 bước. Làm như bước 6.2, rồi chạy lại **đúng** lệnh `e1_sdft` cũ, nó sẽ train tiếp |
| Quota sắp hết | Báo nhóm ngay để chuyển job sang tài khoản khác |

## Quy tắc

- **Không đổi `N_SCI`** giữa train và eval, và không đổi sau khi đã báo cho B và C. SFT và SDFT phải dùng cùng một tập con.
- **Không sửa code trực tiếp trên Kaggle.** Cần sửa thì sửa trên GitHub để mọi tài khoản chạy cùng một phiên bản (ô 2 in commit đang dùng).
- Seed giữ mặc định `42`.
