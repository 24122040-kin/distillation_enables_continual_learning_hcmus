# Báo cáo kiểm tra log lỗi & hướng dẫn xử lý (20/11/2026)

> Tổng hợp đầy đủ **Mục A (W&B)** theo checklist trong [`docs/thu_thap_log_loi.md`](thu_thap_log_loi.md), chẩn đoán nguyên nhân gốc và hướng dẫn các bước tiếp theo cho **Người 2 / `htti`**.
> Phần **Mục B** (file trong tab Output của `sdft-B-night`) thuộc máy Kaggle của Người 2 nên cần Người 2 kiểm tra và điền nốt **sau khi `sdft-B-night` chạy xong** (không bấm Stop/Cancel giữa chừng).

---

## 1. Mẫu tin nhắn tóm tắt (theo khuôn mẫu `thu_thap_log_loi.md`)

```text
[sdft_tool_use 7B]  State: Running trên W&B (process GPU0 đã crash CUDA OOM lúc 02:36:42; GPU1 giữ container sống)
                    Step cuối: _step=5 (log in [ep 1 step 6/33] lúc 02:12:39, crash ở step 7/33 lúc 02:36:42) / 66
                    kl_loss cuối: 0.2450 (Summary step 5) / 0.1555 (log step 6)
Log cuối (A1.2):
(đủ 21 dòng ở Mục 2.1 bên dưới — đích xác torch.OutOfMemoryError ở src/sdft_trainer.py:77 compute_analytic_kl_loss)
RAM cuối (A1.5): 92.63% (~2602 MB non-swap)   GPU mem cuối (A1.6): GPU0 = 95.71% (trước OOM, sau về 0%); GPU1 = 76.45% (lúc GPU0 crash)
[GPU1] sft_science: Finished (22/22 step, 01:02:11 -> ~02:22:47, 1h 20m 36s, loss=1.0942)
       sft_tool_use: Finished (33/33 step, 02:23:30 -> ~04:43:01, 2h 19m 31s, loss=0.1632)
       3B: test-base Finished (2h 58m 39s, run n42ssnh2)
           teacher Finished (07:43:07 -> ~08:29:27, 46m 20s, acc=89.55%)
           sft_science Finished (08:29:53 -> ~09:05:13, 22/22 step, 35m 20s, loss=1.1673)
           test-sft Running (bắt đầu 09:05:34, đã đạt 25/32 batches ~78% lúc kiểm tra)
[Kin] sdft_science (9af4d088): v2 crash CUDA OOM lúc 10:07:32 ở step 30/44 (kl=0.0856; đã lưu epoch1/ + resume/step 30).
      Đã push fix chunked KL (88ba744) và resume qua sdft-A-e1-eval (đã nạp global_step=30 lúc ~11:06;
      lúc ~13:00 W&B báo 9h48m -> Crashed trong lúc W&B có banner trễ ingestion, cần check lại Kaggle sau khi login lại).
Quota còn: Kin = 8h 33m (trước khi chạy sdft-A-e1-eval) | Người 2 = <điền sau khi sdft-B-night xong>
```

---

## 2. Chi tiết thu thập Mục A (trên W&B `sdft-continual-learning`)

### 2.1. A1 — Run bị lỗi: `sdft_tool_use_Qwen2.5-7B-Instruct_n528_ema0.01_forward_s42` (`db63e600`)

| Mục | Giá trị quan sát trên W&B |
|---|---|
| Run ID | `db63e600` |
| State (trên W&B) | `Running` (process Python trên GPU0 đã dừng ở `02:36:42`, nhưng phiên `night2` vẫn chạy GPU1 nên run chưa chuyển trạng thái) |
| Bắt đầu / Runtime | `2026-11-20 01:02:12` (giờ VN) · Runtime hiển thị `10h 04m 26s` → `10h 16m 11s` |
| Config | `data_source_resolved = hf:Baokin/sdft-official-data`, `n_train = 528`, `total_steps = 66`, `max_hours = 10.5` |
| Summary cuối (`_step = 5`) | `epoch = 0.1515`, `train/kl_loss = 0.24501`, `train/grad_norm = 1.10706`, `train/resp_tokens_mean = 127`, `train/truncated_frac = 0`, `train/skipped = 0`, `time/step_sec = 897.58s` (`gen=505.59s`, `train=391.92s`), `time/elapsed_h = 0.8663h` |
| System RAM (A1.5) | Peak & cuối trước khi chết (phút `94.51`, ~`02:36:42`): **`92.63%`**; `Process Memory In Use (non-swap)`: **`~2602 MB`** |
| GPU Memory (A1.6) | **GPU 0:** `90.21%` (0.58h) → đạt đỉnh **`95.71%`** (1.56h, ngay trước OOM) → tụt về **`0%`** từ `1.88h` tới `10.12h`. **GPU 1:** chạy bình thường suốt phiên (`76.53%` → `82.09%` → `66.62%`). |

**Toàn bộ 21 dòng trong tab Logs của `db63e600` (copy nguyên văn):**

```text
 1 2026-11-20 01:04:00 `use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.
 2 2026-11-20 01:04:00 /usr/local/lib/python3.13/dist-packages/torch/_dynamo/eval_frame.py:1263: UserWarning: torch.utils.checkpoint: the use_reentrant parameter should be passed explicitly. Starting in PyTorch 2.9, calling checkpoint without use_reentrant will raise an exception. use_reentrant=False is recommended, but if you need to preserve the current default behavior, you can pass use_reentrant=True. Refer to docs for more details on the differences between the two variants.
 3 2026-11-20 01:04:00   return fn(*args, **kwargs)
 4 2026-11-20 01:11:58 [ep 1 step 1/33] kl=0.2554 len=68 gen=230s train=355s
 5 2026-11-20 01:21:29 [ep 1 step 2/33] kl=0.4372 len=60 gen=201s train=370s
 6 2026-11-20 01:30:17 [ep 1 step 3/33] kl=0.3322 len=60 gen=194s train=334s
 7 2026-11-20 01:39:14 [ep 1 step 4/33] kl=0.3154 len=71 gen=216s train=321s
 8 2026-11-20 01:54:12 [ep 1 step 5/33] kl=0.2450 len=127 gen=506s train=392s
 9 2026-11-20 02:12:39 [ep 1 step 6/33] kl=0.1555 len=222 gen=666s train=441s
10 2026-11-20 02:36:42 Traceback (most recent call last):
11 2026-11-20 02:36:42   File "/kaggle/working/distillation_enables_continual_learning_hcmus/src/sdft_trainer.py", line 430, in <module>
12 2026-11-20 02:36:42     train_sdft(build_parser().parse_args())
13 2026-11-20 02:36:42     ~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
14 2026-11-20 02:36:42   File "/kaggle/working/distillation_enables_continual_learning_hcmus/src/sdft_trainer.py", line 342, in train_sdft
15 2026-11-20 02:36:42     loss = sdft_example_loss(model, tokenizer, ex, prompt, ro["ids"], student_params, args)
16 2026-11-20 02:36:42   File "/kaggle/working/distillation_enables_continual_learning_hcmus/src/sdft_trainer.py", line 183, in sdft_example_loss
17 2026-11-20 02:36:42     return compute_analytic_kl_loss(s_logits, t_logits, mask, args.kl_mode)
18 2026-11-20 02:36:42   File "/kaggle/working/distillation_enables_continual_learning_hcmus/src/sdft_trainer.py", line 77, in compute_analytic_kl_loss
19 2026-11-20 02:36:42     token_kl = (t_logp.exp() * (t_logp - s_logp)).sum(-1)
20 2026-11-20 02:36:42                 ~~~~~~~~~~~~~^~~~~~~~~~~~~~~~~~~
21 2026-11-20 02:36:42 torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 466.00 MiB. GPU 0 has a total capacity of 14.56 GiB of which 308.81 MiB is free. Including non-PyTorch memory, this process has 14.26 GiB memory in use. Of the allocated memory 13.40 GiB is allocated by PyTorch, and 741.97 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf)
```

---

### 2.2. A2 — Các run trên GPU1 trong `sdft-B-night`

| Run | ID | State | Bắt đầu (giờ VN) | Runtime / Kết thúc | Step / Metric | Peak System RAM |
|---|---|---|---|---|---|---|
| `sft_science_Qwen2.5-7B-Instruct_n352_s42` | `dsk0618v` | `Finished` | `01:02:11` | `1h 20m 36s` (~`02:22:47`) | `22/22` (`sft_loss = 1.0942`) | `91.94%` |
| `sft_tool_use_Qwen2.5-7B-Instruct_n528_s42` | `f41929fz` | `Finished` | **`02:23:30`** | `2h 19m 31s` (~`04:43:01`) | `33/33` (`sft_loss = 0.1632`) | `~92.6%` (13 phút đầu tới `02:36:42`), sau giảm còn `~87.0%` |
| `test-base-Qwen2.5-3B-Instruct-science` | `n42ssnh2` | `Finished` | *(chưa lấy được giờ chi tiết do W&B ngắt kết nối)* | `2h 58m 39s` | *(chưa đọc Summary)* | *(chưa đọc)* |
| `teacher-Qwen2.5-3B-Instruct-science` | `b1wumidt` | `Finished` | `07:43:07` | `46m 20s` (~`08:29:27`) | `eval/acc = 89.55%` (`134` mẫu, `trunc=0%`) | `95.55%` |
| `sft_science_Qwen2.5-3B-Instruct_n352_s42` | `xjf9h0qa` | `Finished` | `08:29:53` | `35m 20s` (~`09:05:13`) | `22/22` (`sft_loss = 1.1673`) | `95.54%` |
| `test-sft_science_Qwen2.5-3B-Instruct_n352_s42-epoch1` | `mefkrrft` | `Running` | `09:05:34` | `2h 13m 16s` (đang chạy) | Logs: `25/32` (`78%`) | `95.52%` |

*(Ghi chú: Toàn bộ chuỗi GPU1 chạy khỏe mạnh, không chết do RAM hay GPU.)*

---

### 2.3. A3 — Run của Kin: `sdft_science_Qwen2.5-7B-Instruct_n352_ema0.01_forward_s42` (`9af4d088`)

- **Phiên 1 (`sdft-A-e1-sdft` v2):** Bắt đầu `01:17:59`, hoàn thành toàn bộ Epoch 1 (`22/22`, `kl = 0.0802`), tới **Epoch 2 bước `8/22` (`global_step = 30/44`, `kl = 0.0856`, `elapsed_h = 8.505h`)** thì dừng lúc **`10:07:32`** (tổng phiên `8h 53m 26s` trên Kaggle) vì **cùng một lỗi `torch.OutOfMemoryError` tại `src/sdft_trainer.py:77` (`compute_analytic_kl_loss`)**:
  - `Tried to allocate 470.00 MiB. GPU 0 has a total capacity of 14.56 GiB of which 464.81 MiB is free... 12.49 GiB allocated, 1.47 GiB reserved`.
  - System tab trước khi OOM: GPU0 = **`94.39%`**, System RAM = `85.89%` (peak `90.81%`), Process mem = `2492.5 MB` (peak `2973.7 MB`).
- **Xử lý & Resume:** Đã đóng gói Output (`epoch1/` + `resume/ckpt.pt` ở `global_step = 30`) thành dataset private `sdft-a-e1-ckpt-step30`, push fix chunked KL (commit **`88ba744`**), và chạy nối qua `sdft-A-e1-eval` (log Kaggle xác nhận đã nạp `global_step = 30` và in `wandb: Resuming run ... 9af4d088` lúc ~`11:06`). *(Tại thời điểm ~`13:00` W&B có banner bảo trì/trễ ingestion và hiện `9h 48m` → `Crashed` với log vẫn dừng ở dòng 45 cũ; cần kiểm tra lại tab Logs trên Kaggle sau khi đăng nhập lại.)*

---

## 3. Chẩn đoán nguyên nhân gốc (Root Cause)

1. **KHÔNG phải hết RAM hệ thống hay đụng nhau giữa GPU0 và GPU1:**
   - Khi GPU0 chết lúc `02:36:42`, System RAM mới ở mức `92.63%` (`~2.6 GB` process non-swap), không phải Linux OOM Killer (`Killed`).
   - Run `sft_tool_use` 7B (`f41929fz`) trên GPU1 đã khởi động và load xong model từ `02:23:30` (chạy song song êm 13 phút trước lúc `02:36:42`), và sau khi GPU0 giải phóng RAM thì GPU1 còn tụt xuống `87.0%`.
2. **Đích xác là CÙNG MỘT BUG CUDA OOM VRAM trên GPU 0 tại `src/sdft_trainer.py:77` (`compute_analytic_kl_loss`):**
   - `compute_analytic_kl_loss` ép kiểu toàn bộ tensor logits `(T, 152064)` của câu trả lời sang `float32` một lượt (`log_softmax` cho student & teacher, `exp()`, hiệu, tích + đồ thị autograd), tiêu tốn tới **~4–5 GB VRAM đỉnh** khi một rollout tình cờ sinh dài (gần `max_new_tokens=1024`).
   - Ở `sdft_tool_use` (`db63e600`): 6 bước đầu các câu trả lời ngắn (`len = 60–222`) nên lọt vừa 14.56 GiB VRAM của T4; sang **step 7/33** (lúc `02:36:42`), một mẫu sinh dài vượt ngưỡng khiến `GPU 0` chạm `95.71%` và văng `torch.OutOfMemoryError` (`Tried to allocate 466.00 MiB`, chỉ còn `308.81 MiB` free).
   - Ở `sdft_science` của Kin (`9af4d088`): cũng bị đúng dòng 77 ở bước `31/44` (thiếu `5.19 MiB`).
3. **Về checkpoint resume của `sdft_tool_use`:** Vì `--save_every` mặc định là `10` bước mà `db63e600` dừng ở **bước 7/33**, nhiều khả năng **chưa có thư mục `resume/`** (chỉ mất ~1,5h chạy của phần GPU0, toàn bộ SFT 7B và 3B trên GPU1 vẫn còn nguyên).

---

## 4. Hướng dẫn bước tiếp theo cho bạn làm (`htti` / Người 2)

### Bước 1: Code đã được sửa sẵn trên nhánh `main` (commit `88ba744`)
- Commit **`88ba744`** (`fix(sdft): chunk analytic KL loss with checkpointing to prevent T4 OOM`) đã push lên `main` trong [`src/sdft_trainer.py`](../src/sdft_trainer.py):
  - Tách `compute_analytic_kl_loss` thành từng đoạn **`chunk_size=128` token** theo trục thời gian có `torch.utils.checkpoint.checkpoint(..., use_reentrant=False)` — **giữ nguyên 100% giá trị toán học và tương thích hoàn toàn với mọi checkpoint**, đồng thời giảm đỉnh VRAM phần KL từ ~4–5 GB xuống còn ~350 MB.
  - Đổi `t_logits` ở `sdft_example_loss` thành `.detach().clone()` để giải phóng ngay tensor logits của prompt teacher trước khi student forward.
- **Khuyến nghị thêm ở Ô 1** của mọi notebook SDFT sau này:
  ```python
  os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
  ```

### Bước 2: Với `sdft-B-night` đang chạy của Người 2
1. **KHÔNG bấm Stop / Cancel** nếu `sdft-B-night` vẫn đang chạy phần GPU1 (`test-sft 3B` / `sdft 3B`). Đợi phiên kết thúc tự nhiên để Kaggle giữ Output đầy đủ (`checkpoints/` + `eval_results/`).
2. Khi `sdft-B-night` xong, thực hiện **Mục B** trong [`thu_thap_log_loi.md`](thu_thap_log_loi.md):
   - Kiểm tra `eval_results/gpu0_e2_sdft.log`, `eval_results/gpu1_chain.log`, `eval_results/FAILED.txt`, và các file `*.json` (kết quả 3B).
   - Kiểm tra cây `checkpoints/`: xem `sft_science_.../epoch1/`, `sft_tool_use_.../epoch1/`, và `3B` đã đủ chưa; kiểm tra `sdft_tool_use_.../resume/state.json` (dự kiến chưa có vì chết trước step 10).
   - Ghi lại quota GPU còn lại của tài khoản Người 2.

### Bước 3: Chạy bù `e2_sdft` (Tool Use 7B) và `day2_b`
- **Không chạy lại `night2`** (vì GPU1 đã chạy xong SFT 7B Science, SFT 7B Tool Use và phần 3B rồi).
- Tạo một notebook mới chạy **riêng `e2_sdft`** (1 GPU, đã lấy commit mới nhất `>= 88ba744` ở Ô 2 và thêm `PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"` ở Ô 1):
  ```python
  !MAX_HOURS=10.5 N_TOOL=528 bash scripts/kaggle_jobs.sh e2_sdft
  ```
  *(Lưu ý: với `N_TOOL=528`, 66 bước × ~14–16 phút/bước ≈ ~15–17 tiếng. Sau phiên 10,5h này sẽ lưu được `epoch1/` + `resume/` ở khoảng ~40–45/66 bước; sau đó nối tiếp ~4–5 tiếng train nốt + eval nếu quota còn đủ, hoặc xem lại thời gian từng bước sau khi có bản vá `88ba744`.)*
- **Mẹo lấy Output khi notebook cũ có job con bị lỗi:** nếu gắn *Add Input → Your Work → Notebooks → `sdft-B-night`* không hiện đủ `checkpoints/`, hãy vào trang **Output** của version đó trên Kaggle → bấm **`+ New Dataset`** để đóng gói Output thành dataset private, rồi gắn dataset đó làm Input cho notebook tiếp theo (`sdft-B-day2`).
