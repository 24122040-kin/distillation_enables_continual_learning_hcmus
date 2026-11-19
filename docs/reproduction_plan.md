# Kế hoạch tái hiện thí nghiệm — SDFT (Shenfeld et al., 2026)

> Tài liệu làm việc nội bộ của nhóm. Viết ngày 2026-11-19.
> Nguồn: paper `arXiv:2601.19897v2`, các trang yêu cầu môn học (`index.md`, `reading.md`, `reproduction.md`, `analysis.md`, `empirics.md`), code hiện có trong repo, và repo chính thức `github.com/idanshen/Self-Distillation` (chỉ đọc để đối chiếu — **không copy code**).

---

## 0. Tóm tắt nhanh

| Câu hỏi | Trả lời ngắn |
|---|---|
| Code hiện có dùng được không? | **Giữ được phần lõi thuật toán** (rollout on-policy → logits của teacher → KL giải tích theo từng token → cập nhật EMA). **Phải viết lại** phần dữ liệu và đánh giá: hiện tại cả hai task đều đang huấn luyện/đánh giá **sai bài toán**, nên mọi con số chạy ra từ code này đều không so được với paper (chi tiết ở §1). |
| Có cần làm lại từ đầu không? | Không cần làm lại toàn bộ. Khoảng 30% code (loss, EMA, khung train) giữ lại; `data_loader.py` và `evaluate.py` viết lại; thêm sinh theo batch (hoặc vLLM) và script lm-eval. |
| Dùng 3B hay 7B? | **Cả hai, như một thí nghiệm có chủ đích.** Hình 5 (trái) của paper báo SDFT − SFT = **−3.3** điểm ở 3B, **+4.0** ở 7B, **+6.9** ở 14B. Chạy 3B và 7B trên Science Q&A chính là tái hiện thí nghiệm "Effect of model size" và trả lời luôn thắc mắc của bạn. Kết quả chính (Table 5) bắt buộc phải là 7B. |
| 7B mất bao lâu? | Một lượt SDFT 7B trên Science Q&A (2.674 mẫu, 2 epoch): **~19 giờ trên Kaggle T4** (QLoRA 4-bit, sinh theo batch); **~70+ giờ** nếu giữ sinh từng mẫu như code hiện tại; **~1,5–3 giờ trên A100 80GB thuê**. Cả core scope (2 task, 1 seed): **~55–60 GPU-giờ T4** hoặc **~12 GPU-giờ A100** (≈ 15–25 USD). Chi tiết ở §4. |

---

## 1. Đánh giá code hiện có (commit `8b32e58`)

### 1.1 Phần làm tốt — giữ lại

- `src/sdft_trainer.py:243-282` — đúng cấu trúc Algorithm 1: student tự sinh `y ~ π_θ(·|x)` → teacher chấm điểm **trên cùng quỹ đạo** với ngữ cảnh có demonstration → KL giải tích trên toàn vocab (App. A.1) → cập nhật EMA teacher (dòng 19 của Algorithm 1).
- `compute_analytic_kl_loss` hỗ trợ cả forward và reverse KL → dùng được ngay cho phần phân tích "lý thuyết reverse KL vs thực tế forward KL".
- Có mask vài token đầu (Mục 5 "Learned Artifacts"), seed cố định, docstring, log W&B, README có sẵn bảng mapping.

### 1.2 Lỗi phải sửa (xếp theo mức độ)

| # | Mức | Vị trí | Vấn đề | Hậu quả |
|---|---|---|---|---|
| 1 | **Nghiêm trọng** | `data_loader.py:44-46` | Trên HF, `level` nằm trong `row["details"]["level"]`, không ở top-level → bộ lọc Chem L-3 **luôn rỗng** → rơi xuống nhánh fallback (dòng 72-85) lấy **mọi** câu Chemistry (L1–L5), prompt **không có đáp án A/B/C/D**, demonstration chỉ là `"Final Answer: X"`. | Sai tập dữ liệu, sai prompt, demonstration không có lời giải → teacher không có gì để "học trong ngữ cảnh". Kết quả vô nghĩa so với paper. |
| 2 | **Nghiêm trọng** | `data_loader.py:122-152` | ToolAlpaca: lấy `output` (câu trả lời ngôn ngữ tự nhiên cuối cùng) làm demonstration và nhãn, thay vì lời gọi API (`Action` + `Action Input`). Tập test cắt từ `train_data.json`, không theo split gốc như paper. | Huấn luyện sai mục tiêu; tập test bị rò rỉ cùng phân phối API với train. |
| 3 | **Nghiêm trọng** | `evaluate.py:41` | Regex `(?:Final Answer:\s*|Answer:\s*|\b)([A-D])\b` bắt **chữ "A" đầu tiên đứng riêng** (ví dụ mạo từ "A molecule…"). | Độ chính xác Science bị lệch có hệ thống về đáp án A. |
| 4 | **Nghiêm trọng** | `evaluate.py:128-132` | Tool Use chấm bằng "chuỗi đáp án nằm trong output" sau khi bỏ khoảng trắng; paper chấm theo **tên API + tham số** (không phụ thuộc thứ tự tham số). | Độ chính xác Tool Use gần như 0 hoặc không so được. |
| 5 | Cao | `sdft_trainer.py:323`, `evaluate.py:51` | `max_new_tokens=384`; paper ghi 2048, code chính thức dùng 1024. Demonstration Science dài trung bình ~380 token, p90 ~480 token. | Lời giải CoT bị cắt giữa chừng → mất `<answer>` → sai cả khi lập luận đúng. |
| 6 | Cao | `sdft_trainer.py:174`, `sft_baseline.py:62` | Tính toán ở `float16`; Qwen2.5 được huấn luyện bf16, fp16 dễ tràn số. (T4 không có bf16 gốc — lý do nên dùng A100/L4 cho bản cuối.) | Có thể NaN/loss nổ, nhất là với logits 152k vocab. |
| 7 | Cao | `sdft_trainer.py:247` | Sinh **từng mẫu một** (batch = 1) bằng HF `generate`. | Chiếm ~80% thời gian; chậm hơn ~4× so với sinh theo batch (xem §4). |
| 8 | Trung bình | `sdft_trainer.py:110-131` | EMA áp lên **từng ma trận LoRA A và B riêng rẽ**. Vì ΔW = BA là tích, EMA(B)·EMA(A) ≠ EMA(BA). | Teacher không đúng là "EMA của trọng số student" như paper. Không sửa được rẻ — **ghi vào bảng deviation**. |
| 9 | Trung bình | toàn bộ | Batch hiệu dụng 8 (paper sweep 16/32/64, code chính thức 32); không có tập validation để chọn checkpoint; không có vòng lặp 3 seed; README chạy lm-eval với `--limit 200` và **thiếu IFEval, HumanEval**; kết quả lm-eval không log lên W&B. | Không đáp ứng giao thức của paper và yêu cầu W&B của môn. |
| 10 | Thấp | `sdft_trainer.py:239` | Cắt prompt teacher từ bên phải ở 1536 token có thể cắt mất dòng "Now answer with a response of your own…". | Hiếm gặp với Science; cần kiểm tra với Tool Use (prompt ~750 token). |
| 11 | Thấp | `sdft_trainer.py:293` | `step_sec` chỉ đo micro-batch cuối, không phải cả bước tối ưu. | Số đo thời gian trên W&B sai → không dùng được cho slide về chi phí. |

### 1.3 Quyết định

Giữ khung, viết lại theo thứ tự:

1. `data_loader.py` → dùng **dữ liệu chính thức của tác giả** (đã có demonstration GPT-4o cho Science, split gốc cho Tool Use), mirror lên **Hugging Face dataset của nhóm** (môn học yêu cầu dữ liệu tải từ HF/Kaggle) và ghi nguồn trong README.
2. `evaluate.py` → parse `<answer>…</answer>` cho Science; so khớp tên action + JSON tham số cho Tool Use; sinh theo batch.
3. `sdft_trainer.py` → sinh rollout theo batch, `max_new_tokens=1024`, bf16 khi phần cứng cho phép, batch hiệu dụng 32, lưu checkpoint/resume (Kaggle giới hạn phiên 12 giờ).
4. Thêm `src/eval_prior.py` gọi lm-eval đủ 6 benchmark và log lên W&B.
5. Thêm `configs/*.yaml` + script chạy theo seed để mọi run tái lập được.

---

## 2. Phạm vi đề xuất: **Core scope**

Lý do (ghi trong deck): paper chạy full fine-tuning 7B trên H200 với sweep 54 cấu hình mỗi phương pháp; ngân sách nhóm không cho phép full scope. Medical cần GPT-5-mini làm giám khảo (tốn API, không tái lập được), Knowledge Acquisition chưa có dữ liệu công khai (tác giả hứa "upload soon").

| Thí nghiệm | Tương ứng paper | Mô hình | Phương pháp | Bắt buộc? |
|---|---|---|---|---|
| **E1 — Science Q&A (headline)** | Table 5 (khối 1), Fig. 4 trái | Qwen2.5-7B-Instruct | Base, SFT, SDFT | ✅ |
| **E2 — Tool Use** | Table 5 (khối 2), Fig. 4 giữa, §3.2 | Qwen2.5-7B-Instruct | Base, SFT, SDFT | ✅ |
| **E3 — Phân tích bổ sung: kích thước mô hình** | Fig. 5 trái | Qwen2.5-1.5B / 3B / 7B | SFT, SDFT trên Science | ✅ (thay cho phần bị bỏ) |
| E4 — (tùy chọn) Sweep EMA α ∈ {0 (teacher cố định), 0.01, 0.05, 1 (teacher = student)} | Fig. 8 / App. A.3 | 1.5B hoặc 3B, Tool Use | SDFT | Nếu dư ngân sách |

E3 dùng lại run 7B của E1, chỉ thêm 1.5B và 3B (rẻ hơn nhiều). Nó tái hiện đúng câu mà bạn hỏi: *ở 3B, SDFT có thua SFT không?* — và là cầu nối sang poster (giả định ICL ở §3.2 của paper).

**Ý tưởng cho Lab 2 / poster** (để chọn sớm, không chạy bây giờ):
- *Khoảng cách lý thuyết–thực hành*: phép chứng minh "SDFT ≡ IRL" (§3.1) chỉ đúng với **reverse KL**, nhưng README chính thức ghi rằng *mọi kết quả trong paper* dùng **forward KL theo từng token**. → So reverse vs forward bằng code sẵn có (`--kl_mode`).
- *Giả định ICL bị vi phạm*: demonstration nhiễu / sai / bị xáo trộn → đo teacher accuracy và độ quên khi tỉ lệ nhiễu tăng (đường suy giảm).
- *Chi phí*: paper nói 2,5× FLOPs, 4× wall-clock so với SFT; trên phần cứng của nhóm tỉ lệ thực tế là bao nhiêu (§4 dự báo ~11× trên T4).

---

## 3. Thiết lập thí nghiệm

### 3.1 Dữ liệu (đã kiểm tra trực tiếp từ repo chính thức)

| Task | Train | Test | Độ dài prompt | Độ dài demonstration | Cách chấm |
|---|---|---|---|---|---|
| Science (SciKnowEval Chem L-3) | 2.674 | 507 | ~680 ký tự (~200 token, có system prompt yêu cầu `<reasoning>`/`<answer>`) | ~1.380 ký tự (~380 token), p90 ~480 token | khớp chính xác chữ cái trong `<answer>` |
| Tool Use (ToolAlpaca) | 4.046 | 97 | ~2.600 ký tự (~750 token, gồm tài liệu API) | ~330 ký tự (~95 token) | cùng tập tên action + cùng JSON tham số |

Validation: tách 5% từ train (seed cố định) để chọn checkpoint như paper.

### 3.2 Siêu tham số: paper ↔ code chính thức ↔ nhóm

| Tham số | Paper (Table 3) | Code chính thức | Nhóm (đề xuất) | Ghi chú deviation |
|---|---|---|---|---|
| Fine-tuning | Full FT | Full FT | **LoRA r=32 all-linear** (QLoRA 4-bit nếu T4) | Phần cứng. LoRA quên ít hơn → khoảng cách SFT–SDFT về forgetting có thể **nhỏ lại**. |
| Learning rate | sweep {5e-6, 1e-5, 5e-5} | mặc định 2e-5, README dùng 5e-5 | 1e-4 (LoRA cần LR cao hơn full FT) | Ghi rõ; sweep nhỏ {5e-5, 1e-4, 2e-4} trên 1.5B. |
| Batch | {16, 32, 64} | 1 × grad-accum 32 | 32 (micro-batch 4–8 × accum) | |
| Epochs | SFT 1, SDFT 2 | 2 | SFT 1, SDFT 2 | |
| EMA α | {0.01, 0.02, 0.05} | 0.01, cập nhật mỗi bước | 0.01 | |
| Loss | lý thuyết reverse KL, thực tế forward KL | forward KL theo token | forward KL (chính), reverse (phân tích) | |
| Bỏ token đầu | "vài token" | 3 | 3 | |
| Max generation | 2048 | 1024 | 1024 | Paper và code chính thức không khớp → ghi vào slide so sánh code. |
| Warmup | 10 bước | 10% số bước | 10 bước | |
| Sinh rollout | không nêu | vLLM + importance-sampling correction | HF batch (T4) / vLLM (A100) | |
| Precision | bf16 | bf16 | bf16 (A100/L4), fp16 (T4) | |
| Seeds | 3 | — | 1 seed cho tất cả; 3 seed cho E1 nếu kịp | |

### 3.3 Đánh giá

- **Task mới**: greedy (temperature 0), `max_new_tokens=1024`.
- **Khả năng cũ** (lm-eval-harness, đủ 6 bộ): HellaSwag, TruthfulQA (mc2), MMLU, IFEval, Winogrande, HumanEval. Trên T4: dùng `--limit` theo tỉ lệ và **ghi rõ** trong deck; trên A100: chạy đầy đủ. Base model chỉ đánh giá một lần, dùng chung.
- **Bảng so sánh**: số của paper đặt cạnh số của nhóm, kèm sai lệch tuyệt đối và tương đối; mỗi số có link run W&B.

---

## 4. Ước tính thời gian chạy 7B

### 4.1 Cách ước tính (để nhóm tự hiệu chỉnh)

Với Qwen2.5-7B (N ≈ 7,6 tỷ tham số), một token qua forward tốn ≈ 2N FLOPs; forward + backward có gradient checkpointing ≈ 4 × forward.

**Một mẫu Science trong SDFT** (prompt ≈ 200, demo ≈ 380, response ≈ 450 token):
- Sinh rollout 450 token — bị giới hạn bởi **băng thông bộ nhớ**, không phải FLOPs → phần đắt nhất.
- Teacher forward trên ~1.070 token: 2N × 1.070 ≈ 1,6·10¹³ FLOPs.
- Student forward + backward trên ~650 token: 4 × 2N × 650 ≈ 4,0·10¹³ FLOPs.

**Một mẫu SFT**: 4 × 2N × 580 ≈ 3,5·10¹³ FLOPs, không có bước sinh.

Giả định thông lượng (ước lượng thận trọng, **cần đo lại bằng pilot**):

| | T4 16GB, QLoRA 4-bit | A100 80GB, LoRA bf16 |
|---|---|---|
| Train, hiệu dụng | ~15 TFLOPs (bnb dequant tốn kém) | ~100 TFLOPs |
| Sinh, HF batch = 1 | ~10 token/s | ~35 token/s |
| Sinh, HF batch = 8–16 | ~50 token/s tổng | ~350 token/s tổng |
| Sinh, vLLM | — (T4 khó chạy vLLM với LoRA + 4-bit) | ~2.000 token/s tổng |

### 4.2 Kết quả ước tính — mỗi run, 1 seed, đủ dữ liệu train

| Hạng mục | Kaggle T4 · code hiện tại (sinh bs=1) | Kaggle T4 · sinh theo batch | A100 80GB · HF batch | A100 80GB · vLLM |
|---|---|---|---|---|
| SFT Science, 1 epoch | ~1,7 h | ~1,7 h | ~0,3 h | ~0,3 h |
| **SDFT Science, 2 epoch** | **~70–75 h** | **~19 h** | ~2,8 h | **~1,3–2 h** |
| SFT Tool Use, 1 epoch | ~3,8 h | ~3,8 h | ~0,6 h | ~0,6 h |
| **SDFT Tool Use, 2 epoch** | ~35 h | **~16 h** | ~2,5 h | ~2 h |
| Eval task mới (Science, 507 câu) | ~1,5 h | ~0,5 h | ~10 phút | ~2 phút |
| lm-eval 6 bộ, **đầy đủ**, mỗi model | ~4–5 h | ~4–5 h | ~0,75–1 h | ~0,3–0,5 h |
| lm-eval 6 bộ, **lấy mẫu ~20%** | ~1 h | ~1 h | — | — |

So sánh với paper: tác giả báo SDFT tốn **~4× wall-clock** so với SFT (H200 + vLLM). Trên T4 tỉ lệ này thành **~11×** vì bước sinh chậm — đây là một con số đáng đưa vào slide chi phí.

### 4.3 Tổng ngân sách cho core scope

| Kịch bản | E1 + E2 (7B, 1 seed) | E3 (thêm 1.5B + 3B) | Tổng (+20% debug) | Thời gian thực |
|---|---|---|---|---|
| **Kaggle 2×T4 miễn phí**, đủ dữ liệu, lm-eval lấy mẫu | ~47 h | ~15 h | **~75 GPU-h** | Kaggle cho ~30 GPU-h/tuần/tài khoản, phiên ≤ 12 h → **chia cho 3–4 tài khoản trong nhóm ≈ 1 tuần**, SDFT phải resume qua 2 phiên |
| Kaggle T4, **cắt train còn 1.000 mẫu/task** | ~22 h | ~7 h | **~35 GPU-h** | ~1 tuần với 2 tài khoản; ghi rõ là deviation |
| **A100 80GB thuê** (RunPod/Vast/Lambda…), đủ dữ liệu, lm-eval đầy đủ | ~9 h | ~3 h | **~15 GPU-h** | 1–2 ngày; ~1,2–2 USD/giờ (**giá tham khảo, cần kiểm tra**) → **~20–30 USD**; 3 seed cho E1+E2 → ~40–60 USD |
| H100 thuê | ~5 h | ~2 h | ~9 GPU-h | ~1 ngày, giá/giờ cao hơn nhưng tổng tương đương A100 |

3B tốn khoảng **40–45%** thời gian của 7B; 1.5B khoảng **20–25%**.

### 4.4 Khuyến nghị

1. **Phát triển và debug bằng Qwen2.5-0.5B / 1.5B trên Kaggle** (mỗi vòng lặp vài phút đến < 1 giờ).
2. **Chạy pilot 7B 20 bước** trên đúng phần cứng sẽ dùng, đọc `step_sec` trên W&B, rồi nhân lên để thay các con số ở §4.2 bằng số đo thật.
3. **Run cuối 7B**: nếu nhóm chi được ~30–60 USD thì thuê A100/H100 1–2 ngày — sạch hơn (bf16, không lượng tử hóa, lm-eval đầy đủ, 3 seed). Nếu không, dùng Kaggle T4 + sinh theo batch + chia quota cho các thành viên.
4. Dù phần cứng nào: **sửa lỗi #1–#7 trước khi chạy bất kỳ run 7B nào** — chạy 19 giờ trên dữ liệu sai là mất cả tuần quota. *(Đã sửa ngày 2026-11-19, xem §4.7.)*

### 4.5 Chạy một phần dữ liệu để vừa một phiên 12 giờ có ổn không?

**Kết luận: chấp nhận được về mặt môn học nếu giữ đúng 5 điều kiện bên dưới.**

**Cập nhật 19/11:** deadline là 23:59 ngày 21/11, nên **chọn phương án B** (tập con, gói trong 1 phiên). Phương án A (đủ dữ liệu qua 2 phiên) chỉ xong vào chiều 20/11, không còn dự phòng, và cần có người khởi động lại phiên lúc sáng sớm. Ngoại lệ: nếu `pilot` báo "2 epochs of full data" ≤ 10 giờ, cứ chạy đủ dữ liệu.

Giới hạn 12 giờ áp dụng **cho từng run**, không phải cho cả thí nghiệm: SFT, SDFT và từng lần đánh giá là các phiên riêng. Chỉ có SDFT 7B vượt 12 giờ:

| Run trên T4 (ước tính) | Đủ dữ liệu | Vừa 1 phiên (~10,5 h train)? |
|---|---|---|
| SFT 7B Science / Tool | ~1,7 h / ~3,8 h | ✅ đủ dữ liệu |
| SDFT 1.5B, 3B (E3) | ~4 h, ~8,5 h | ✅ đủ dữ liệu |
| **SDFT 7B Science** | ~19 h | ❌ → ~1.450 mẫu (~57%) |
| **SDFT 7B Tool Use** | ~16 h | ❌ → ~2.700 mẫu (~70%) |

**Phương án A (khuyến nghị): đủ dữ liệu + resume.** `sdft_trainer.py --max_hours 11` tự lưu checkpoint và thoát; chạy lại đúng lệnh đó ở phiên sau là chạy tiếp, cùng thứ tự dữ liệu và cùng run W&B. Không phát sinh deviation, chỉ tốn thêm 1 phiên cho mỗi run SDFT 7B.

**Phương án B: dùng tập con, gói trong 1 phiên.** Hợp lệ theo `reproduction.md` ("the point is to say precisely how far the rebuild landed and why") nếu:

1. **SFT và SDFT dùng đúng cùng một tập con.** `--max_train N` luôn lấy cùng N mẫu đầu của một hoán vị cố định (`data_seed=0`), bất kể training seed.
2. **Không bao giờ cắt tập test** (507 / 97) và tập validation.
3. **Giữ số bước tối ưu gần với khi chạy đủ dữ liệu.** Ví dụ 1.300 mẫu với batch 16 cho ~160 bước trong 2 epoch, xấp xỉ 2.540 mẫu với batch 32. Batch 16 vẫn nằm trong khoảng sweep {16, 32, 64} của paper.
4. **Không cắt các thứ khác thay cho dữ liệu:** giữ 2 epoch cho SDFT và `max_new_tokens` ≥ 768 (demo Science có p90 khoảng 480 token).
5. **Ghi deviation và đo ảnh hưởng của nó.** Chạy thêm một sweep rẻ trên 1.5B với 25% / 50% / 100% dữ liệu để chỉ ra khoảng cách SDFT − SFT thay đổi thế nào theo lượng dữ liệu. Đây là bằng chứng cho slide phân tích sai lệch.

**Ảnh hưởng dự kiến:** ít bước hơn làm cả hai phương pháp lệch khỏi base ít hơn. SFT sẽ quên ít hơn, SDFT học task mới ít hơn, nên khoảng cách so với paper nhiều khả năng **thu hẹp**. Nếu SDFT vẫn thắng trên tập con thì kết luận càng chắc.

**Điều quan trọng hơn chuyện tập con: nhiễu thống kê của tập test.** Khoảng tin cậy 95% của độ chính xác là khoảng **±4 điểm** với 507 câu Science và **±9 điểm** với 97 câu Tool Use. Trong khi đó paper báo SDFT − SFT = **+4,0** (Science) và **+7,4** (Tool). Vậy với 1 seed, *dù chạy đủ dữ liệu*, chênh lệch độ chính xác trên task mới vẫn có thể nằm trong vùng nhiễu. Cách xử lý:
- Dùng **kiểm định cặp McNemar**: SFT và SDFT được chấm trên cùng các câu hỏi, nên so cặp có phương sai nhỏ hơn nhiều so với so hai tỉ lệ độc lập. Mọi output theo từng câu đã được lưu trong `eval_results/*.json`.
- Đặt trọng tâm vào **mức độ quên** (paper: 64,5 so với 53,4 điểm, tức chênh 11 điểm, đo trên hàng nghìn câu benchmark): đây là claim mà ngân sách của nhóm kiểm chứng được chắc chắn nhất.
- Nếu còn quota, ưu tiên **thêm seed** hơn là thêm dữ liệu.

**Đo thật trước khi quyết định:** chạy 3 bước SDFT 7B trên đúng máy (`--max_train 96 --epochs 1`). Trên W&B có `time/gen_sec` và `time/train_sec` của từng bước. Số giờ mỗi epoch ≈ `time/step_sec` × (số mẫu / batch). Dùng số này thay các ước tính ở trên.

### 4.6 Lưu ý khi resume trên Kaggle

- Checkpoint resume của 7B (2 adapter LoRA cộng trạng thái Adam) nặng khoảng 1,3 GB, lưu trong `checkpoints/<run>/resume/`.
- Muốn giữ nó qua phiên: chạy notebook bằng **Save Version → Save & Run All**, khi đó `/kaggle/working` trở thành output. Phiên sau thêm output đó làm input rồi copy thư mục về. Cách khác là upload thư mục lên một HF repo private.
- Thứ tự dữ liệu của mỗi epoch được sinh từ `(seed, epoch)`, nên resume không làm lệch dữ liệu.

### 4.7 Trạng thái sửa lỗi của §1.2 (cập nhật 2026-11-19)

| Lỗi §1.2 | Trạng thái |
|---|---|
| #1, #2 dữ liệu sai | ✅ Dùng dữ liệu chính thức (`data_loader.py`), có test kiểm tra kích thước và trường dữ liệu |
| #3, #4 chấm điểm sai | ✅ `answer_parsing.py`: parse `<answer>`; Tool Use có metric merged (kiểu paper) và strict, JSON lồng nhau parse đúng |
| #5 384 token | ✅ Mặc định 1024 |
| #6 fp16 | ✅ Tự chọn bf16 khi GPU hỗ trợ |
| #7 sinh từng mẫu | ✅ `--gen_batch_size` (mặc định 8) |
| #8 EMA trên A, B | ⚠️ Giữ nguyên, ghi là deviation |
| #9 batch / val / seed / lm-eval | ✅ batch 32, tập val cố định, lưu adapter mỗi epoch; lệnh lm-eval đủ 6 bộ trong README (chưa log tự động) |
| #10 cắt prompt teacher | ✅ Bỏ truncation |
| #11 `step_sec` | ✅ Đo cả bước, tách riêng thời gian sinh và train |

**Phát hiện mới khi đọc dữ liệu và code chính thức** (dùng cho slide 32):
- Hàm chấm Tool Use chính thức trích JSON bằng regex `{.*?}`, nên cắt hỏng tham số lồng nhau. Ngay cả câu trả lời chép y nguyên đáp án cũng chỉ được chấm đúng **95/97**, tức trần điểm là 97,9%.
- Một tên action trong tập test là `[Optional]updateUserProfile`, một lỗi dữ liệu mà regex `\w+` của tác giả không bắt được.
- 10 API của tập test **không xuất hiện** trong tập train, nên Tool Use đo khả năng tổng quát hóa sang công cụ mới.

---

## 5. Lịch đến deadline 23:59 thứ Bảy 21/11/2026 (3 tài khoản Kaggle)

Thời gian GPU là ràng buộc lớn nhất: mỗi run SDFT 7B mất khoảng 9–10 giờ, nên **3 run dài phải chạy song song từ tối nay (19/11)**. Phần đọc paper trong deck (slide 1–23) không cần kết quả, nên viết trong lúc GPU chạy. Job tương ứng nằm trong `scripts/kaggle_jobs.sh`.

| Thời điểm | Tài khoản A | Tài khoản B | Tài khoản C | Người không trực GPU |
|---|---|---|---|---|
| **T5 19/11** 17:00–18:00 | Mỗi tài khoản: xác minh số điện thoại (để bật GPU và Internet), thêm secret `WANDB_API_KEY`, kiểm tra quota còn ≥ 20 giờ | ← | ← | Commit + push code; tạo HF mirror (`scripts/mirror_official_data.py`, 5 phút) |
| 18:00–19:00 | `setup` + `pilot` → ra `N_SCI`, `N_TOOL` | — | — | Slide 1–8 (bài toán, các hướng đi trước) |
| ~19:00 | **`e1_sdft`** (Save & Run All, chạy nền ~9–10 h) | **`e1_sft_base` rồi `e2_sft`** (~8 h) | **`e2_sdft`** (~8–9 h) | Slide 9–13 (đóng góp, ký hiệu, pipeline) |
| **T6 20/11** ~07:00 | Kiểm tra W&B: loss có NaN không, tỉ lệ bị cắt, đã có `epoch2/` chưa | ← | ← | |
| 07:30–10:00 | `eval_sdft e1` | **`e3_small`** (1.5B + 3B, ~8 h, đến ~17:00) | `eval_sdft e2` | Slide 14–23 (thành phần, lý thuyết, độ phức tạp, kết quả của paper) |
| 10:00–22:00 | (dự phòng: chạy lại job hỏng) | | (dự phòng) | Slide 24–26 (scope, setup, kiến trúc); vẽ biểu đồ E1/E2 khi có số |
| **T7 21/11** sáng | | Kết quả E3 | | Slide 27–33: kết quả, phân tích sai lệch, so sánh với code gốc; bảng link W&B trong README |
| 16:00 | | | | Soát checklist trong `slide_outline.md` (notes ≤ 500 từ, trích dẫn, < 20 MB); để W&B project ở chế độ public |
| **20:00** | | | | **Nộp sớm** (pptx + link repo riêng), còn 4 giờ dự phòng |

**Nếu chỉ có 2 tài khoản:** C không có. A chạy `e2_sdft` từ ~10:00 ngày 20/11 sau `eval_sdft e1`, xong khoảng 20:00, rồi chạy `eval_sdft e2` tối 20/11. B chạy `e3_small` chỉ với 1.5B (`M_E3=Qwen/Qwen2.5-1.5B-Instruct`).

**Nếu pilot báo lỗi:**
- **OOM:** giảm `--gen_batch_size` xuống 4.
- **Loss NaN với fp16 trên T4:** chuyển run chính sang A100 thuê theo giờ hoặc Colab Pro (`QUANT=16`). Nếu không có tiền, dùng 3B làm mô hình chính và ghi rõ là deviation.
- **Run chết giữa epoch 2:** vẫn còn `epoch1/` để báo cáo, có thể chạy lại cùng lệnh để resume.

**Cắt theo thứ tự này nếu bị trễ:**
1. E3 chỉ giữ 1.5B.
2. lm-eval giảm `LIMIT=0.1`.
3. Bỏ việc chạy code gốc; slide so sánh dùng các phát hiện từ việc đọc code và dữ liệu (§4.7), những phát hiện này đã có sẵn.

**Không cắt:** tập test, chung tập con cho SFT và SDFT, 2 epoch cho SDFT.

---

## 6. Quy ước W&B và repo

- Một project public, ví dụ `sdft-continual-learning`.
- Tên run: `{lab}-{exp}-{method}-{model}-{task}-s{seed}`, ví dụ `lab1-E1-sdft-7b-science-s42`.
- Tag: `lab1` / `lab2`, `E1`…`E4`, `sft`/`sdft`/`base`, `eval-newtask`/`eval-prior`.
- Log tối thiểu: config đầy đủ, loss, LR, độ dài response, `step_sec` đúng, accuracy task mới, từng benchmark lm-eval, GPU type.
- README: bảng `thí nghiệm → script/notebook → link W&B → một dòng kết luận`; danh sách thư viện tái sử dụng có trích dẫn; ghi rõ dữ liệu lấy từ repo chính thức và đã mirror lên HF.

---

## 7. Danh sách deviation cần theo dõi (để đưa vào deck)

| Deviation | Nguyên nhân | Hướng ảnh hưởng dự kiến | Bằng chứng cần có |
|---|---|---|---|
| LoRA/QLoRA thay full FT | Phần cứng | SFT quên ít hơn paper; SDFT có thể học ít hơn | So số prior-tasks của SFT với Table 5 |
| EMA trên A, B riêng | Dùng 2 adapter LoRA | Teacher hơi khác EMA thật | Ghi chú; có thể thử teacher cố định (α=0) để so |
| 4-bit NF4 (nếu T4) | VRAM 16 GB | Base accuracy lệch nhẹ | Đo Base 4-bit vs bf16 trên 1.5B |
| max gen 1024 thay 2048 | Theo code chính thức, tiết kiệm thời gian | Gần như không (p90 demo ~480 token) | Tỉ lệ response bị cắt, log trên W&B |
| Không sweep đầy đủ | Ngân sách | Số của nhóm có thể thấp hơn "best-of-sweep" của paper | Ghi rõ cấu hình đã chọn |
| 1 seed thay 3 | Ngân sách | Không có khoảng tin cậy | Nếu kịp: 3 seed cho E1 |
| lm-eval lấy mẫu (nếu T4) | Thời gian | Phương sai cao hơn | Ghi `--limit`; khoảng tin cậy bootstrap |
| Phiên bản thư viện khác | Môi trường | Nhỏ | Ghi `pip freeze` vào W&B |

---

## 8. Rủi ro

- **Kaggle ngắt phiên 12 giờ** → bắt buộc lưu checkpoint + trạng thái optimizer + trạng thái EMA mỗi N bước.
- **OOM trên T4** khi tính log-softmax trên 152k vocab cho response dài → tính KL theo từng đoạn (chunk) dọc trục thời gian.
- **Teacher của 3B/1.5B yếu** (đúng như paper dự đoán) → không phải lỗi code; đo **độ chính xác của teacher** (sinh với ngữ cảnh có demo) để chứng minh.
- **HumanEval cần chạy code** → bật `HF_ALLOW_CODE_EVAL=1`, chạy trong môi trường cách ly.
- **Liêm chính học thuật**: không copy từ repo chính thức hay từ `SDFTTrainer` mới của TRL vào phần lõi; chỉ dùng để đối chiếu kết quả.
