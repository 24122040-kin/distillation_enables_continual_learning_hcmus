# Khung slide deck — Milestone 2 (Reading) + Lab 1 (Reproduction)

> Khung để nhóm điền nội dung. Viết ngày 2026-11-19.
> Ràng buộc từ `reading.md`: **20–40 slide** (tính cả phụ lục), một file `.pptx` **< 20 MB**, **mỗi slide có speaker notes ≤ 500 từ**, trích dẫn **IEEE đánh số ở chân mỗi slide** (không có slide References riêng), **mọi hình do nhóm tự vẽ lại** (dán hình của paper bị tính là copy), không dán code thô.
> Không có thuyết trình — **deck + notes là toàn bộ bài nộp**, nên notes phải tự đứng được.

Khung dưới đây có **36 slide** (32 slide chính + 4 phụ lục), chừa chỗ để thêm hoặc bớt.

Mỗi slide ghi: **Hình chính** (thứ cần vẽ), **Nội dung** (chữ trên slide, ngắn), **Notes** (ý cần viết trong speaker notes), **Trích dẫn**.

---

## Bảng trích dẫn dùng chung (giữ nguyên số trong cả deck)

| # | Tài liệu |
|---|---|
| [1] | I. Shenfeld, M. Damani, J. Hübotter, P. Agrawal, "Self-Distillation Enables Continual Learning," arXiv:2601.19897v2, 2026. |
| [2] | S. Ross, G. Gordon, D. Bagnell, "A reduction of imitation learning and structured prediction to no-regret online learning," AISTATS, 2011. |
| [3] | I. Shenfeld, J. Pari, P. Agrawal, "RL's Razor: Why online reinforcement learning forgets less," arXiv:2509.04259, 2025. |
| [4] | A. Y. Ng, S. Russell, "Algorithms for inverse reinforcement learning," ICML, 2000. |
| [5] | C. Snell, D. Klein, R. Zhong, "Learning by distilling context," arXiv:2209.15189, 2022. |
| [6] | R. Agarwal et al., "On-policy distillation of language models: Learning from self-generated mistakes," ICLR, 2024. |
| [7] | J. Kirkpatrick et al., "Overcoming catastrophic forgetting in neural networks," PNAS, 2017. |
| [8] | J. Schulman et al., "Trust region policy optimization," ICML, 2015. |
| [9] | R. Rafailov et al., "Direct preference optimization," NeurIPS, 2023. |
| [10] | Y. Tang, R. Munos, "On a few pitfalls in KL divergence gradient estimation for RL," arXiv:2506.09477, 2025. |
| [11] | T. Brown et al., "Language models are few-shot learners," NeurIPS, 2020. |
| [12] | K. Feng et al., "SciKnowEval," arXiv:2406.09098, 2024. |
| [13] | Q. Tang et al., "ToolAlpaca," arXiv:2306.05301, 2023. |
| [14] | L. Gao et al., "The language model evaluation harness," Zenodo, 2024. |
| [15] | Y. Wu et al., "On the generalization of SFT: A RL perspective with reward rectification (DFT)," arXiv:2508.05629, 2025. |
| [16] | K. Lu, Thinking Machines Lab, "On-policy distillation," 2025. |
| [17] | E. J. Hu et al., "LoRA: Low-rank adaptation of large language models," ICLR, 2022. |
| [18] | T. Dettmers et al., "QLoRA: Efficient finetuning of quantized LLMs," NeurIPS, 2023. |
| [19] | A. Yang et al. (Qwen Team), "Qwen2.5 technical report," arXiv:2412.15115, 2024. |
| [20] | I. Shenfeld, "Self-Distillation" (official code), GitHub: idanshen/Self-Distillation, 2026. |

---

## Phần A — Mở đầu (3 slide)

### 1. Trang tiêu đề
- **Hình chính:** một sơ đồ biểu tượng nhỏ "1 mô hình — 2 vai trò (thầy/trò)".
- **Nội dung:** tên paper, tác giả, venue/arXiv; tên nhóm, thành viên, MSSV; link repo + link W&B.
- **Notes:** paper nói gì trong 3 câu; nhóm chọn **core scope** (giải thích ở slide 24).
- **Trích dẫn:** [1]

### 2. Bài toán: học liên tục mà không quên
- **Hình chính:** trục thời gian: mô hình học kỹ năng 1 → 2 → 3; hai nhánh: "SFT: học cái mới, quên cái cũ" và "mục tiêu: tích lũy".
- **Nội dung:** định nghĩa continual learning bằng một câu; **ví dụ riêng của nhóm** (gợi ý: trợ lý bệnh viện Việt Nam được fine-tune theo phác đồ mới của Bộ Y tế rồi mất khả năng viết code, tuân thủ chỉ dẫn, hoặc trả lời tiếng Việt tự nhiên).
- **Notes:** vì sao vấn đề quan trọng (mô hình tĩnh sau triển khai; fine-tune lại toàn bộ tốn kém); catastrophic forgetting [7].
- **Trích dẫn:** [1], [7]

### 3. Lộ trình của deck
- **Hình chính:** sơ đồ 6 khối: Bối cảnh → Đóng góp → Phương pháp → Lý thuyết → Thực nghiệm của paper → Tái hiện.
- **Notes:** người đọc nên đọc theo thứ tự nào; phần tái hiện nằm ở slide 24–32.

---

## Phần B — Bối cảnh và các hướng đi trước (5 slide)

`reading.md` yêu cầu **≥ 2 hướng đi trước**, mỗi hướng phải có: ý tưởng cốt lõi → hạn chế còn lại → **hậu quả thực tế** của hạn chế đó.

### 4. On-policy và off-policy (khái niệm nền)
- **Hình chính:** hai hình minh họa song song: (a) off-policy: mô hình học trên quỹ đạo của chuyên gia; (b) on-policy: mô hình học trên quỹ đạo chính nó sinh ra. Vẽ thêm "compounding error": lệch khỏi trạng thái đã thấy → sai dồn lên.
- **Notes:** kết quả kinh điển của Ross et al. [2]: lỗi của imitation off-policy tăng theo bình phương độ dài chuỗi; on-policy (DAgger) tuyến tính.
- **Trích dẫn:** [2]

### 5. Hướng đi trước 1 — SFT (behavior cloning)
- **Hình chính:** khối "Ý tưởng / Hạn chế / Hậu quả" (3 cột).
- **Nội dung:** ý tưởng: tối đa log-likelihood của demonstration. Hạn chế: off-policy, kéo phân phối về phía demonstration. Hậu quả: Table 5 của paper — IFEval của SFT giảm từ 74,3 xuống 35,3 khi học Science (vẽ lại thành biểu đồ cột).
- **Trích dẫn:** [1], [2]

### 6. Hướng đi trước 2 — On-policy RL (RL's Razor)
- **Hình chính:** 3 cột như slide 5.
- **Nội dung:** ý tưởng: học trên mẫu tự sinh + reward → quên ít hơn [3]. Hạn chế: **cần hàm reward**. Hậu quả: không dùng được khi chỉ có demonstration (y khoa, tool use, kiến thức mới).
- **Trích dẫn:** [3]

### 7. Hướng đi trước 3 & 4 — IRL và Context distillation
- **Hình chính:** 2 khối 3 cột thu nhỏ.
- **Nội dung:** IRL [4]: suy ra reward từ demo rồi chạy RL; hạn chế: cần giả định cấu trúc mạnh (MaxEnt, adversarial, preference); hậu quả: khó mở rộng cho LLM. Context distillation [5]: mô hình có ngữ cảnh làm thầy cho chính nó; hạn chế: **offline**, ngữ cảnh cố định cho mọi câu hỏi; hậu quả: vẫn mắc lỗi off-policy.
- **Trích dẫn:** [4], [5], [6]

### 8. Khoảng trống paper lấp vào
- **Hình chính:** **bảng so sánh** — hàng: SFT, DFT [15], SFT + Re-invoke [16], On-policy RL, IRL, Context distillation, **SDFT**; cột: chỉ cần demo? on-policy? cần reward? cần mô hình thầy bên ngoài? số giai đoạn huấn luyện.
- **Notes:** SDFT là ô duy nhất "chỉ cần demo + on-policy + không cần reward + không cần thầy ngoài".
- **Trích dẫn:** [1], [3], [15], [16]

---

## Phần C — Đóng góp (2 slide)

### 9. Đóng góp theo cách paper tự nêu
- **Hình chính:** 4 thẻ (card): (1) thuật toán SDFT; (2) diễn giải SDFT như IRL với reward ngầm; (3) bằng chứng: học kỹ năng + nạp kiến thức, ít quên hơn SFT; (4) học tuần tự 3 kỹ năng không thoái lui.
- **Trích dẫn:** [1]

### 10. Phân loại đóng góp và tính mới
- **Hình chính:** bảng 3 hàng (Lý thuyết / Kỹ thuật / Thực nghiệm) × 2 cột ("Đóng góp" / "Trước đây chưa làm được gì").
- **Notes:** lý thuyết: phép quy reverse-KL-distillation ≡ policy gradient với reward ngầm `log π(y|x,c) − log π_k(y|x)`; kỹ thuật: teacher = chính mô hình + demo **riêng cho từng câu**, EMA teacher, mask token đầu; thực nghiệm: Pareto tốt hơn trên 3 task, scaling theo kích thước, model suy luận. **Nêu trung thực**: hai paper đồng thời (Zhao et al., Hübotter et al.) cũng đề xuất self-distillation.
- **Trích dẫn:** [1], [5], [6]

---

## Phần D — Phương pháp (10 slide) — trọng tâm kỹ thuật

### 11. Ký hiệu thống nhất của nhóm
- **Hình chính:** bảng ký hiệu: `x` (prompt), `c` (demonstration), `y = (y₁…y_T)` (response), `V` (vocab), `π_θ` (student), `π_φ` (teacher, EMA), `Ctx_S(x)`, `Ctx_T(x,c)`, `α` (tốc độ EMA), `β` (hệ số trust-region), `π_k` (policy ở bước k).
- **Notes:** paper dùng `π` không có chỉ số cho teacher; nhóm viết `π_φ` để rõ teacher là EMA. Ghi rõ mọi ký hiệu sẽ dùng nhất quán.

### 12. Phát biểu bài toán hình thức
- **Hình chính:** công thức mục tiêu: `min_θ E_{(x,c)~D} [ D_KL( π_θ(·|x) ‖ π_φ(·|x,c) ) ]`, kỳ vọng lấy theo `y ~ π_θ`.
- **Notes:** giải thích từng biến; vì sao kỳ vọng theo mẫu của chính student = on-policy; so với SFT: `min −E_{(x,c)} log π_θ(c|x)` (kỳ vọng theo dữ liệu chuyên gia = off-policy).
- **Trích dẫn:** [1]

### 13. Pipeline end-to-end ⭐ (hình quan trọng nhất)
- **Hình chính:** sơ đồ luồng do nhóm tự vẽ: `(x, c)` → **Student** sinh `y` → `y` được đưa vào **hai** lượt forward: student `π_θ(·|y<t, x)` và teacher `π_φ(·|y<t, x, c)` → KL theo từng token → gradient chỉ vào θ → cập nhật θ → `φ ← αθ + (1−α)φ`. Tô màu khác cho "không có gradient".
- **Notes:** đi qua một ví dụ thật từ tập Science (lấy 1 câu trong data, rút gọn): prompt, demo, câu trả lời student sinh ra, token nào teacher "không đồng ý".
- **Trích dẫn:** [1]

### 14. Thành phần 1 — Teacher có điều kiện demonstration
- **Hình chính:** khung prompt teacher (vẽ lại template 3 dòng) + minh họa: teacher **không chép nguyên văn** demo mà viết lời giải của riêng nó.
- **Nội dung:** *Làm gì / Vì sao cần / Hoạt động thế nào* (3 ý theo yêu cầu `reading.md`).
- **Notes:** dựa vào in-context learning [11]; paper báo teacher đạt 100% trên ToolAlpaca so với 42% của base.
- **Trích dẫn:** [1], [11]

### 15. Thành phần 2 — Rollout on-policy
- **Hình chính:** so sánh phân phối trạng thái: SFT học trên prefix của chuyên gia, SDFT học trên prefix do chính student tạo ra.
- **Nội dung:** làm gì / vì sao / như thế nào (1 rollout mỗi prompt; paper thử nhiều rollout nhưng không có lợi).
- **Trích dẫn:** [1], [2], [6]

### 16. Thành phần 3 — Hàm loss KL theo token
- **Hình chính:** 3 công thức xếp dọc: (Eq.1) KL mức chuỗi → (Eq.2) phân rã theo token → ước lượng giải tích trên toàn vocab (App. A.1). Bên cạnh: bảng 3 estimator (token-level / analytic / Rao-Blackwellized) × (bias, variance, chi phí).
- **Notes:** **điểm quan trọng**: lý thuyết dùng reverse KL, nhưng paper (Mục 3) và README code chính thức xác nhận **mọi kết quả dùng forward KL theo token** → ghi chú này là mầm cho poster.
- **Trích dẫn:** [1], [10], [20]

### 17. Thành phần 4 — EMA teacher và mask token đầu
- **Hình chính:** đường thẳng α: α=0 (teacher đông cứng = base: ổn định nhưng kém) — α nhỏ (EMA: tốt nhất) — α=1 (teacher = student: phân kỳ). Bên dưới: ví dụ artifact "Based on the example…" và cách mask 3 token đầu.
- **Notes:** vì sao vòng phản hồi on-policy khuếch đại nhiễu khi α=1; mask là heuristic, tác giả tự thừa nhận.
- **Trích dẫn:** [1]

### 18. Tính chất lý thuyết — SDFT là một dạng IRL
- **Hình chính:** chuỗi suy luận 5 bước dạng mũi tên: (1) RL có trust-region [8] → (2) nghiệm đóng `π* ∝ π_k exp(r/β)` [9] → (3) đảo lại: `r = β(log π* − log π_k)` → (4) **giả định ICL**: `π* ≈ π(·|x,c)` → (5) policy gradient với reward này = gradient của reverse KL.
- **Notes:** diễn giải trực quan: reward của một token = "teacher thích token này hơn student bao nhiêu". Nêu giới hạn: tương đương chỉ đúng **theo kỳ vọng**, chỉ với **reverse** KL, và hoàn toàn dựa trên giả định (4).
- **Trích dẫn:** [1], [8], [9]

### 19. Kiểm chứng giả định ICL
- **Hình chính:** (a) biểu đồ cột độ chính xác: base 42% vs teacher 100% (ToolAlpaca); (b) biểu đồ khoảng cách KL tới base: SFT 1,26 nats vs teacher 0,68 nats. **Vẽ lại bằng số**, không chụp hình paper. Nếu kịp: thêm cột số đo của nhóm.
- **Notes:** hai điều kiện: *tối ưu* (teacher đạt reward gần tối đa) và *lệch tối thiểu* (gần π_k nhất trong các policy tối ưu). Paper chỉ kiểm chứng thực nghiệm, không chứng minh.
- **Trích dẫn:** [1]

### 20. Độ phức tạp
- **Hình chính:** bảng: phương pháp × (forward/backward mỗi mẫu, số token sinh, bộ nhớ, số giai đoạn). Hàng: SFT, SDFT, GRPO, SFT + Re-invoke. Thêm cột "số đo của nhóm" (giờ GPU từ W&B).
- **Notes:** SFT: 1 fwd+bwd trên `|x|+|c|`. SDFT: sinh `T` token (giới hạn bởi băng thông bộ nhớ) + teacher fwd trên `|x|+|c|+T` + student fwd+bwd trên `|x|+T`; bộ nhớ thêm 1 bản teacher (nhóm dùng adapter LoRA thứ hai nên gần như miễn phí). Paper báo 2,5× FLOPs, ~4× wall-clock; dự báo của nhóm trên T4: ~11×. GRPO cần nhiều rollout mỗi prompt; Re-invoke cần 2 giai đoạn.
- **Trích dẫn:** [1], [16]

---

## Phần E — Phương pháp đánh giá và kết quả của paper (3 slide)

### 21. Dữ liệu và metric
- **Hình chính:** bảng 4 task: Science (SciKnowEval Chem L-3 [12]), Tool Use (ToolAlpaca [13]), Medical (HuatuoGPT-o1), Knowledge (thảm họa 2025) × (train/test, nguồn demo, cách chấm). Bên phải: 6 benchmark "khả năng cũ" [14].
- **Notes:** greedy decoding; Medical và Knowledge chấm bằng LLM-judge (GPT-5-mini) → khó tái lập; 3 seed, khoảng tin cậy 95%; mỗi phương pháp sweep LR × batch × epoch × α.
- **Trích dẫn:** [1], [12], [13], [14]

### 22. Kết quả chính của paper (headline)
- **Hình chính:** **vẽ lại biểu đồ Pareto** (trục x: độ chính xác task mới, trục y: trung bình khả năng cũ) cho Science và Tool Use **từ số của Table 5**.
- **Notes:** SDFT đạt 70,2 / 64,5 (Science) và 70,6 / 65,4 (Tool) so với SFT 66,2 / 53,4 và 63,2 / 56,0; base là 65,5 điểm khả năng cũ.
- **Trích dẫn:** [1]

### 23. Các kết quả khác của paper
- **Hình chính:** lưới 2×2 thu nhỏ: (a) học tuần tự 3 task; (b) nạp kiến thức: strict 89 vs SFT 80, OOD 98 vs 80; (c) scaling 3B/7B/14B: −3,3 / +4,0 / +6,9; (d) Olmo-3-7B-Think: SFT 23,5 vs SDFT 43,7.
- **Notes:** chọn kết quả nào để tái hiện và vì sao (dẫn sang slide 24).
- **Trích dẫn:** [1]

---

## Phần F — Tái hiện (9 slide)

### 24. Phạm vi đã chọn: Core scope
- **Hình chính:** bảng E1/E2/E3 (theo `reproduction_plan.md` §2), đánh dấu thí nghiệm nào của paper được giữ / bỏ / thay thế.
- **Notes:** lý do: H200 + full FT + sweep 54 cấu hình nằm ngoài ngân sách; Medical/Knowledge cần LLM-judge trả phí và chưa có dữ liệu công khai. E3 (kích thước mô hình) là phân tích bổ sung, đồng thời kiểm tra giả định ICL.

### 25. Thiết lập của nhóm so với paper
- **Hình chính:** bảng siêu tham số 3 cột: paper / code chính thức / nhóm (lấy từ `reproduction_plan.md` §3.2), tô màu các ô lệch.
- **Notes:** phần cứng thực tế, phiên bản thư viện, seed.
- **Trích dẫn:** [1], [17], [18], [20]

### 26. Kiến trúc cài đặt của nhóm
- **Hình chính:** sơ đồ module: `data_loader` → `sft_baseline` / `sdft_trainer` → `evaluate` + `eval_prior` → W&B. Tô màu: **tự cài đặt** (loss, rollout, EMA) vs **tái sử dụng** (transformers, peft, bitsandbytes, lm-eval).
- **Notes:** chi tiết cài EMA trên hai adapter LoRA và vì sao nó chỉ xấp xỉ EMA trọng số (EMA(B)·EMA(A) ≠ EMA(BA)). Không dán code.

### 27. E1 — Science Q&A: độ chính xác task mới
- **Hình chính:** biểu đồ cột nhóm: Base / SFT / SDFT, cột paper cạnh cột nhóm, ghi chênh lệch tuyệt đối và tương đối. Mỗi số có link W&B ở chân slide.

### 28. E1 — Khả năng cũ (forgetting)
- **Hình chính:** bảng 6 benchmark + trung bình, giống bố cục Table 5, hàng paper xen hàng nhóm; heatmap màu theo mức giảm so với base.

### 29. E1 + E2 — Biểu đồ Pareto: paper và nhóm
- **Hình chính:** biểu đồ Pareto của nhóm chồng lên điểm của paper (marker rỗng = paper, đặc = nhóm), 2 panel Science / Tool Use.
- **Notes:** kết luận định tính có giữ được không: SDFT có nằm ở góc trên-phải so với SFT không?

### 30. E3 — Kích thước mô hình: 1.5B / 3B / 7B
- **Hình chính:** biểu đồ đường giống Fig. 5 trái: độ chính xác SFT và SDFT theo kích thước, chú thích chênh lệch SDFT − SFT; thêm đường **độ chính xác của teacher** (mô hình có demo trong ngữ cảnh) để kiểm tra giả thuyết ICL.
- **Notes:** paper: −3,3 / +4,0 / +6,9 (3B/7B/14B). Nhóm có tái hiện được việc đổi dấu giữa 3B và 7B không?

### 31. Phân tích sai lệch ⭐
- **Hình chính:** bảng **Sai lệch → Nguyên nhân → Hướng ảnh hưởng → Bằng chứng** (từ `reproduction_plan.md` §7), mỗi dòng gắn link W&B.
- **Notes:** đây là phần chấm điểm cao nhất của lab ("separates a run from an understanding"). Gồm: LoRA vs full FT, 4-bit, EMA trên adapter, không sweep, 1 seed, phiên bản thư viện, phần cứng.

### 32. So sánh với code chính thức
- **Hình chính:** bảng "Paper nói / Code chính thức làm / Nhóm làm" cho các điểm đã phát hiện: (1) loss: paper trình bày reverse KL, README thừa nhận mọi kết quả dùng forward KL theo token; (2) max generation: paper 2048, code 1024; (3) LR: code mặc định 2e-5, README 5e-5; (4) warmup: paper 10 bước, code 10% số bước; (5) code dùng vLLM + importance-sampling correction (Algorithm 1 dòng 17 chỉ nhắc "if needed"); (6) template teacher của Science đặt thêm system prompt. Nếu nhóm chạy code gốc trên mô hình nhỏ: so số với bản cài đặt của nhóm.
- **Trích dẫn:** [1], [20]

---

## Phần G — Kết (1 slide)

### 33. Kết luận và cầu nối sang phần phê bình
- **Hình chính:** 3 cột: "Tái hiện được" / "Không tái hiện được hoặc lệch" / "Câu hỏi mở cho poster" (ví dụ: khoảng cách reverse/forward KL; giả định ICL với demo nhiễu; chi phí thực tế).

---

## Phụ lục (3 slide, vẫn tính vào giới hạn 40)

### A1. Siêu tham số đầy đủ của mọi run
### A2. Nhật ký tính toán: GPU, giờ chạy, chi phí, link W&B từng run
### A3. Ví dụ định tính: cùng một câu hỏi, câu trả lời của Base / SFT / SDFT (rút gọn, đánh dấu chỗ khác nhau)

---

## Checklist trước khi nộp

- [ ] 20 ≤ số slide ≤ 40; file `.pptx` < 20 MB (nén ảnh, dùng hình vector khi có thể).
- [ ] **Mọi slide** có speaker notes, mỗi notes ≤ 500 từ (đếm bằng script).
- [ ] Mọi trích dẫn ở dạng IEEE `[n]` ở chân đúng slide; **không** có slide References.
- [ ] Không có hình nào chụp hoặc cắt từ paper; mọi biểu đồ vẽ lại từ số liệu.
- [ ] Không có slide dán code thô.
- [ ] Mọi con số của nhóm có link run W&B (project để public).
- [ ] Link repo public được nộp **riêng** trên Moodle.
- [ ] Mỗi thành viên giải thích được mọi slide.
