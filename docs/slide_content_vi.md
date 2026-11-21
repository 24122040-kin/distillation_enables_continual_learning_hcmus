# Nội dung chi tiết 31 Slides (Tiếng Việt) — SDFT Term Project (HCMUS) — bản đã sửa

> **Quy ước:**
> - Ngôn ngữ: Tiếng Việt (giữ thuật ngữ ML chuẩn bằng tiếng Anh).
> - Mọi slide đều có: **Nội dung trên Slide** + **Trích dẫn IEEE cuối slide** + **Speaker Notes (≤ 500 từ)**.
> - **Trích dẫn đánh số thống nhất cho cả deck, theo thứ tự xuất hiện**; danh sách đầy đủ ở **Slide 31 — References**. Trên mỗi slide chỉ ghi số, ví dụ `[1, Sec. 3]`.
> - Ô `[...]` / `[Link W&B]` là chỗ điền số liệu sau khi chạy xong trên Kaggle.
> - Đơn vị: độ chính xác ghi bằng **%**, hiệu số giữa hai độ chính xác ghi bằng **pp** (điểm phần trăm).
> - 3 hình tự vẽ để upload lên Canva: `presentation/assets/pipeline_sdft.png`, `presentation/assets/pareto_paper_vs_repro.png`, `presentation/assets/sequential_cl_concept.png`.

## Những gì đã sửa so với bản trước

1. Trích dẫn đánh lại số theo thứ tự xuất hiện; thêm **Slide 31 — References** (deck thành 31 slide).
2. Slide 11: tách *cách ước lượng* (analytic, giảm phương sai) khỏi *hướng KL* (forward/reverse); Forward KL được ghi nguồn là **README repo gốc [13]**, không phải paper.
3. Slide 20, 25: ghi rõ tập con huấn luyện **352 mẫu Science / 528 mẫu Tool Use** (từ pilot ngày 19/11), lr, số epoch, GPU train/eval.
4. Slide 22: thêm cột **khoảng tin cậy 95%**; SFT ghi `epoch 1` (SFT chỉ chạy 1 epoch).
5. Slide 23: thêm 2 hàng Tool Use; ghi rõ metric của từng benchmark.
6. Slide 24: ghi chú E3 đánh giá thẳng trên test (không chọn epoch qua val) và cột Teacher đo trên val.
7. Slide 25: thêm độ lệch **train NF4 4-bit / eval fp16**; con số `~11x` đổi thành số đo thực tế.
8. Slide 21: bảng mapping dùng đúng tên job trong `scripts/kaggle_jobs.sh`.
9. Sửa câu chữ: slide 18 ("80% → 80%" không phải "tụt"), slide 12, 15 (bỏ "hoàn hảo", "triệt tiêu"), thống nhất số cấu hình sweep, Base ToolAlpaca thống nhất 42.9%.
10. Slide 29: nhấn mạnh chứng minh IRL (Eq. 6) chỉ đúng với Reverse KL.

## ⚠ Cần mở paper kiểm tra lại trước khi nộp

Các chi tiết dưới đây không xác minh được từ repo; trong bài có đánh dấu ⚠:

- Slide 10: các số 89% / 75% / 37% ở Appendix A.2 có phải thí nghiệm Knowledge Acquisition không.
- Slide 12: tập giá trị α được thử trong Appendix A.3.
- Slide 14: Teacher 100% và kiểm tra tay 50/50 trên ToolAlpaca.
- Slide 15: số FLOPs / wall-clock của SFT + Re-invoke.
- Slide 16, 20, 26: số cấu hình sweep (đang ghi "27 cấu hình mỗi phương pháp"), warmup 10 bước.
- Slide 18: thứ tự học nối tiếp Tool Use → Science → Medical.
- Slide 20: dữ liệu Knowledge Acquisition đã công khai hay chưa.
- Slide 22: độ chính xác Base 32.1% (Science) và 42.9% (Tool Use).
- Slide 23: paper dùng metric nào cho từng benchmark (acc hay acc_norm, IFEval strict hay loose).

---

## Slide 1 — Trang bìa (Cover)
* **Tiêu đề chính:** SELF-DISTILLATION ENABLES CONTINUAL LEARNING (SDFT)
* **Phụ đề:** Đọc hiểu Paper & Tái tạo Thực nghiệm (Core Scope) | Shenfeld et al., arXiv:2601.19897v2 (2026)
* **Thông tin nhóm:**
  * Môn học: Nhập môn Học máy (Introduction to Machine Learning) — HCMUS
  * Thành viên: `[Họ tên — MSSV]`, `[Họ tên — MSSV]`, `[Họ tên — MSSV]`
  * Public GitHub: `https://github.com/24122040-kin/distillation_enables_continual_learning_hcmus`
  * Public W&B Project: `[https://wandb.ai/<entity>/sdft-continual-learning]` *(nhớ chuyển project sang Public)*
* **IEEE Citation:** `[1]`
* **Speaker Notes:** Bộ slide trình bày quá trình đọc hiểu độc lập và tái tạo thực nghiệm theo Core Scope cho bài báo "Self-Distillation Enables Continual Learning" của Shenfeld và cộng sự (MIT & ETH Zurich, 2026). Phần cốt lõi của thuật toán do nhóm tự cài đặt lại từ mô tả toán học của bài báo; repo chính thức của tác giả chỉ được đọc để đối chiếu. Các sơ đồ và biểu đồ đều do nhóm tự vẽ, và mọi thực nghiệm được theo dõi công khai trên Weights & Biases, gắn với kho GitHub của nhóm.

---

## Slide 2 — Mục lục (Table of Contents)
* **Tiêu đề:** NỘI DUNG TRÌNH BÀY
* **Nội dung:**
  * `01` Bối cảnh & Động lực nghiên cứu
  * `02` Các phương pháp tiền nhiệm & Khoảng trống
  * `03` Đóng góp chính của Paper
  * `04` Phương pháp SDFT (Phân tích 5 lớp)
  * `05` Thiết lập & Kết quả công bố trong Paper
  * `06` Tái tạo Thực nghiệm (Core Scope) & Phân tích Sai số
  * `07` Đối chiếu Code gốc & Kết luận
* **Speaker Notes:** Bài trình bày gồm 31 slide chia làm 7 phần, bám sát yêu cầu Reading Milestone và Experimental Reproduction: từ động lực của bài toán Continual Learning, phân tích các phương pháp tiền nhiệm, giải phẫu 5 lớp của thuật toán SDFT, tóm tắt kết quả gốc, đến phần tái tạo bằng 4-bit QLoRA trên GPU miễn phí và phân tích độ lệch so với bài báo. Slide cuối là danh sách tài liệu tham khảo.

---

## Slide 3 — Bối cảnh & Ví dụ thực tế
* **Tiêu đề:** BỐI CẢNH: VÌ SAO LLM TĨNH THẤT BẠI SAU TRIỂN KHAI?
* **Khối trái (Vấn đề thực tế):**
  * Foundation models thường bị đóng băng trọng số sau khi deploy.
  * Prompting hay RAG chỉ thích nghi tạm thời lúc inference, không tích lũy kỹ năng mới vào tham số.
  * **Ví dụ:** Một trợ lý AI trong bệnh viện cần học thêm kỹ năng chẩn đoán lâm sàng mới, nhưng không được quên kỹ năng lập trình, toán học hay an toàn hội thoại đã có.
* **Khối phải (Nghịch lý cốt lõi):**
  * Cập nhật tham số cho tác vụ mới thường gây **Catastrophic Forgetting** (quên thảm khốc).
  * **Câu hỏi nghiên cứu:** *Làm sao để LLM học liên tục từ expert demonstrations mà vẫn giữ được năng lực cũ?*
* **IEEE Citation:** `[1, Sec. 1], [2], [3]`
* **Speaker Notes:** Các mô hình ngôn ngữ lớn sau khi triển khai thường không tự cập nhật trọng số. Nếu chỉ dùng retrieval hoặc prompt engineering, mô hình không thể nội hóa kỹ năng mới lâu dài. Nhưng khi fine-tune trực tiếp vào tham số, mô hình dễ quên các năng lực nền tảng — hiện tượng catastrophic forgetting đã được ghi nhận từ lâu trong học liên tục. Trong ví dụ trợ lý y tế, việc học thêm tri thức lâm sàng không được làm hỏng khả năng suy luận hay viết code sẵn có. Đây là động lực chính của bài toán Continual Learning cho LLM.

---

## Slide 4 — Khoảng trống giữa SFT và On-policy RL
* **Tiêu đề:** KHOẢNG TRỐNG GIỮA SFT VÀ ON-POLICY RL
* **Bảng so sánh 2 trục:**
  * **On-policy RL (PPO, GRPO):** Học trên chuỗi tự sinh → quên ít hơn, tổng quát hóa tốt → **Nhược điểm:** cần *explicit reward function* (khó định nghĩa trong nhiều bài toán thực tế).
  * **Supervised Fine-Tuning (SFT):** Chỉ cần *expert demonstrations* → dễ mở rộng → **Nhược điểm:** *off-policy* → quên nhiều và dễ học vẹt.
* **Mục tiêu của Paper:** Giữ được tính **on-policy** của RL nhưng chỉ dùng **expert demonstrations** (không cần reward function).
* **IEEE Citation:** `[1, Sec. 1], [4], [5]`
* **Speaker Notes:** Các nghiên cứu gần đây cho thấy học on-policy — huấn luyện trên dữ liệu do chính chính sách hiện tại sinh ra — giúp giảm catastrophic forgetting rõ rệt so với học off-policy. Tuy nhiên, học on-policy truyền thống gắn với Reinforcement Learning, vốn cần hàm thưởng tường minh. Ở nhiều bài toán như tư vấn y khoa hay nạp tri thức mới, ta chỉ có ví dụ mẫu từ chuyên gia chứ không có hàm thưởng. SDFT được đề xuất để lấp khoảng trống này.

---

## Slide 5 — Phương pháp tiền nhiệm 1: Supervised Fine-Tuning (SFT)
* **Tiêu đề:** TIỀN NHIỆM 1: SUPERVISED FINE-TUNING (SFT)
* **3 cột (Ý tưởng | Hạn chế | Hệ quả):**
  * **Ý tưởng cốt lõi:** Tối thiểu hóa cross-entropy trên các cặp $(x, y^*)$ cố định của chuyên gia.
  * **Hạn chế:** Hoàn toàn **off-policy** — mô hình chỉ học trên phân phối trạng thái của chuyên gia, chưa từng được sửa lỗi trên trạng thái do chính mình sinh ra.
  * **Hệ quả:**
    1. Tích tụ sai số (*compounding errors*) khi suy luận lệch khỏi quỹ đạo mẫu.
    2. Quên năng lực cũ (Table 5: Prior Avg giảm `65.5% → 53.4%` sau khi SFT trên Science Q&A).
    3. Rút ngắn chuỗi suy luận (CoT collapse) nếu dữ liệu mẫu chỉ có đáp án ngắn (Table 2).
* **IEEE Citation:** `[1, Table 2, Table 5], [6], [7]`
* **Speaker Notes:** Phương pháp tiền nhiệm đầu tiên là SFT. Về cốt lõi, SFT là behavioral cloning trên tập dữ liệu ngoại tuyến. Hạn chế lớn nhất là tính off-policy: khi suy luận, chỉ cần mô hình chọn một token khác dữ liệu mẫu là nó rơi vào trạng thái chưa từng gặp, và sai số tích tụ nhanh. Bài báo ghi nhận hai hệ quả rõ ràng: hiệu năng trên các benchmark cũ giảm mạnh, và chuỗi suy luận của mô hình reasoning bị rút ngắn.

---

## Slide 6 — Phương pháp tiền nhiệm 2: Inverse RL & Offline Context Distillation
* **Tiêu đề:** TIỀN NHIỆM 2: INVERSE RL & CONTEXT DISTILLATION
* **Khối trái (Inverse RL — IRL):**
  * **Ý tưởng:** Suy ra hàm reward ẩn từ demonstrations, sau đó chạy on-policy RL.
  * **Hạn chế:** Bài toán ill-posed nếu thiếu giả định cấu trúc mạnh (MaxEnt/Boltzmann, adversarial classifier, hoặc cặp preference như RLHF).
  * **Hệ quả:** Khó mở rộng sang không gian token lớn của LLM; cần thêm dữ liệu gán nhãn.
* **Khối phải (Offline Context Distillation):**
  * **Ý tưởng:** Cho teacher đọc ngữ cảnh bổ sung (hướng dẫn / few-shot) rồi distill sang student không có ngữ cảnh đó.
  * **Hạn chế:** Ngữ cảnh cố định (static prefix) và huấn luyện **off-policy** trên dữ liệu teacher sinh sẵn.
  * **Hệ quả:** Vẫn lệch phân phối khi student tự sinh; không sửa được lỗi của chính student.
* **IEEE Citation:** `[8], [9], [10]`
* **Speaker Notes:** Nhóm tiền nhiệm thứ hai là Inverse RL và Offline Context Distillation. IRL cố khôi phục hàm thưởng từ dữ liệu mẫu để chạy on-policy RL, nhưng cần các giả định chặt nên khó scale lên LLM tổng quát. Context Distillation truyền thống dùng prompt tĩnh và chưng cất offline trên output của teacher, nên vẫn gặp vấn đề lệch phân phối như SFT. Các công trình on-policy distillation như GKD đã cho student học trên chuỗi tự sinh, nhưng cần một teacher riêng. SDFT kết hợp cả hai ý: chưng cất on-policy, và teacher chính là mô hình đang học, được điều kiện hóa bằng từng ví dụ cụ thể.

---

## Slide 7 — Đóng góp của Paper (3 nhóm)
* **Tiêu đề:** ĐÓNG GÓP CHÍNH & TÍNH MỚI SO VỚI TIỀN NHIỆM
* **3 thẻ:**
  * **Lý thuyết:** Chỉ ra rằng tối thiểu hóa Reverse KL giữa student và teacher có demonstration tương đương với on-policy RL tối ưu một *intrinsic reward* kiểu IRL, dựa trên *In-Context Assumption* — không cần học reward model.
  * **Kỹ thuật:** Đề xuất **SDFT** — một LLM đóng 2 vai Student & EMA Teacher; 1 on-policy rollout mỗi prompt; ước lượng KL giải tích theo từng token; mask vài token đầu để tránh artifact.
  * **Thực nghiệm:** Nằm trên Pareto frontier ở 3 bài toán Skill Learning; 98% OOD ở Knowledge Acquisition; học nối tiếp 3 tác vụ mà không quên tác vụ trước; cải thiện reasoning model khi chỉ có đáp án cuối.
* **IEEE Citation:** `[1, Sec. 3–4]`
* **Speaker Notes:** Đóng góp của bài báo chia thành 3 nhóm. Về lý thuyết, tác giả nối chưng cất ngữ cảnh với Inverse RL mà không cần hàm thưởng tường minh. Về kỹ thuật, SDFT chỉ cần một rollout cho mỗi prompt — nhẹ hơn group sampling của GRPO — và ổn định nhờ EMA teacher. Về thực nghiệm, SDFT tích lũy kỹ năng liên tục và nạp tri thức mới tốt hơn SFT. Lưu ý: chứng minh lý thuyết dựa trên Reverse KL, trong khi kết quả thực nghiệm dùng Forward KL; nhóm sẽ quay lại điểm này ở slide 11 và 29.

---

## Slide 8 — Method (Lớp 1): Ký hiệu & Phát biểu Bài toán
* **Tiêu đề:** PHƯƠNG PHÁP (LỚP 1): KÝ HIỆU & PHÁT BIỂU BÀI TOÁN
* **Bảng ký hiệu:**
  * $\mathcal{D} = \{(x_i, c_i)\}_{i=1}^N$: câu hỏi $x_i$ và expert demonstration $c_i$.
  * $\pi_\theta(\cdot \mid x)$: Student (tham số $\theta$), chỉ nhận $x$.
  * $\pi_\phi(\cdot \mid x, c)$: Teacher (tham số EMA $\phi$), nhận cả $x$ và $c$.
  * $\text{Ctx}_S(x), \text{Ctx}_T(x, c)$: hàm định dạng prompt cho Student và Teacher.
  * $\alpha, \eta, T$: tốc độ cập nhật EMA, learning rate, độ dài sinh tối đa.
* **Phát biểu bài toán (Eq. 1):**
  $$\mathcal{L}(\theta) = \mathbb{E}_{(x,c)\sim\mathcal{D}}\left[ D_{\text{KL}}\big(\pi_\theta(\cdot \mid x) \parallel \pi_\phi(\cdot \mid x, c)\big) \right]$$
* **IEEE Citation:** `[1, Sec. 3, Alg. 1]`
* **Speaker Notes:** Lớp đầu tiên là chuẩn hóa ký hiệu và phát biểu bài toán. Đầu vào là các cặp câu hỏi x và lời giải mẫu c. Student pi_theta chỉ thấy x và tự sinh chuỗi y. Teacher pi_phi có cùng kiến trúc, nhưng tham số là trung bình trượt mũ (EMA) của theta, và được điều kiện hóa thêm bởi c. Mục tiêu là thu hẹp phân kỳ KL giữa Student và Teacher dọc theo chính các chuỗi do Student sinh ra. Phương trình 1 viết theo Reverse KL — student đứng trước.

---

## Slide 9 — Method (Lớp 2): Pipeline End-to-End (Tự vẽ)
* **Tiêu đề:** PHƯƠNG PHÁP (LỚP 2): LUỒNG XỬ LÝ END-TO-END CỦA SDFT
* **Hình:** `presentation/assets/pipeline_sdft.png`
* **4 bước dưới hình:**
  1. Student nhận $\text{Ctx}_S(x)$ và tự sinh rollout $y \sim \pi_\theta(\cdot|x)$.
  2. Teacher nhận $\text{Ctx}_T(x, c)$ và chấm trên **cùng chuỗi $y$** để lấy logits $\ell^T_t$ (không lan truyền gradient).
  3. Student tính logits $\ell^S_t$ trên $y$; tính KL giải tích theo từng token (Forward KL như code gốc), bỏ qua $k$ token đầu.
  4. Cập nhật $\theta$ bằng AdamW, rồi cập nhật EMA Teacher: $\phi \leftarrow \alpha\theta + (1-\alpha)\phi$.
* **IEEE Citation:** `[1, Fig. 2, Alg. 1]` *(sơ đồ do nhóm tự vẽ lại)*
* **Speaker Notes:** Sơ đồ do nhóm tự thiết kế, mô tả một bước huấn luyện SDFT. Điểm mấu chốt là cả Student và Teacher đều tính phân phối token trên cùng một chuỗi y do Student tự sinh — đó là tính on-policy. Khi Student đi lệch ở một bước nào đó, Teacher có ngữ cảnh c sẽ cho tín hiệu hiệu chỉnh ngay tại token đó. Cuối bước, Student được cập nhật bằng gradient, còn Teacher được cập nhật mượt qua EMA.

---

## Slide 10 — Method (Lớp 3a): Demonstration-Conditioned Teacher
* **Tiêu đề:** LỚP 3A: TEACHER ĐIỀU KIỆN HÓA BẰNG DEMONSTRATION
* **Khối trái (What & Why):**
  * **What:** Dùng chính LLM đang học, ghép thêm ví dụ mẫu $c$ vào prompt $\text{Ctx}_T(x, c)$.
  * **Why:** Tận dụng **In-Context Learning (ICL)** để biến một demonstration rời rạc thành phân phối xác suất ở mức token, không cần teacher lớn hơn.
* **Khối phải (How — template ở Section 3):**
  ```text
  <Question>
  This is an example for a response to the question:
  <Demonstration>
  Now answer with a response of your own, including the thinking process:
  ```
  * Câu *"answer with a response of your own, including the thinking process"* giúp Teacher không chép nguyên văn $c$ mà tự tái tạo lập luận trung gian.
* **IEEE Citation:** `[1, Sec. 3, App. A.2], [11]`
* **Speaker Notes:** Thành phần thứ nhất là cách xây dựng Teacher. Thay vì huấn luyện một mô hình thầy riêng, SDFT tận dụng khả năng học theo ngữ cảnh của chính mô hình. Template được thiết kế để mô hình không copy đáp án mẫu c mà tự diễn giải lại quá trình suy nghĩ dẫn tới đáp án. ⚠ Appendix A.2 so sánh các cách xây dựng ngữ cảnh cho Teacher trong thí nghiệm Knowledge Acquisition: đưa cả văn bản nguồn lẫn đáp án mẫu đạt 89% strict, chỉ văn bản đạt 75%, chỉ đáp án đạt 37%.

---

## Slide 11 — Method (Lớp 3b): On-Policy Rollout & Analytic KL
* **Tiêu đề:** LỚP 3B: ON-POLICY ROLLOUT & ƯỚC LƯỢNG KL
* **Khối trái (On-policy Rollout):**
  * Sinh đúng **1 trajectory** $y = (y_1, \dots, y_T) \sim \pi_\theta(\cdot|x)$ cho mỗi prompt.
  * Appendix A.1: tăng số rollout mỗi prompt không cải thiện đáng kể nhưng tốn compute gấp nhiều lần.
* **Khối phải (2 lựa chọn tách biệt):**
  * **(a) Cách ước lượng — giảm phương sai:** cộng giải tích trên toàn bộ từ vựng $\mathcal{V}$ tại mỗi vị trí $t$ (*full analytic per-token estimator*), thay vì chỉ dùng token đã sample. Appendix A.1 so sánh 3 estimator và chọn cách này.
  * **(b) Hướng KL — paper vs code:**
    * Paper (Eq. 1–2): **Reverse KL** $D_{\text{KL}}(\pi_\theta \parallel \pi_\phi)$.
    * README repo chính thức: mọi kết quả trong paper dùng **Forward KL** theo từng token: $\sum_{v\in\mathcal{V}} \pi_\phi(v\mid\cdot)\log\frac{\pi_\phi(v\mid\cdot)}{\pi_\theta(v\mid\cdot)}$.
    * Bản của nhóm cài cả hai (`--kl_mode forward|reverse`), mặc định `forward` để so được với Table 5.
* **IEEE Citation:** `[1, Sec. 3, App. A.1], [12], [13]`
* **Speaker Notes:** Thành phần thứ hai là rollout và hàm mất mát. Khác GRPO phải sinh nhiều mẫu để ước lượng advantage, SDFT chỉ cần một rollout mỗi prompt. Có hai lựa chọn độc lập cần tách bạch. Thứ nhất là cách ước lượng: Appendix A.1 so sánh ước lượng dựa trên token đã sample, ước lượng giải tích theo từng token, và ước lượng Rao-Blackwellized, rồi chọn cách giải tích — cộng trên toàn bộ từ vựng — vì ổn định nhất. Thứ hai là hướng của KL: lý thuyết viết Reverse KL, nhưng README của repo chính thức thừa nhận mọi kết quả công bố dùng Forward KL. Đây không phải chi tiết ghi trong paper, mà là phát hiện khi đối chiếu code.

---

## Slide 12 — Method (Lớp 3c): EMA Teacher & Artifact Masking
* **Tiêu đề:** LỚP 3C: EMA TEACHER & XỬ LÝ LEARNED ARTIFACTS
* **Khối trái (Chọn tham số Teacher — Appendix A.3):**
  * **Frozen Base ($\alpha = 0$):** ổn định nhưng không theo kịp tiến bộ của Student → kết quả kém hơn.
  * **Live Student ($\alpha = 1$):** vòng phản hồi dương khuếch đại nhiễu → **phân kỳ** (Fig. 8).
  * **EMA Teacher** (⚠ paper thử $\alpha \in \{0.01, 0.02, 0.05\}$; code gốc dùng `0.01`): vừa theo kịp Student vừa làm mượt cập nhật.
* **Khối phải (Learned Artifacts — Section 5):**
  * Vì thấy ngữ cảnh $c$, Teacher hay mở đầu bằng *"Based on the text..."* hoặc *"Following the example..."*.
  * Student dù không thấy $c$ vẫn học theo các cụm này.
  * **Cách xử lý:** bỏ loss trên vài token đầu của câu trả lời. Paper không nêu con số cụ thể; code gốc đặt `3`, nhóm dùng `3`.
* **IEEE Citation:** `[1, App. A.3, Fig. 8, Sec. 5], [13]`
* **Speaker Notes:** Thành phần thứ ba giải quyết độ ổn định và nhiễu ngôn ngữ. Phụ lục A.3 cho thấy nếu dùng mô hình gốc đóng băng làm thầy thì trần hiệu năng bị giới hạn, còn nếu dùng chính student hiện tại làm thầy thì tối ưu nhanh chóng phân kỳ. EMA nằm giữa hai thái cực đó. Ngoài ra, mask loss ở vài token mở đầu là mẹo thực dụng để Student không bắt chước các câu dẫn thừa từ Teacher.

---

## Slide 13 — Method (Lớp 4a): SDFT như một bài toán Inverse RL
* **Tiêu đề:** LỚP 4A: CƠ SỞ LÝ THUYẾT — SDFT NHƯ MỘT BÀI TOÁN INVERSE RL
* **Nội dung toán học (Section 3.1):**
  * Trust-Region RL (Eq. 3): $\pi_{k+1} = \arg\max_\pi \mathbb{E}[r(y,x)] - \beta D_{\text{KL}}(\pi(\cdot|x) \parallel \pi_k(\cdot|x))$.
  * Nghiệm dạng đóng: $\pi^*_{k+1}(y|x) \propto \pi_k(y|x)\exp\big(\tfrac{1}{\beta}r(y,x)\big)$.
  * **Giả định In-Context (Eq. 4):** $\pi^*_{k+1}(y|x) \approx \pi(y|x, c)$.
  * Suy ra **intrinsic token-level reward** (Eq. 5):
    $$r_t(y_t \mid y_{<t}, x, c) = \log \pi(y_t \mid y_{<t}, x, c) - \log \pi_k(y_t \mid y_{<t}, x)$$
  * **Kết quả (Eq. 6):** policy gradient tối ưu reward này trùng với gradient của **Reverse KL** ở Eq. 2.
* **IEEE Citation:** `[1, Sec. 3.1], [14]`
* **Speaker Notes:** Section 3.1 chứng minh SDFT không chỉ là nén ngữ cảnh mà tương đương với một thuật toán Inverse RL on-policy. Từ bài toán tối ưu vùng tin cậy, nghiệm tối ưu là phân phối cũ nghiêng theo hàm mũ của reward. Giả định rằng mô hình khi thấy mẫu c sẽ xấp xỉ chính sách tối ưu đó, ta suy ra được phần thưởng nội tại ở từng token. Giới hạn: chứng minh dựa trên Reverse KL và cần ICL đủ mạnh — hai điểm nhóm sẽ phân tích ở slide 29.

---

## Slide 14 — Method (Lớp 4b): Kiểm chứng Giả định ICL
* **Tiêu đề:** LỚP 4B: KIỂM CHỨNG 2 ĐIỀU KIỆN CỦA GIẢ ĐỊNH ICL
* **Khối trái (Điều kiện 1 — Optimality):**
  * Teacher $\pi(\cdot|x,c)$ phải đạt reward gần tối ưu trên tác vụ mới.
  * **Trên ToolAlpaca (`Qwen2.5-7B-Instruct`):**
    * Base (không có $c$): **42.9%**.
    * ⚠ Teacher (có $c$): **100%**; kiểm tra tay 50/50 chuỗi suy luận đều có CoT hợp lệ, không chép máy móc.
* **Khối phải (Điều kiện 2 — Minimal Deviation):**
  * Chính sách tối ưu phải gần $\pi_k$ theo nghĩa KL.
  * **$D_{\text{KL}}(\pi \parallel \pi_0)$ trên ToolAlpaca:**
    * SFT so với Base: **1.26 nats**.
    * Teacher so với Base: **0.68 nats** (khoảng một nửa).
* **Nhóm tự đo lại:** độ chính xác Teacher (`--with_demo`) trên tập val — xem slide 24.
* **IEEE Citation:** `[1, Sec. 3.2, Fig. 2 (phải)]`
* **Speaker Notes:** Để giả định In-Context cho ra một Teacher tốt trong khuôn khổ trust-region, cần hai điều kiện: tính tối ưu và độ lệch tối thiểu. Trên ToolAlpaca với Qwen2.5-7B-Instruct, thêm mẫu c nâng độ chính xác từ khoảng 43% lên 100%. Đồng thời, KL của Teacher so với mô hình gốc chỉ 0.68 nats, khoảng một nửa so với 1.26 nats của SFT. Nhóm cũng tự đo độ chính xác của Teacher ở cả 3 kích thước mô hình để kiểm chứng điều kiện thứ nhất.

---

## Slide 15 — Method (Lớp 5): Độ phức tạp Thời gian, Bộ nhớ & Tính toán
* **Tiêu đề:** LỚP 5: PHÂN TÍCH ĐỘ PHỨC TẠP & SO SÁNH TIỀN NHIỆM
* **Bảng so sánh:**

| Phương pháp | Rollout | FLOPs | Wall-clock | Bộ nhớ |
| :--- | :---: | :---: | :---: | :--- |
| SFT | 0 | `1.0x` | `1.0x` | 1 model + optimizer |
| ⚠ SFT + Re-invoke | on-policy ở pha 2 | `> 2.5x` | `> 4.0x` (2 pha nối tiếp) | Student + Frozen Base |
| GRPO (on-policy RL) | $G \approx 4\text{--}16$ / prompt | rất cao | rất cao | Policy + Reference (+ reward) |
| **SDFT (Paper)** | **1 / prompt** | **`~2.5x`** | **`~4.0x`** | Student + EMA Teacher |

* **Thích nghi bộ nhớ của nhóm:** hai LoRA adapter (`default` = student, `teacher` = EMA) gắn trên **cùng một base model lượng tử hóa** → không phải nạp bản model thứ hai cho Teacher.
* **IEEE Citation:** `[1, Sec. 5]`
* **Speaker Notes:** Về chi phí, SDFT đắt hơn SFT khoảng 2.5 lần về FLOPs và 4 lần về thời gian thực, do phải sinh chuỗi autoregressive ở mỗi bước. Nhưng so với các phương pháp chống quên hai giai đoạn như SFT kết hợp Re-invoke, hay RL theo nhóm như GRPO, SDFT rẻ hơn vì chỉ sinh một rollout mỗi prompt và có tín hiệu học dày đặc ở mức token. Trong bản tái tạo, nhóm dùng hai adapter LoRA trên cùng một base model, nên Teacher chỉ tốn thêm bộ nhớ cho adapter và activation chứ không phải cả một mô hình 7B.

---

## Slide 16 — Thiết lập Thực nghiệm trong Paper
* **Tiêu đề:** PHƯƠNG PHÁP THỰC NGHIỆM TRONG PAPER
* **Khối trái (2 bối cảnh & dữ liệu):**
  * **Skill Learning (`Qwen2.5-7B-Instruct`):**
    1. *Science Q&A:* SciKnowEval Chemistry L-3 (chia 75/5/20%).
    2. *Tool Use:* ToolAlpaca (chấm bằng regex khớp API call).
    3. *Medical:* HuatuoGPT-o1 (~20K train, 1K eval, chấm bằng GPT-5-mini).
  * **Knowledge Acquisition:** bài Wikipedia về thiên tai năm 2025 (~200K tokens); đo Strict, Lenient và **OOD Accuracy**.
* **Khối phải (Đánh giá quên & giao thức):**
  * **Prior Capabilities (`lm-evaluation-harness`):** 6 benchmark — HellaSwag, TruthfulQA, MMLU, IFEval, Winogrande, HumanEval.
  * **Phần cứng & sweep:** 1× NVIDIA H200, Full Fine-Tuning `bfloat16`, ⚠ sweep 27 cấu hình mỗi phương pháp, 3 random seeds, greedy decoding khi đánh giá.
* **IEEE Citation:** `[1, Sec. 4.1, App. B], [15], [16], [17], [18], [19]`
* **Speaker Notes:** Bài báo đánh giá SDFT trên Qwen2.5-7B-Instruct ở hai bối cảnh: học kỹ năng mới (Science Q&A, Tool Use, Medical) và nạp tri thức xuất hiện sau thời điểm cutoff (thiên tai năm 2025). Để đo mức quên, tác giả dùng lm-evaluation-harness trên 6 benchmark tổng quát. Mọi thí nghiệm gốc chạy full fine-tuning trên một GPU H200 và lấy trung bình qua 3 seed.

---

## Slide 17 — Kết quả công bố: Pareto Frontier (Skill Learning)
* **Tiêu đề:** KẾT QUẢ CÔNG BỐ: PARETO FRONTIER (SKILL LEARNING)
* **Hình:** `presentation/assets/pareto_paper_vs_repro.png`
* **Khối phải (Table 5 — `Qwen2.5-7B`):**
  * **Base:** Prior Avg = **65.5%**.
  * **Science Q&A:** SDFT **70.2%** vs SFT 66.2% (+4.0 pp) | Prior Avg: SDFT **64.5%** vs SFT **53.4%**.
  * **Tool Use:** SDFT **70.6%** vs SFT 63.2% (+7.4 pp) | Prior Avg: SDFT **65.4%** vs SFT **56.0%**.
  * **Medical:** SDFT **40.2%** vs SFT 35.5% (+4.7 pp) | Prior Avg: SDFT **65.4%** vs SFT **60.2%**.
* **IEEE Citation:** `[1, Table 5, Fig. 4]` *(biểu đồ do nhóm vẽ lại)*
* **Speaker Notes:** Hình minh họa biên Pareto giữa độ chính xác trên tác vụ mới (trục hoành) và năng lực cũ (trục tung). Trên cả Science Q&A, Tool Use và Medical, SDFT đều nằm ở góc trên-phải: điểm tác vụ mới cao hơn SFT từ 4.0 đến 7.4 pp, đồng thời giữ điểm trung bình 6 tác vụ cũ ở mức 64.5–65.4%, gần như bằng mốc 65.5% của mô hình gốc.

---

## Slide 18 — Kết quả công bố: Knowledge Acquisition & Sequential CL
* **Tiêu đề:** KẾT QUẢ CÔNG BỐ: NẠP TRI THỨC & HỌC NỐI TIẾP
* **Hình trái:** `presentation/assets/sequential_cl_concept.png`
* **Khối phải (Table 1 — Knowledge Acquisition):**

| Phương pháp | Strict | Lenient | OOD |
| :--- | :---: | :---: | :---: |
| Base | 0% | 0% | 0% |
| CPT (Continual Pre-training) | 9% | 37% | 7% |
| SFT | 80% | 95% | 80% |
| **SDFT** | **89%** | **100%** | **98%** |
| Oracle RAG (tham chiếu) | — | — | 100% |

* **Ý nghĩa:** SDFT không chỉ nhớ câu trả lời mà dùng được tri thức mới để trả lời câu hỏi gián tiếp (OOD).
* **IEEE Citation:** `[1, Fig. 3, Table 1]` *(hình do nhóm vẽ lại)*
* **Speaker Notes:** Ở thí nghiệm học nối tiếp 3 kỹ năng (⚠ Tool Use → Science → Medical), SDFT tích lũy dần kỹ năng mà không làm giảm tác vụ trước, trái với dao động mạnh của SFT. Ở thí nghiệm nạp tri thức năm 2025, SFT dừng ở 80% với câu hỏi OOD, trong khi SDFT đạt 98%, gần mức Oracle RAG 100%. Khoảng cách 18 pp ở cột OOD lớn hơn nhiều so với 9 pp ở cột Strict, cho thấy lợi thế của SDFT nằm chủ yếu ở khả năng tổng quát hóa.

---

## Slide 19 — Kết quả công bố: Model Scale, Pass@k & Reasoning
* **Tiêu đề:** CÁC PHÁT HIỆN QUAN TRỌNG TỪ ABLATION TRONG PAPER
* **3 thẻ:**
  * **1. Kích thước mô hình (Fig. 5 trái), SDFT − SFT trên Science Q&A:**
    * `3B`: ICL quá yếu → SDFT **thua** SFT (`−3.3 pp`).
    * `7B`: `+4.0 pp` | `14B`: `+6.9 pp`.
    * *(Lý do nhóm giữ 7B cho kết quả chính và chạy thêm E3.)*
  * **2. Pass@k tới $k=128$ (Fig. 5 phải):**
    * SDFT cao hơn SFT và Base ở mọi $k$ → học được kỹ năng thật, không phải chỉ giảm entropy.
  * **3. Reasoning model không cần CoT (Table 2):**
    * `Olmo-3-7B-Think`: Base `31.2%` (4612 tokens) → **SFT** `23.5%` (3273 tokens) → **SDFT** **`43.7%`** (4180 tokens).
* **IEEE Citation:** `[1, Sec. 4.4–4.5, Fig. 5, Table 2]`
* **Speaker Notes:** Ba kết quả này quan trọng cho việc tái tạo. Thứ nhất, ICL phụ thuộc vào quy mô: ở 3B, SDFT thua SFT 3.3 pp và chỉ vượt từ 7B trở lên. Thứ hai, pass@k tăng đều tới k = 128, cho thấy mô hình thực sự mở rộng khả năng giải bài chứ không chỉ trở nên chắc chắn hơn với một đáp án. Thứ ba, SDFT tinh chỉnh được mô hình suy luận chỉ bằng đáp án cuối mà không làm ngắn chuỗi suy nghĩ, trong khi SFT làm độ dài giảm từ khoảng 4600 xuống 3300 token và độ chính xác giảm theo.

---

## Slide 20 — Phạm vi Tái tạo (Core Scope) & Thích nghi phần cứng
* **Tiêu đề:** PHẠM VI TÁI TẠO: CORE SCOPE & THÍCH NGHI PHẦN CỨNG
* **Khối trái (Core Scope):**
  * Paper chạy Full FT trên 1× H200 (141 GB VRAM) với sweep lớn; Medical cần GPT-5-mini làm giám khảo; ⚠ dữ liệu Knowledge Acquisition chưa công khai.
  * Với GPU miễn phí (**Kaggle T4 16 GB**), nhóm chọn:
    1. **E1 — Science Q&A (headline):** Base, SFT, SDFT trên `Qwen2.5-7B-Instruct`.
    2. **E2 — Tool Use (ToolAlpaca):** Base, SFT, SDFT trên `Qwen2.5-7B-Instruct`.
    3. **E3 — Model Size:** `1.5B / 3B / 7B` trên Science Q&A (tái hiện Fig. 5 trái, kiểm chứng giả định ICL).
* **Khối giữa (Dữ liệu):**

| Tác vụ | Train chính thức | Val (tách 5%) | **Train dùng thực tế** | Test |
| :--- | :---: | :---: | :---: | :---: |
| Science | 2.674 | 134 | **352** (~14% phần còn lại) | 507 (đủ) |
| Tool Use | 4.046 | 202 | **528** (~14% phần còn lại) | 97 (đủ) |

  * Kích thước tập con tính từ pilot (19/11): 50.6 s/mẫu Science, 33.4 s/mẫu Tool Use → vừa 2 epoch trong 1 phiên Kaggle ~10 giờ. SFT và SDFT dùng **đúng cùng** tập con.
* **Khối phải (Siêu tham số):**
  * 4-bit NF4 QLoRA, `all-linear`, $r=32$, $\alpha_{\text{LoRA}}=64$, dropout 0.05, compute `fp16`.
  * `lr=1e-4`, cosine, warmup 10 bước, `batch_size=16`, grad-clip 1.0; **SFT 1 epoch, SDFT 2 epoch**.
  * SDFT: `max_new_tokens=1024`, sampling `temperature=1.0`, `ema_alpha=0.01`, Forward KL, `artifact_mask_tokens=3`, sinh 8 rollout cùng lúc.
  * Train trên **1 T4**; đánh giá chia model `fp16` qua **2 T4**.
* **IEEE Citation:** `[1, Sec. 4.4, App. B.1], [20], [21]`
* **Speaker Notes:** Nhóm chọn Core Scope gồm E1 Science Q&A và E2 Tool Use ở quy mô 7B, cộng thêm E3 so sánh 1.5B, 3B, 7B để kiểm chứng hiệu ứng kích thước ở Hình 5 trái. Điều quan trọng nhất cần nói rõ: do giới hạn 12 giờ mỗi phiên Kaggle và hạn nộp, nhóm chỉ train trên 352 mẫu Science và 528 mẫu Tool Use, tức khoảng 14% dữ liệu train. Con số này được tính từ lần chạy thử đo tốc độ thật trên T4. Tập test vẫn giữ đủ. SFT và SDFT luôn dùng chung tập con lấy từ một hoán vị cố định, nên so sánh giữa hai phương pháp vẫn công bằng. Batch 16 được chọn vì nằm trong khoảng sweep của paper và giữ số bước tối ưu không quá ít.

---

## Slide 21 — Quy trình Tái tạo Độc lập & Theo dõi Thực nghiệm
* **Tiêu đề:** QUY TRÌNH TÁI TẠO & THEO DÕI THỰC NGHIỆM (W&B)
* **Khối trên (Liêm chính & tự động hóa):**
  * **Tự cài đặt phần cốt lõi** (`src/sdft_trainer.py`) từ Algorithm 1 và các phương trình của paper; repo `idanshen/Self-Distillation` chỉ đọc để đối chiếu, không sao chép.
  * Dữ liệu chính thức của tác giả được mirror lên Hugging Face (`scripts/mirror_official_data.py`) và tải trong code.
  * Tập con lấy theo hoán vị cố định (`data_seed=0`), dùng chung cho SFT và SDFT; seed huấn luyện `42`.
  * SDFT lưu checkpoint mỗi 10 bước và tự dừng an toàn sau `--max_hours 11` → resume được khi phiên Kaggle bị ngắt.
  * Mọi run (train, eval tác vụ mới, `lm-eval`) đều log lên W&B project `sdft-continual-learning`.
* **Bảng mapping run (điền link W&B sau khi chạy):**

| Job (`scripts/kaggle_jobs.sh`) | Tài khoản | Nội dung | W&B run chính | Trạng thái |
| :--- | :---: | :--- | :--- | :---: |
| `pilot` | A | Đo tốc độ, chọn N_SCI / N_TOOL | `pilot-sdft-7b-science`, `pilot-sdft-7b-tool_use` | ✅ |
| `e1_sdft` → `eval_sdft e1` | A | SDFT 7B Science: train 2 epoch (12.4 h), val 2 epoch, test epoch 2 | `9af4d088` (train), `ka6eu8rn` / `s8n7epps` (val), `4px6nl5i` (test) | ✅ (prior bị huỷ, chạy lại ở `prior_rest`) |
| `night2` / `night3` | B | SFT 7B ×2 tác vụ; E3 3B (Base, Teacher, SFT, SDFT) | `0oxz9j4b` (SFT 3B test), `a0d9b8f7` (SDFT 3B train) | ✅ (SDFT 3B dừng ở bước 41/44) |
| `e2_sdft` | B | SDFT 7B Tool Use | `8a0c196e` | ⚠ CUDA OOM 2 lần (bước 16/33) |
| `final_b` | B | Test 7B: Base ×2, SFT ×2; prior Base 7B | `dkbn0qev`, `brrhsz1p`, `5ovyo04j`, `sv8cpi0m`, `pc471azo` | ✅ |
| `tool_b2` | B | SDFT 7B Tool Use: resume từ bước 10 → epoch 1, test, prior; SDFT 3B test | `8a0c196e` (train, cùng run), `zeqs4yqw` (test) | ✅ 21/11 (chỉ epoch 1) |
| `prior_rest` | B | Prior: SDFT Sci (adapter của A), HumanEval Base, SFT ×2; Teacher 7B | `prior-*`, `teacher-Qwen2.5-7B-Instruct-*` (`l2rv9efr` Tool) | ✅ 21/11 |
| `e3_small` (1.5B) | — | 1.5B Science | — | ❌ không đủ quota |

* **IEEE Citation:** `[22]`
* **Speaker Notes:** Phần cốt lõi của thuật toán được nhóm xây dựng độc lập dựa trên mô tả của bài báo. Dữ liệu chính thức được mirror lên Hugging Face và tải tự động trong code. Toàn bộ job được điều phối qua một script duy nhất và chia cho ba tài khoản Kaggle như trong bảng. Mỗi dòng trong bảng ứng với các run cụ thể trên W&B, nên người chấm có thể đối chiếu từng con số trên slide với run gốc.

---

## Slide 22 — Kết quả Tái tạo: E1 (Science Q&A) & E2 (Tool Use)
* **Tiêu đề:** KẾT QUẢ TÁI TẠO: ĐỘ CHÍNH XÁC TÁC VỤ MỚI (E1 & E2)
* **Bảng đối chiếu** (`Qwen2.5-7B-Instruct`, greedy, tập test đầy đủ; số liệu: `results/`, sinh bằng `scripts/summarize_results.py`):

| Tác vụ / Phương pháp | Paper (%) | Nhóm (%) | 95% CI nhóm (pp) | $\Delta$ (pp) | Sai số tương đối (%) | W&B Run |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Science:** Base 7B | ⚠ 32.1 | **32.5** | ±4.1 | +0.4 | 1.4 | `test-base-7b-science` (`dkbn0qev`) |
| **Science:** SFT (epoch 1) | 66.2 | **36.5** | ±4.2 | −29.7 | 44.9 | `test-sft_science_…_n352_s42-epoch1` (`5ovyo04j`) |
| **Science:** SDFT (epoch 2, chọn qua val) | **70.2** | **41.2** | ±4.3 | −29.0 | 41.3 | `test-sdft_science_…_forward_s42-epoch2` (`4px6nl5i`) |
| **Tool Use:** Base 7B | ⚠ 42.9 | **41.2** | ±9.8 | −1.7 | 3.9 | `test-base-7b-tool` (`brrhsz1p`) |
| **Tool Use:** SFT (epoch 1) | 63.2 | **62.9** | ±9.6 | −0.3 | 0.5 | `test-sft_tool_use_…_n528_s42-epoch1` (`sv8cpi0m`) |
| **Tool Use:** SDFT (epoch 1) | **70.6** | **61.9** | ±9.7 | −8.7 | 12.4 | `test-sdft_tool_use_…_forward_s42-epoch1` (`zeqs4yqw`) |

* **Kiểm định cặp** (cùng 507 / 97 câu hỏi; McNemar chính xác + bootstrap theo cặp 5.000 lần):

| So sánh | $\Delta$ (pp) | 95% CI bootstrap | Chỉ B đúng / chỉ A đúng | McNemar *p* |
| :--- | :---: | :---: | :---: | :---: |
| Science 7B: SDFT − SFT | **+4.7** (paper +4.0) | [+0.2, +9.1] | 79 / 55 | **0.047** |
| Science 7B: SDFT − Base | +8.7 | [+3.6, +13.8] | 108 / 64 | 0.001 |
| Science 7B: SFT − Base | +3.9 | [−1.2, +8.9] | 92 / 72 | 0.138 |
| Tool Use 7B: SFT − Base | +21.6 | [+12.4, +32.0] | 25 / 4 | < 0.001 |
| Tool Use 7B: SDFT − Base | +20.6 | [+12.4, +28.9] | 20 / 0 | < 0.001 |
| Tool Use 7B: SDFT − SFT | −1.0 (paper +7.4) | [−10.3, +9.3] | 11 / 12 | 1.00 |

* **Cách tính:** $\Delta = \text{Nhóm} - \text{Paper}$; sai số tương đối $= |\Delta| / \text{Paper}$; CI một mô hình $\approx \pm 1.96\sqrt{p(1-p)/n}$ ($n = 507$ Science, $n = 97$ Tool Use).
* **Nhận xét:**
  * **Base khớp paper** ở cả hai tác vụ (lệch +0.4 pp và −1.7 pp, nằm trong CI) → prompt, bộ chấm điểm và cách giải mã của nhóm tương đương bản gốc.
  * **Science: thứ tự SDFT > SFT > Base được giữ**, và khoảng cách SDFT − SFT (+4.7 pp) gần đúng con số của paper (+4.0 pp). Khác biệt có ý nghĩa ở mức 5% nhưng sát ngưỡng (*p* = 0.047, 1 seed).
  * **Science: mức tuyệt đối thấp hơn paper ~30 pp** cho cả SFT và SDFT: nhóm chỉ train trên 352 mẫu (~13% dữ liệu) với QLoRA 4-bit → SFT 22 bước, SDFT 44 bước. SFT chưa vượt Base có ý nghĩa (*p* = 0.14).
  * **Tool Use: SFT tái tạo gần như chính xác** (62.9% so với 63.2%).
  * **Tool Use: SDFT (chỉ 1 epoch) = 61.9%, ngang SFT** (−1.0 pp, *p* = 1.00) → **không tái tạo được** mức +7.4 pp của paper. SDFT 7B Tool chỉ kịp 1 epoch (~52 s/mẫu × 528 mẫu ≈ 7.6 h/epoch trên T4), lr schedule vẫn tính cho 2 epoch nên lr chưa giảm hết.
  * Lưu ý giao thức: SFT 1 epoch, SDFT 2 epoch (như paper). Val SDFT: epoch 1 = 39.6%, epoch 2 = 41.8%.
* **IEEE Citation:** `[1, Table 5]`
* **Speaker Notes:** Slide này đặt kết quả của nhóm cạnh Table 5. Trước hết, mô hình Base cho kết quả gần như trùng với paper ở cả hai tác vụ: 32.5% so với 32.1% trên Science và 41.2% so với 42.9% trên Tool Use. Điều này cho thấy prompt, bộ chấm và cách giải mã của nhóm tương đương bản gốc, nên các khác biệt còn lại đến từ khâu huấn luyện. Trên Science, cả SFT và SDFT đều thấp hơn paper khoảng 30 điểm, vì nhóm chỉ train trên khoảng 13% dữ liệu với QLoRA 4-bit. Tuy vậy, thứ tự giữa các phương pháp được giữ: SDFT hơn SFT 4.7 điểm, rất gần con số 4.0 điểm của paper. Vì hai mô hình trả lời cùng 507 câu hỏi, nhóm dùng kiểm định cặp McNemar và bootstrap theo cặp: chênh lệch có ý nghĩa ở mức 5% nhưng sát ngưỡng, và chỉ có một seed, nên nhóm coi đây là bằng chứng ủng hộ chứ chưa phải kết luận chắc chắn. Trên Tool Use, SFT đạt 62.9%, gần như đúng 63.2% của paper.

---

## Slide 23 — Kết quả Tái tạo: Khả năng giữ Kiến thức cũ (6 Benchmarks)
* **Tiêu đề:** KẾT QUẢ TÁI TẠO: ĐO CATASTROPHIC FORGETTING (6 BENCHMARKS)
* **Bảng `lm-evaluation-harness`** (mỗi ô: Paper → Nhóm; `--limit 0.25`, `--apply_chat_template`):

| Phương pháp | HellaSwag | HumanEval | IFEval | MMLU | TruthfulQA | Winogrande | Prior Avg |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Base 7B | 62.0 → **52.9** | 65.8 → **90.2** | 74.3 → **68.4** | 71.7 → **69.1** | 47.9 → **59.8** | 71.1 → **63.4** | 65.5 → **67.3** |
| SFT (Science) | 55.0 → **56.1** | 54.8 → **92.7** | 35.3 → **66.2** | 64.6 → **69.3** | 36.8 → **59.7** | 73.7 → **64.7** | 53.4 → **68.1** |
| SDFT (Science) | 60.9 → **55.8** | 68.9 → **92.7** | 66.8 → **71.3** | 70.7 → **68.7** | 46.5 → **60.7** | 73.1 → **65.0** | 64.5 → **69.0** |
| SFT (Tool Use) | — → **63.2** | — → **90.2** | — → **61.8** | — → **72.1** | — → **57.6** | — → **71.9** | 56.0 → **69.5** |
| SDFT (Tool Use, epoch 1) | — → **57.5** | — → **95.1** | — → **71.3** | — → **69.5** | — → **60.9** | — → **65.9** | 65.4 → **70.0** |

  *(— : số từng benchmark của hàng Tool Use chưa được chép từ Table 5; chỉ dùng Prior Avg. HumanEval của nhóm là `humaneval_instruct` nên không so trực tiếp được với cột paper.)*
* **Ghi chú thực nghiệm:**
  * Metric: HellaSwag `acc_norm`, HumanEval `pass@1`, IFEval `prompt_level_strict_acc`, MMLU `acc`, TruthfulQA `truthfulqa_mc2`, Winogrande `acc` (⚠ đối chiếu với metric paper dùng). W&B: `prior-base-7b` (`pc471azo`).
  * Số câu sau `--limit 0.25`: HellaSwag 2.511, MMLU 3.529, Winogrande 317, TruthfulQA 205, IFEval 136, HumanEval 41 → CI khoảng ±2 pp (HellaSwag) đến ±14 pp (HumanEval).
  * **Sự cố HumanEval:** task `humaneval` gốc là dạng nối tiếp prompt, khi bật chat template thì câu trả lời không ghép được thành hàm → **0/41**. Nhóm chuyển sang `humaneval_instruct` (biến thể cho chat model) và chạy lại.
  * Base: MMLU, IFEval sát paper (−2.6, −5.9 pp); HellaSwag, Winogrande thấp hơn và TruthfulQA cao hơn paper → khác biệt cấu hình harness (chat template trên các task log-likelihood, tập con 25%), không phải do huấn luyện. Vì vậy **so sánh trong nội bộ nhóm (Base → SFT → SDFT, cùng cấu hình) mới là phép đo mức quên**.
  * **Kết quả: không tái tạo được hiện tượng quên của SFT.** Trong cùng cấu hình, Prior Avg: Base 67.3, SFT Sci 68.1, SDFT Sci 69.0, SFT Tool 69.5, SDFT Tool 70.0 — mọi mô hình nằm trong ±3 pp của Base, không có cú sụt như paper (SFT 65.5 → 53.4). IFEval của SFT Science 66.2 (Base 68.4) chứ không rơi xuống 35.3 như paper.
  * Giải thích khả dĩ (chưa kiểm chứng bằng thí nghiệm riêng): QLoRA r=32 giữ nguyên trọng số gốc và nhóm chỉ train 22–66 bước trên ~14% dữ liệu → cập nhật quá nhỏ để gây quên. Vì vậy **thí nghiệm này không phân biệt được SFT và SDFT về mặt chống quên**; SDFT chỉ nhỉnh hơn ~1 pp Avg, nằm trong nhiễu của `--limit 0.25`.
* **IEEE Citation:** `[1, Table 5], [17]`
* **Speaker Notes:** Để đo mức quên tương ứng với Table 5, nhóm chạy lm-evaluation-harness trên đủ sáu benchmark, mỗi bộ lấy 25% số câu để vừa quỹ thời gian T4. Với mô hình Base, MMLU và IFEval gần với paper, còn HellaSwag, Winogrande và TruthfulQA lệch theo hai chiều khác nhau, nên nguyên nhân là cấu hình đánh giá chứ không phải mô hình. Vì vậy nhóm chỉ đọc mức quên bằng cách so các mô hình của chính nhóm với nhau dưới cùng một cấu hình. Một sự cố đáng chú ý: task HumanEval dạng gốc cho điểm 0 khi bật chat template, vì câu trả lời dạng hội thoại không nối được vào phần đầu hàm; nhóm chuyển sang biến thể humaneval_instruct và chạy lại.

---

## Slide 24 — Phân tích bổ sung (E3): Hiệu ứng Kích thước Mô hình
* **Tiêu đề:** PHÂN TÍCH BỔ SUNG (E3): ẢNH HƯỞNG KÍCH THƯỚC MÔ HÌNH
* **Bảng E3 trên Science Q&A** (đối chiếu Fig. 5 trái):

| Model | Base (%, test) | Teacher `--with_demo` (%, val) | SFT (%, test) | SDFT (%, test) | SDFT − SFT (pp) | Paper (pp) | W&B Run |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Qwen2.5-1.5B` | — | — | — | — | — | — | không chạy (hết quota) |
| `Qwen2.5-3B` | 27.8 | 89.6 | 34.7 | 37.3 (epoch 1) | +2.6 (*p* = 0.35) | chỉ có trên hình (Fig. 5), không ghi số | `test-sft_science_Qwen2.5-3B-…-epoch1` (`0oxz9j4b`), SDFT train `a0d9b8f7` |
| `Qwen2.5-7B` (từ E1) | 32.5 | 92.5 | 36.5 | 41.2 | **+4.7** (*p* = 0.047) | **+4.0** | xem Slide 22 |

* **Ghi chú giao thức:**
  * 3B: SFT epoch 1; SDFT dừng ở bước 41/44 do giới hạn thời gian nên **đánh giá epoch 1** (không có epoch 2). 7B: SDFT epoch 2 chọn qua val.
  * Cột Teacher đo trên **tập val** (134 câu, có lời giải mẫu trong ngữ cảnh), các cột khác trên **tập test** (507 câu).
  * 3B: SFT hơn Base +6.9 pp (CI bootstrap [+2.0, +11.8], McNemar *p* = 0.007); SDFT hơn Base +9.5 pp (*p* < 0.001); SDFT − SFT = +2.6 pp, CI [−2.4, +7.5], *p* = 0.35 → **không khác biệt có ý nghĩa**.
* **Quan sát:** Teacher có lời giải mẫu đạt 89.6% (3B) và 92.5% (7B) trên val → giả định in-context đứng vững ở cả hai cỡ trong setup này. Nhóm **không thấy SDFT thua SFT ở 3B** như Fig. 5 của paper (+2.6 pp, không có ý nghĩa); với 1 seed và SDFT 3B chỉ 1 epoch, nhóm không kết luận được về chiều hướng ở 3B.
* **Mục tiêu:** kiểm chứng giả định ICL (Sec. 3.2, 4.4): với mô hình nhỏ, Teacher dù có mẫu trong ngữ cảnh vẫn yếu nên SDFT không thắng SFT; từ 7B mới chuyển sang dương.
* **IEEE Citation:** `[1, Sec. 4.4, Fig. 5 (trái)]`
* **Speaker Notes:** Nhóm chạy thêm E3 để so sánh SFT và SDFT theo kích thước mô hình, đồng thời đo độ chính xác của Teacher, tức mô hình được xem lời giải mẫu trong prompt. Do hết quota, nhóm chỉ chạy được 3B và 7B. Ở 7B, SDFT hơn SFT 4.7 điểm, khớp xu hướng của paper. Ở 3B, Teacher đạt gần 90% trên tập val, nên tín hiệu từ giáo viên không yếu về mặt độ chính xác; nếu SDFT 3B vẫn không hơn SFT thì nguyên nhân nằm ở khả năng học trò hấp thụ tín hiệu, không phải ở chất lượng giáo viên. Lưu ý SDFT 3B chỉ có epoch 1 vì phiên Kaggle hết giờ ở bước 41/44.

---

## Slide 25 — Phân tích Sai số (1/2): Phần cứng, QLoRA & Dữ liệu
* **Tiêu đề:** PHÂN TÍCH SAI SỐ (DEVIATION ANALYSIS — 1/2)
* **Khối trái (1. QLoRA thay Full Fine-Tuning):**
  * **Paper:** Full FT `bfloat16` trên H200. **Nhóm:** 4-bit NF4 QLoRA (`all-linear`, $r=32$), compute `fp16` trên T4.
  * **Tác động:**
    * (a) LoRA giữ trọng số gốc → SFT có thể quên ít hơn Full FT.
    * (b) **EMA áp lên từng ma trận LoRA $A$, $B$ riêng rẽ**: $\text{EMA}(B)\cdot\text{EMA}(A) \neq \text{EMA}(BA)$ → chỉ xấp xỉ EMA của $\Delta W$.
    * (c) **Train trên base NF4, đánh giá trên base `fp16`**: adapter học bù cho sai số lượng tử hóa, khi gắn lên base `fp16` có thể lệch nhẹ.
* **Khối phải (2. Tập con huấn luyện):**
  * **Paper:** đủ 2.674 / 4.046 mẫu. **Nhóm:** **352 / 528 mẫu** (~14% sau khi tách val).
  * Batch 16 → SFT ~22 bước Science / ~33 bước Tool Use; SDFT ~44 / ~66 bước.
  * **Tác động:** mọi phương pháp đều học ít hơn → độ chính xác tác vụ mới có thể thấp hơn paper; SFT bị ảnh hưởng ít hơn về mức quên vì ít bước cập nhật.
* **IEEE Citation:** `[1, App. B.1, Table 3], [20], [21]`
* **Speaker Notes:** Phần này giải trình các nguyên nhân chính gây lệch so với Table 5. Thứ nhất là QLoRA thay cho full fine-tuning, với ba hệ quả: SFT quên ít hơn; EMA cập nhật riêng hai ma trận A và B của LoRA, mà tích của hai trung bình không bằng trung bình của tích, nên Teacher chỉ là xấp xỉ của EMA trọng số; và adapter được train trên base 4-bit nhưng đánh giá trên base 16-bit. Thứ hai — và có lẽ quan trọng nhất — là nhóm chỉ dùng khoảng 14% dữ liệu train do giới hạn thời gian, nên số bước tối ưu ít hơn paper nhiều lần.

---

## Slide 26 — Phân tích Sai số (2/2): Triển khai, Chi phí & Nhiễu thống kê
* **Tiêu đề:** PHÂN TÍCH SAI SỐ (DEVIATION ANALYSIS — 2/2)
* **Khối trái (3. Chi tiết triển khai):**
  * Số token bị mask ($k$) không có trong paper → nhóm dùng `3` theo code gốc.
  * Paper ghi `max_gen=2048`, code gốc dùng `1024` → nhóm dùng `1024` (demo Science có p90 ~480 token; theo dõi tỉ lệ bị cắt trên W&B).
  * Code gốc sinh rollout bằng `vLLM` kèm importance sampling; nhóm dùng `transformers.generate()` theo batch trên **đúng trọng số student hiện tại** → on-policy chính xác, không cần IS.
* **Khối giữa (4. Chi phí thực tế):**
  * Paper: SDFT/SFT wall-clock $\approx 4\times$ (H200 + vLLM).
  * Nhóm (T4, HF generate): SDFT 7B Science **63.4 s/mẫu** (12.4 h cho 44 bước × 16 mẫu; pilot dự báo 50.6), SDFT 7B Tool Use ~52 s/mẫu (≈14 phút/bước, một nửa là sinh rollout); SFT `[...]` s/mẫu (W&B `time/step_sec`) → tỉ lệ **`[...]×`** (plan dự báo ~11×).
  * Bộ nhớ: SDFT 7B Tool Use bị **CUDA OOM 2 lần** trên T4 16 GB (prompt ToolAlpaca dài): lần 1 ở khâu KL theo từ vựng (sửa bằng KL chia khúc 128 token + `logits_to_keep`), lần 2 ở khâu sinh 8 rollout cùng lúc (sửa: tự chia đôi batch khi OOM, chạy tiếp với nhóm 4 rollout).
* **Khối phải (5. Seed & nhiễu thống kê):**
  * Paper: ⚠ sweep 27 cấu hình mỗi phương pháp, trung bình **3 seeds** + CI 95%.
  * Nhóm: 1 cấu hình, `seed=42` → không có phương sai giữa các seed.
  * Tập test nhỏ: CI ≈ ±4 pp (Science, 507 câu), ±9 pp (Tool Use, 97 câu) — cùng cỡ với hiệu số SDFT − SFT của paper.
* **IEEE Citation:** `[1, Sec. 5, App. B.2], [13]`
* **Speaker Notes:** Ba nguồn sai số tiếp theo. Một, chi tiết triển khai: số token bị mask không có trong bài báo, độ dài sinh tối đa giữa paper và code không khớp, và engine sinh chuỗi khác nhau. Một điểm cộng là nhóm sinh rollout trực tiếp từ đúng trọng số student đang học, nên không cần hiệu chỉnh importance sampling. Hai, chi phí: trên T4 không có vLLM, bước sinh chiếm phần lớn thời gian, nên SDFT chậm hơn SFT nhiều hơn so với con số bốn lần trên H200. Ba, nhiễu thống kê: nhóm chỉ chạy một seed, và tập test Tool Use chỉ có 97 câu, nên khoảng tin cậy rộng bằng chính hiệu số mà paper báo cáo.

---

## Slide 27 — Đối chiếu Paper vs Code gốc vs Bản tái tạo
* **Tiêu đề:** ĐỐI CHIẾU PAPER VS. CODE GỐC VS. BẢN TÁI TẠO CỦA NHÓM
* **Bảng 6 phát hiện:**

| Khía cạnh | Paper | Repo gốc (`idanshen/Self-Distillation`) | Bản của nhóm |
| :--- | :--- | :--- | :--- |
| **Hướng KL** | Reverse KL (Eq. 1–2) | README: mọi kết quả dùng **Forward KL** | Có cả `forward` (mặc định) và `reverse` |
| **Max generation** | `2048` tokens (Table 3) | `1024` tokens | `1024` tokens |
| **Learning rate** | Sweep `{5e-6, 1e-5, 5e-5}` | Mặc định `2e-5`, README dùng `5e-5` | `1e-4` (LoRA cần LR cao hơn Full FT) |
| **Warmup** | `10` bước | `10%` tổng số bước | `10` bước |
| **Rollout & IS** | Alg. 1 chỉ ghi IS "if needed" | `vLLM` + importance sampling | HF `generate()` theo batch, on-policy chính xác (không cần IS) |
| **Prompt Science** | Template 3 dòng (Sec. 3) | Thêm **system prompt** yêu cầu `<reasoning>` / `<answer>` | Dùng system prompt giống code gốc cho cả Student và Teacher |

* **IEEE Citation:** `[1], [13]` *(repo gốc chỉ dùng để đối chiếu)*
* **Speaker Notes:** Khi đọc đối chiếu repo chính thức — chỉ để xác minh, không sao chép — nhóm thấy sáu điểm không khớp giữa bài báo và code. Nổi bật nhất là code dùng Forward KL trong khi lý thuyết viết Reverse KL. Ngoài ra, độ dài sinh là 1024 thay vì 2048, warmup tính theo 10% số bước, rollout dùng vLLM kèm importance sampling, và Science Q&A có thêm system prompt ép định dạng reasoning/answer. Nhóm theo code gốc ở những điểm ảnh hưởng trực tiếp tới việc so sánh với Table 5.

---

## Slide 28 — Tóm tắt Điểm mạnh
* **Tiêu đề:** TỔNG KẾT CÁC ĐIỂM MẠNH CỦA SDFT
* **3 điểm mạnh (kèm bằng chứng):**
  1. **Không cần reward model:** biến expert demonstrations thành tín hiệu học on-policy thông qua In-Context Learning (Sec. 3).
  2. **Quên ít hơn hẳn SFT:** Prior Avg gần bằng Base (Table 5) và học nối tiếp nhiều tác vụ (Fig. 3). Bản tái tạo: `[điền: xu hướng có giữ được không]`.
  3. **Tổng quát hóa OOD & giữ reasoning:** 98% OOD ở Knowledge Acquisition (Table 1); cải thiện reasoning model khi chỉ có đáp án cuối (Table 2).
* **IEEE Citation:** `[1, Sec. 3–5]`
* **Speaker Notes:** Tổng kết phần đọc hiểu và tái tạo, SDFT có ba ưu điểm được hậu thuẫn bởi cả lý thuyết lẫn thực nghiệm: không phụ thuộc vào hàm thưởng tường minh; giữ năng lực nền tảng gần như nguyên vẹn khi học kỹ năng mới; và giúp mô hình dùng được tri thức mới để trả lời các câu hỏi gián tiếp ngoài phân phối. Ở ý thứ hai, nhóm bổ sung nhận định từ chính kết quả tái tạo.

---

## Slide 29 — Hướng Phê bình Lý thuyết cho Poster A2 (Checkpoint 2)
* **Tiêu đề:** HƯỚNG PHÊ BÌNH LÝ THUYẾT CHO POSTER CUỐI KỲ
* **3 điểm yếu nhóm sẽ kiểm chứng ở Checkpoint 2:**
  1. **Giả định In-Context (Eq. 4) mong manh:** $\pi^* \approx \pi(\cdot|x,c)$ không còn đúng khi mô hình nhỏ (3B) hoặc khi demonstration $c$ bị nhiễu / lệch định dạng.
  2. **Khoảng cách lý thuyết – thực thi:**
     * Sự tương đương IRL ở Eq. 6 **chỉ đúng với Reverse KL**; kết quả công bố lại dùng **Forward KL** (mode-covering) — chứng minh không còn bảo đảm cho thuật toán thực sự được chạy.
     * Ước lượng KL theo từng token bỏ qua ảnh hưởng của $y_t$ lên các token sau $y_{>t}$ → gradient lệch so với KL ở mức cả chuỗi.
  3. **Chi phí rollout & mẹo xử lý artifact:** ~4× thời gian so với SFT trên H200 (`[...]×` trên T4), và cần mask token đầu để che artifact từ ngữ cảnh của Teacher.
* **IEEE Citation:** `[1, Sec. 3.1, 4.4, 5, App. A.1], [12], [13]`
* **Speaker Notes:** Từ việc đọc sâu và tự cài đặt, nhóm xác định ba điểm yếu để làm trọng tâm cho phần phê bình và thí nghiệm trên Poster A2. Thứ nhất là giả định ICL: với mô hình nhỏ hoặc demonstration kém, Teacher không còn gần với chính sách tối ưu. Thứ hai, và rõ ràng nhất, là chứng minh IRL chỉ đúng cho Reverse KL, trong khi mọi kết quả công bố dùng Forward KL; ngoài ra ước lượng theo từng token là một xấp xỉ có độ chệch. Nhóm đã cài cả hai hướng KL nên có thể so sánh trực tiếp. Thứ ba là chi phí wall-clock thực tế và sự phụ thuộc vào mẹo mask token.

---

## Slide 30 — Kết luận & Tài nguyên Công khai
* **Tiêu đề:** KẾT LUẬN & TÀI NGUYÊN DỰ ÁN
* **Tóm tắt:**
  * Đọc hiểu đầy đủ 5 lớp phương pháp của SDFT [1].
  * Tự cài đặt SDFT bằng PyTorch + QLoRA, chạy trên GPU miễn phí (Kaggle T4).
  * Báo cáo định lượng độ lệch ($\Delta$, sai số tương đối, CI 95%) và giải trình các nguồn sai số.
* **Liên kết kiểm chứng:**
  * **GitHub:** `https://github.com/24122040-kin/distillation_enables_continual_learning_hcmus`
  * **Weights & Biases:** `https://wandb.ai/baokin91ltri-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/sdft-continual-learning` (⚠ đổi project sang Public trước khi nộp)
  * **Số liệu gốc:** thư mục `results/` (JSON từng câu trả lời) + `python scripts/summarize_results.py results/*` để sinh lại mọi bảng.
* **IEEE Citation:** `[1], [22]`
* **Speaker Notes:** Xin cảm ơn thầy và hội đồng. Toàn bộ mã nguồn tự cài đặt, bảng ánh xạ thí nghiệm và các đường dẫn Weights & Biases đều công khai tại kho GitHub của nhóm. Nhóm mong nhận góp ý để hoàn thiện phần Critical Analysis và Empirical Demonstration cho buổi Poster Workshop cuối kỳ.

---

## Slide 31 — References
* **Tiêu đề:** TÀI LIỆU THAM KHẢO
* **Nội dung (IEEE, theo thứ tự xuất hiện):**

[1] I. Shenfeld, M. Damani, J. Hübotter, and P. Agrawal, "Self-distillation enables continual learning," arXiv:2601.19897v2, 2026.

[2] M. De Lange *et al.*, "A continual learning survey: Defying forgetting in classification tasks," *IEEE Trans. Pattern Anal. Mach. Intell.*, vol. 44, no. 7, pp. 3366–3385, Jul. 2022.

[3] J. Kirkpatrick *et al.*, "Overcoming catastrophic forgetting in neural networks," *Proc. Natl. Acad. Sci. USA*, vol. 114, no. 13, pp. 3521–3526, 2017.

[4] H. Chen *et al.*, "Retaining by doing: The role of on-policy data in mitigating forgetting," arXiv:2510.18874, 2025.

[5] I. Shenfeld, J. Pari, and P. Agrawal, "RL's razor: Why online reinforcement learning forgets less," arXiv:2509.04259, 2025.

[6] S. Ross, G. Gordon, and D. Bagnell, "A reduction of imitation learning and structured prediction to no-regret online learning," in *Proc. AISTATS*, 2011, pp. 627–635.

[7] T. Chu *et al.*, "SFT memorizes, RL generalizes: A comparative study of foundation model post-training," arXiv:2501.17161, 2025.

[8] A. Y. Ng and S. Russell, "Algorithms for inverse reinforcement learning," in *Proc. ICML*, 2000, pp. 663–670.

[9] C. Snell, D. Klein, and R. Zhong, "Learning by distilling context," arXiv:2209.15189, 2022.

[10] R. Agarwal *et al.*, "On-policy distillation of language models: Learning from self-generated mistakes," in *Proc. ICLR*, 2024.

[11] T. Brown *et al.*, "Language models are few-shot learners," in *Proc. NeurIPS*, 2020, pp. 1877–1901.

[12] Y. Tang and R. Munos, "On a few pitfalls in KL divergence gradient estimation for RL," arXiv:2506.09477, 2025.

[13] I. Shenfeld, "Self-Distillation" (official code). GitHub. [Online]. Available: https://github.com/idanshen/Self-Distillation

[14] J. Schulman, S. Levine, P. Abbeel, M. Jordan, and P. Moritz, "Trust region policy optimization," in *Proc. ICML*, 2015, pp. 1889–1897.

[15] K. Feng *et al.*, "SciKnowEval: Evaluating multi-level scientific knowledge of large language models," arXiv:2406.09098, 2024.

[16] Q. Tang *et al.*, "ToolAlpaca: Generalized tool learning for language models with 3000 simulated cases," arXiv:2306.05301, 2023.

[17] L. Gao *et al.*, "The language model evaluation harness," Zenodo, 2024.

[18] A. Yang *et al.* (Qwen Team), "Qwen2.5 technical report," arXiv:2412.15115, 2024.

[19] J. Chen *et al.*, "HuatuoGPT-o1, towards medical complex reasoning with LLMs," arXiv:2412.18925, 2024.

[20] E. J. Hu *et al.*, "LoRA: Low-rank adaptation of large language models," in *Proc. ICLR*, 2022.

[21] T. Dettmers, A. Pagnoni, A. Holtzman, and L. Zettlemoyer, "QLoRA: Efficient finetuning of quantized LLMs," in *Proc. NeurIPS*, 2023.

[22] Nhóm thực hiện, "distillation_enables_continual_learning_hcmus." GitHub. [Online]. Available: https://github.com/24122040-kin/distillation_enables_continual_learning_hcmus

* **Speaker Notes:** (Không trình bày; slide để tra cứu.)
