# Nội dung chi tiết 30 Slides (Tiếng Việt — Tự đứng độc lập) — SDFT Term Project (HCMUS)

> **Quy ước soạn thảo cho Canva & PowerPoint:**
> - **100% tự đứng độc lập:** Mọi ký hiệu, công thức, baseline, tập dữ liệu và chỉ số đánh giá đều được giải thích đầy đủ ngay trong slide và Speaker Notes để giám khảo chưa đọc bài báo gốc vẫn hiểu trọn vẹn.
> - **Ký hiệu Toán sạch (Unicode):** Không dùng mã LaTeX thô có dấu gạch chéo ngược để copy-paste thẳng vào Canva không bị lỗi hiển thị.
> - **Trích dẫn IEEE:** Đánh số thống nhất `[1]`–`[20]` đặt gọn ở chân mỗi slide trích dẫn (không ghi dòng chữ thừa `IEEE Citation:`).
> - **Ô `[...]` / `[Link W&B]`:** Chỗ chừa trống để điền kết quả chạy thực tế từ Kaggle & W&B.
> - **3 hình tự vẽ sẵn để upload lên Canva:** `presentation/assets/pipeline_sdft.png` (Slide 9), `presentation/assets/pareto_paper_vs_repro.png` (Slide 17), `presentation/assets/sequential_cl_concept.png` (Slide 18).

---

## Slide 1 — Trang bìa (Cover)
* **Tiêu đề chính:** SELF-DISTILLATION ENABLES CONTINUAL LEARNING (SDFT)
* **Phụ đề:** Đọc hiểu Bài báo & Tái tạo Thực nghiệm (Core Scope) — Môn Nhập môn Học máy (HCMUS)
* **Thông tin nhóm:**
  * Nhóm 4 — Thành viên: Trần Trung Hiếu (24122033), Cái Hoàng Bảo Kin (24122040)
  * Bài báo gốc: I. Shenfeld, M. Damani, J. Hübotter, P. Agrawal (MIT & ETH Zurich, 2026)
  * Public GitHub: `https://github.com/24122040-kin/distillation_enables_continual_learning_hcmus`
  * Public W&B Project: `[Dán link W&B Project trên CoreWeave Forge]`
* [1] I. Shenfeld, M. Damani, J. Hübotter, and P. Agrawal, "Self-Distillation Enables Continual Learning," arXiv:2601.19897v2, 2026.
* **Speaker Notes:** Bộ slide này trình bày toàn bộ kết quả đọc hiểu độc lập và tái tạo thực nghiệm theo phạm vi Core Scope đối với công trình "Self-Distillation Enables Continual Learning" của nhóm tác giả Idan Shenfeld, Mehul Damani, Jonas Hübotter và Pulkit Agrawal (MIT và ETH Zurich, công bố năm 2026). Bài toán trung tâm mà công trình giải quyết là làm sao giúp một mô hình ngôn ngữ lớn (LLM) liên tục tiếp thu kỹ năng và tri thức mới từ dữ liệu mẫu của chuyên gia (expert demonstrations) mà không bị quên mất các năng lực nền tảng đã có sẵn. Để đáp ứng yêu cầu liêm chính học thuật của môn học, toàn bộ thuật toán cốt lõi của phương pháp Self-Distillation Fine-Tuning (SDFT) được nhóm tự cài đặt lại từ mô tả toán học trong bài báo, mọi sơ đồ và biểu đồ đều do nhóm tự vẽ lại từ dữ liệu gốc, và toàn bộ các lần chạy thực nghiệm đều được ghi nhận công khai theo từng bước trên nền tảng Weights & Biases gắn trực tiếp với kho mã nguồn GitHub của nhóm. Do bài báo gốc sử dụng phần cứng H200 rất lớn và một số tác vụ cần giám khảo trả phí, nhóm lựa chọn phạm vi Core Scope với ba thực nghiệm trọng tâm và giải trình đầy đủ các điều chỉnh phần cứng trong phần sau của deck.

---

## Slide 2 — Mục lục (Table of Contents)
* **Tiêu đề:** NỘI DUNG TRÌNH BÀY
* **Nội dung (7 phần — 30 Slides):**
  * `01` Bối cảnh & Động lực nghiên cứu (Slides 3–4)
  * `02` Các phương pháp tiền nhiệm & Khoảng trống (Slides 5–6)
  * `03` Đóng góp chính & Tính mới của bài báo (Slide 7)
  * `04` Giải phẫu phương pháp SDFT qua 5 lớp (Slides 8–15)
  * `05` Giao thức & Kết quả công bố trong bài báo (Slides 16–19)
  * `06` Tái tạo Thực nghiệm (Core Scope) & Phân tích Sai số (Slides 20–26)
  * `07` Đối chiếu Code gốc, Điểm mạnh & Kết luận (Slides 27–30)
* **Speaker Notes:** Bài trình bày được tổ chức thành 30 slides chia theo đúng tiến trình từ đọc hiểu khái niệm đến thực nghiệm kiểm chứng. Hai slide đầu của phần nội dung (Slides 3–4) đặt ra vấn đề học liên tục trong thực tế và chỉ ra khoảng trống giữa học tăng cường theo chính sách hiện tại (on-policy RL) với tinh chỉnh có giám sát (SFT). Tiếp theo, Slides 5–6 phân tích kỹ hai nhóm hướng tiếp cận tiền nhiệm gồm SFT, Inverse RL và Offline Context Distillation cùng các hệ quả thực tế của chúng. Slide 7 tổng hợp và phân loại các đóng góp của bài báo theo ba nhóm lý thuyết, kỹ thuật và thực nghiệm. Trọng tâm kỹ thuật nằm ở Slides 8–15, nơi phương pháp SDFT được trình bày lần lượt qua năm lớp: ký hiệu và bài toán hình thức, luồng xử lý tổng thể, ba thành phần cấu thành theo cấu trúc Làm gì / Vì sao / Như thế nào, tính chất lý thuyết liên hệ với Inverse RL, và phân tích độ phức tạp tính toán. Slides 16–19 tóm tắt dữ liệu, chỉ số và các kết quả chính mà tác giả công bố. Cuối cùng, Slides 20–30 trình bày thiết kế Core Scope của nhóm, kết quả tái tạo trên Kaggle GPU T4 miễn phí, phân tích sai số, đối chiếu với mã nguồn chính thức và cầu nối sang giai đoạn phản biện Poster.

---

## Slide 3 — Bối cảnh & Ví dụ thực tế
* **Tiêu đề:** BỐI CẢNH: VÌ SAO LLM TĨNH THẤT BẠI SAU TRIỂN KHAI?
* **Khối trái — Vấn đề thực tế & Ví dụ của nhóm:**
  * Các Foundation Models hiện đại thường bị đóng băng trọng số sau khi triển khai (static post-deployment).
  * Prompting hoặc RAG chỉ giúp thích nghi tạm thời lúc suy luận, không nội hóa được kỹ năng mới vào tham số mô hình.
  * **Ví dụ thực tế:** Một trợ lý AI hỗ trợ bác sĩ tại bệnh viện cần cập nhật liên tục phác đồ điều trị mới, nhưng sau khi cập nhật lại mất khả năng viết mã phân tích dữ liệu, suy luận logic hoặc tuân thủ định dạng chỉ dẫn cơ bản.
* **Khối phải — Nghịch lý Catastrophic Forgetting:**
  * Cập nhật trọng số cho tác vụ mới thường gây **quên thảm khốc (catastrophic forgetting)** — năng lực cũ tụt dốc nhanh chóng.
  * **Bài toán:** *Làm sao cập nhật tham số để LLM học liên tục từ ví dụ mẫu (expert demonstrations) mà không đánh đổi năng lực sẵn có?*
* [1] I. Shenfeld et al., "Self-Distillation Enables Continual Learning," arXiv:2601.19897v2, Sec. 1, 2026.
* [7] J. Kirkpatrick et al., "Overcoming catastrophic forgetting in neural networks," PNAS, vol. 114, no. 13, pp. 3521–3526, 2017.
* **Speaker Notes:** Trong thực tế ứng dụng, hầu hết các mô hình ngôn ngữ lớn sau khi phát hành đều ở trạng thái tĩnh. Dù người dùng có thể bổ sung ngữ cảnh thông qua truy xuất tài liệu (RAG) hay thiết kế câu lệnh (prompting), mô hình vẫn không thực sự học thêm kỹ năng mới hay ghi nhớ tri thức lâu dài vào bộ trọng số của mình. Tuy nhiên, nếu ta đem mô hình đi huấn luyện tiếp (fine-tune) trên từng đợt dữ liệu mới, hiện tượng quên thảm khốc — tức việc tối ưu cho tác vụ mới phá hủy các biểu diễn đã học cho tác vụ cũ — sẽ xảy ra nghiêm trọng. Ví dụ cụ thể là một hệ thống trợ lý y khoa tại bệnh viện: khi được huấn luyện bổ sung trên tập ca bệnh lâm sàng mới, mô hình dễ bị suy giảm mạnh khả năng tuân thủ chỉ dẫn, lập trình hoặc hỏi đáp kiến thức phổ thông. Do đó, thách thức cốt lõi của Continual Learning là tìm ra cơ chế cập nhật trọng số giúp mô hình tích lũy nhiều kỹ năng nối tiếp theo thời gian mà không làm thoái lui hiệu năng trên những gì đã biết.

---

## Slide 4 — Khoảng trống giữa SFT và On-policy RL
* **Tiêu đề:** KHOẢNG TRỐNG GIỮA SFT VÀ ON-POLICY RL
* **Khối trái — On-policy RL (ví dụ PPO, GRPO):**
  * **Cơ chế:** Mô hình tự sinh câu trả lời từ chính sách hiện tại (`on-policy`), rồi nhận điểm thưởng để cập nhật.
  * **Ưu điểm:** Giảm quên kiến thức cũ rất tốt (chỉ thay đổi phân phối ở nơi cần thiết) và tổng quát hóa cao.
  * **Hạn chế:** Bắt buộc phải có **hàm thưởng tường minh (explicit reward function)** — thứ cực kỳ khó viết cho các tác vụ mở như y khoa hay tri thức mới.
* **Khối phải — Supervised Fine-Tuning (SFT):**
  * **Cơ chế:** Bắt chước trực tiếp các cặp câu hỏi – đáp án mẫu (`expert demonstrations`), không cần hàm thưởng.
  * **Hạn chế:** Hoàn toàn **off-policy** (chỉ học trên văn phong và quỹ đạo cố định của chuyên gia).
  * **Hệ quả:** Dễ học vẹt bề mặt và gây quên thảm khốc trên năng lực tổng quát.
* **Mục tiêu SDFT:** Đưa lợi ích **on-policy** vào bối cảnh chỉ có **expert demonstrations** (không cần reward function).
* [1] I. Shenfeld et al., "Self-Distillation Enables Continual Learning," arXiv:2601.19897v2, Sec. 1, 2026.
* [3] I. Shenfeld, J. Pari, and P. Agrawal, "RL's Razor: Why online reinforcement learning forgets less," arXiv:2509.04259, 2025.
* **Speaker Notes:** Một chuỗi công trình gần đây, tiêu biểu là RL's Razor, đã chỉ ra một quy luật quan trọng: các thuật toán học on-policy — tức mô hình tự sinh ra dữ liệu huấn luyện theo chính phân phối xác suất hiện tại của nó — có xu hướng giữ lại các năng lực cũ tốt hơn nhiều so với học off-policy trên dữ liệu tĩnh. Lý do trực quan là học on-policy tìm nghiệm gần phân phối gốc nhất trong không gian chính sách thỏa mãn yêu cầu tác vụ. Tuy vậy, thành công của hướng on-policy đến nay chủ yếu giới hạn trong Reinforcement Learning (RL), nơi ta có thể định nghĩa hàm thưởng chính xác như giải toán đúng/sai hay chạy unit test cho code. Ở phần lớn bài toán thực tế như tư vấn lâm sàng, gọi API phức tạp hay tiếp thu kiến thức mới, ta không có hàm thưởng tự động mà chỉ có tập dữ liệu minh họa từ chuyên gia. Ngược lại, SFT dùng được dữ liệu minh họa nhưng lại mang bản chất off-policy. Vì thế, câu hỏi nghiên cứu trung tâm đặt ra là làm cách nào chiết xuất được tín hiệu huấn luyện on-policy trực tiếp từ các ví dụ minh họa.

---

## Slide 5 — Phương pháp tiền nhiệm 1: Supervised Fine-Tuning (SFT)
* **Tiêu đề:** TIỀN NHIỆM 1: SUPERVISED FINE-TUNING (SFT)
* **3 cột (Ý tưởng cốt lõi | Hạn chế còn lại | Hệ quả thực tế):**
  * **Ý tưởng cốt lõi:** Tối đa hóa log-likelihood (giảm cross-entropy) trực tiếp trên các cặp `(x, c)` cố định từ chuyên gia, bất kể mô hình ban đầu diễn đạt ra sao.
  * **Hạn chế còn lại:** Hoàn toàn **off-policy** — quá trình huấn luyện chỉ bao phủ các chuỗi tiền tố do chuyên gia viết, không bao giờ huấn luyện mô hình cách tự sửa sai trên tiền tố do chính nó sinh ra.
  * **Hệ quả thực tế:**
    1. **Tích tụ sai số (Compounding errors):** Chỉ cần sinh lệch một token lúc suy luận, mô hình rơi vào trạng thái lạ và sai dây chuyền.
    2. **Quên thảm khốc:** Trên bài toán suy luận Hóa học (SciKnowEval), điểm trung bình 6 benchmark nền tảng (`Prior Avg`: HellaSwag, HumanEval, IFEval, MMLU, TruthfulQA, Winogrande) tụt từ **65.5% xuống 53.4%** (riêng IFEval giảm từ **74.3% còn 35.3%**).
    3. **Sụp đổ lập luận (CoT collapse):** Khi đáp án mẫu ngắn gọn, SFT ép mô hình suy luận bỏ qua bước phân tích dài.
* [1] I. Shenfeld et al., "Self-Distillation Enables Continual Learning," arXiv:2601.19897v2, Sec. 2 & Table 5, 2026.
* [2] S. Ross, G. Gordon, and D. Bagnell, "A reduction of imitation learning and structured prediction to no-regret online learning," AISTATS, pp. 627–635, 2011.
* **Speaker Notes:** Phương pháp tiền nhiệm đầu tiên là Supervised Fine-Tuning (SFT). Về bản chất, SFT tối ưu hàm mất mát cross-entropy để ép phân phối đầu ra của mô hình trùng khớp từng token với đáp án mẫu cố định từ chuyên gia. Dù đơn giản và không cần hàm thưởng, hạn chế căn bản của SFT nằm ở tính chất off-policy: dữ liệu huấn luyện luôn đi theo quỹ đạo của chuyên gia, trong khi lúc kiểm thử thực tế, mô hình phải tự điều kiện hóa trên các token do chính nó vừa sinh ra. Theo kết quả kinh điển của Ross et al. trong học bắt chước, sự lệch phân phối trạng thái này khiến sai số tích tụ theo bình phương độ dài chuỗi. Hệ quả thực tế thể hiện qua ba biểu hiện rõ rệt: hiệu năng tác vụ mới tổng quát hóa kém; năng lực cũ sụt giảm nghiêm trọng — cụ thể khi tinh chỉnh Qwen2.5-7B trên tập Khoa học, điểm trung bình 6 bộ kiểm tra tổng quát (gồm HellaSwag, HumanEval, IFEval, MMLU, TruthfulQA và Winogrande) rơi từ 65.5% xuống chỉ còn 53.4%, trong đó khả năng tuân thủ chỉ dẫn IFEval tụt hơn một nửa từ 74.3% xuống 35.3%; và cuối cùng là hiện tượng rút ngắn chuỗi suy nghĩ khi dạy mô hình reasoning bằng đáp án ngắn.

---

## Slide 6 — Phương pháp tiền nhiệm 2: Inverse RL & Offline Context Distillation
* **Tiêu đề:** TIỀN NHIỆM 2: INVERSE RL & OFFLINE CONTEXT DISTILLATION
* **Khối trái — Inverse Reinforcement Learning (IRL):**
  * **Ý tưởng cốt lõi:** Suy luận một hàm thưởng ẩn sao cho các mẫu của chuyên gia là tối ưu, rồi dùng hàm thưởng đó chạy on-policy RL.
  * **Hạn chế:** Bài toán bất định (ill-posed) nếu thiếu các giả định cấu trúc mạnh (như phân phối Boltzmann cực đại entropy, bộ phân biệt đối nghịch hoặc cặp so sánh ưa thích trong RLHF).
  * **Hệ quả thực tế:** Khó mở rộng ổn định lên không gian từ vựng hàng trăm nghìn token của LLM và tốn kém chi phí huấn luyện reward model.
* **Khối phải — Offline Context Distillation:**
  * **Ý tưởng cốt lõi:** Cho chính mô hình đọc thêm ngữ cảnh hướng dẫn (teacher) rồi chưng cất tri thức xuống phiên bản không có ngữ cảnh (student).
  * **Hạn chế:** Dùng một tiền tố ngữ cảnh tĩnh chung cho mọi câu hỏi và huấn luyện **off-policy** trên chuỗi câu trả lời do teacher sinh sẵn.
  * **Hệ quả thực tế:** Không truyền đạt được ý định tinh tế theo từng ví dụ và vẫn mắc lỗi lệch phân phối khi student tự suy luận.
* [4] A. Y. Ng and S. Russell, "Algorithms for inverse reinforcement learning," ICML, 2000.
* [5] C. Snell, D. Klein, and R. Zhong, "Learning by distilling context," arXiv:2209.15189, 2022.
* [6] R. Agarwal et al., "On-policy distillation of language models: Learning from self-generated mistakes," ICLR, 2024.
* **Speaker Notes:** Nhóm phương pháp tiền nhiệm thứ hai cố gắng né tránh SFT theo hai hướng khác nhau. Hướng thứ nhất là Inverse RL (IRL), với ý tưởng tách bài toán thành hai bước: suy luận hàm thưởng giải thích hành vi chuyên gia rồi chạy RL on-policy. Tuy đẹp về mặt toán học, IRL truyền thống đòi hỏi các ràng buộc cấu trúc rất chặt để hàm thưởng xác định được, khiến nó khó áp dụng đại trà cho LLM nếu không có dữ liệu so sánh cặp như RLHF. Hướng thứ hai là Offline Context Distillation, nơi cùng một mô hình khi được cấp thêm ngữ cảnh phụ (ví dụ hướng dẫn hệ thống hoặc vài ví dụ mẫu) đóng vai thầy cho chính nó khi không có ngữ cảnh. Tuy nhiên, các phương pháp trước đây chủ yếu dùng ngữ cảnh cố định cho toàn bộ tập dữ liệu và chưng cất ngoại tuyến trên dữ liệu thầy sinh ra. Hệ quả là mô hình học trò vẫn không được sửa lỗi ngay trên quỹ đạo tự sinh của bản thân. SDFT giải quyết cả hai hạn chế này bằng cách kết hợp điều kiện hóa động theo từng mẫu với chưng cất on-policy.

---

## Slide 7 — Đóng góp chính & Tính mới so với Tiền nhiệm
* **Tiêu đề:** ĐÓNG GÓP CHÍNH & TÍNH MỚI SO VỚI TIỀN NHIỆM
* **Tóm tắt lời tác giả:** Đề xuất **SDFT** — biến dữ liệu mẫu chuyên gia thành tín hiệu học on-policy qua In-Context Learning để học kỹ năng, nạp tri thức và học tuần tự không quên.
* **Bảng phân loại đóng góp & Bước tiến so với trước đây:**
  * **Lý thuyết (Theoretical):**
    * *Đóng góp:* Chứng minh chưng cất đảo ngược (Reverse KL) tương đương với On-policy RL tối ưu một hàm thưởng nội tại dạng IRL (`log π_teacher − log π_student`) nhờ giả định In-Context.
    * *Trước đây chưa làm được:* IRL đòi hỏi mô hình hóa reward tường minh hoặc cặp so sánh; Context Distillation chỉ được xem là nén prompt ngoại tuyến chứ chưa được gắn kết như một cơ chế IRL on-policy.
  * **Kỹ thuật (Technical):**
    * *Đóng góp:* Cùng một LLM đóng 2 vai: Student `π_θ(·|x)` và EMA Teacher `π_φ(·|x,c)` điều kiện hóa theo từng ví dụ `c`, dùng chỉ **1 rollout/prompt**, analytic per-token Forward KL và mask token đầu.
    * *Trước đây chưa làm được:* IRL/RLHF cần mô hình thầy hoặc reward riêng; GRPO cần nhiều rollout/prompt; Context Distillation trước đây dùng ngữ cảnh tĩnh và off-policy.
  * **Thực nghiệm (Empirical):**
    * *Đóng góp:* Dẫn đầu biên Pareto trên 3 tác vụ Skill Learning, đạt **98%** câu hỏi suy luận ngoài phân phối (OOD) ở Knowledge Acquisition, học nối tiếp 3 tác vụ không thoái lui, và tinh chỉnh mô hình suy luận chỉ từ đáp án cuối.
    * *Trước đây chưa làm được:* SFT và các phương pháp bù trừ sau SFT vẫn suy giảm năng lực cũ đáng kể và làm sụp đổ chuỗi lập luận dài.
* *Ghi nhận trung thực (bài báo đồng thời):* Hai công trình độc lập cùng thời điểm (Zhao et al., 2026; Hübotter et al., 2026) cũng đề xuất thuật toán tự chưng cất tương tự.
* [1] I. Shenfeld et al., "Self-Distillation Enables Continual Learning," arXiv:2601.19897v2, Sec. 2–4, 2026.
* [5] C. Snell, D. Klein, and R. Zhong, arXiv:2209.15189, 2022.
* [6] R. Agarwal et al., ICLR, 2024.
* **Speaker Notes:** Nếu phân loại các đóng góp của bài báo theo ba trục lý thuyết, kỹ thuật và thực nghiệm, tính mới thể hiện rõ khi đặt cạnh các hướng tiền nhiệm. Về mặt lý thuyết, tác giả nối liền khoảng cách giữa chưng cất ngữ cảnh và Inverse RL bằng cách chỉ ra tỷ lệ log xác suất giữa mô hình có mẫu và mô hình không mẫu đóng vai trò như một phần thưởng nội tại ở từng token, loại bỏ hoàn toàn bước huấn luyện reward model của IRL. Về mặt kỹ thuật, thay vì dùng ngữ cảnh tĩnh và học off-policy như context distillation cổ điển, hay sinh nhiều quỹ đạo như GRPO, SDFT điều kiện hóa Teacher động cho từng câu hỏi, cập nhật Teacher bằng trung bình trượt mũ (EMA) để giữ ổn định, và chỉ cần một rollout mỗi prompt. Về mặt thực nghiệm, SDFT vượt trội các đối thủ trên cả học kỹ năng đơn lẻ, học nối tiếp nhiều kỹ năng, nạp tri thức mới lẫn bảo toàn độ sâu suy luận. Để đảm bảo khách quan khoa học, bài báo cũng ghi nhận hai nghiên cứu độc lập xuất hiện song song vào đầu năm 2026 của Zhao et al. và Hübotter et al. cùng khai thác ý tưởng tự chưng cất on-policy này.

---

## Slide 8 — Phương pháp (Lớp 1): Ký hiệu thống nhất & Bài toán hình thức
* **Tiêu đề:** PHƯƠNG PHÁP (LỚP 1): KÝ HIỆU & PHÁT BIỂU BÀI TOÁN
* **Bảng quy ước Ký hiệu đầy đủ (dùng thống nhất toàn bộ Deck):**
  * `D = {(x_i, c_i)}`: Tập dữ liệu gồm câu hỏi đầu vào `x_i` và lời giải mẫu chuyên gia `c_i`.
  * `y = (y₁, ..., y_T)`: Chuỗi phản hồi độ dài `T` do Student sinh ra; `y_<t` là tiền tố trước bước `t`; `V` là tập từ vựng (vocabulary).
  * `π_θ(· | x)`: Chính sách Student (tham số `θ`), chỉ nhận câu hỏi `x` — vừa sinh `y`, vừa nhận cập nhật gradient.
  * `π_φ(· | x, c)`: Chính sách Teacher (tham số EMA `φ`), dùng chung kiến trúc nhưng nhận cả `x` và mẫu `c` (không tính gradient).
  * `π_0`, `π_k`: Chính sách gốc ban đầu trước khi học và chính sách tại bước tối ưu thứ `k`.
  * `Ctx_S(x)`, `Ctx_T(x, c)`: Hàm ghép prompt cho Student và Teacher; `ℓ^S_t`, `ℓ^T_t`: log-xác suất (logprobs) của Student và Teacher tại token `y_t`.
  * `r(y, x)`, `r_t`: Hàm thưởng cấp chuỗi và hàm thưởng tức thời tại bước token `t`.
  * `α, β, η`: Hệ số cập nhật EMA Teacher (`α`), hệ số phạt KL trong Trust-Region (`β`), và tốc độ học learning rate (`η`).
* **Phát biểu bài toán hình thức (Mục tiêu lý thuyết):**
  * `min_θ  L(θ) = E_{(x,c)~D} [ E_{y ~ π_θ(·|x)} [ log( π_θ(y | x) / π_φ(y | x, c) ) ] ] = E_{(x,c)~D} [ D_KL( π_θ(· | x) ‖ π_φ(· | x, c) ) ]`
  * So sánh với SFT: SFT tối thiểu hóa `−E_{(x,c)~D}[ log π_θ(c | x) ]` (kỳ vọng trên mẫu `c` cố định → off-policy), còn SDFT lấy kỳ vọng trên `y ~ π_θ(· | x)` (chuỗi tự sinh → **on-policy**).
* [1] I. Shenfeld et al., "Self-Distillation Enables Continual Learning," arXiv:2601.19897v2, Sec. 3 & Algorithm 1, 2026.
* **Speaker Notes:** Lớp đầu tiên của phần trình bày phương pháp chuẩn hóa toàn bộ ký hiệu sẽ xuất hiện xuyên suốt các slide tiếp theo. Tập dữ liệu huấn luyện D chứa các cặp (x, c), trong đó x là đề bài và c là lời giải minh họa của chuyên gia. Điểm phân biệt giữa hai vai trò của cùng một mô hình nằm ở đầu vào và bộ trọng số: Student pi_theta chỉ đọc ngữ cảnh Ctx_S(x) và tự sinh ra chuỗi token y = (y_1, ..., y_T) thuộc từ vựng V, trong khi Teacher pi_phi dùng bộ trọng số EMA phi (khởi tạo bằng mô hình gốc pi_0) và đọc ngữ cảnh Ctx_T(x, c) có kèm lời giải mẫu. Tại mỗi bước t, hai mô hình lần lượt xuất ra log-xác suất l^S_t và l^T_t trên tiền tố y_<t do Student tạo ra. Bài toán hình thức được phát biểu là tìm tham số theta nhằm tối thiểu hóa phân kỳ Kullback-Leibler giữa phân phối của Student và Teacher, với kỳ vọng lấy trực tiếp trên quỹ đạo y sinh từ chính pi_theta. Việc lấy kỳ vọng theo phân phối tự sinh của Student chính là điểm biến thuật toán thành học on-policy, trái ngược với SFT vốn tối đa hóa xác suất trên chuỗi c cố định.

---

## Slide 9 — Phương pháp (Lớp 2): Sơ đồ Pipeline End-to-End (Tự vẽ)
* **Tiêu đề:** PHƯƠNG PHÁP (LỚP 2): LUỒNG XỬ LÝ END-TO-END CỦA SDFT
* **Hình ảnh chính:** Chèn hình tự vẽ `presentation/assets/pipeline_sdft.png`.
* **Diễn giải luồng dữ liệu qua 4 bước trên một minibatch `(x, c)`:**
  1. **On-policy Student Rollout:** Student nhận `Ctx_S(x)` và tự lấy mẫu một chuỗi phản hồi `y = (y₁…y_T) ~ π_θ(· | x)`.
  2. **Chấm điểm song song trên cùng quỹ đạo `y`:** Đưa chính chuỗi `y` qua Student (ngữ cảnh `x`) để thu logits/logprobs `ℓ^S_t`, và qua EMA Teacher (ngữ cảnh `x, c`, đóng băng gradient) để thu `ℓ^T_t`.
  3. **Tính hàm mất mát KL giải tích theo từng token:** Tính phân kỳ KL trên toàn bộ từ vựng `V` tại từng bước `t` (bỏ qua `k` token mở đầu để tránh sao chép câu dẫn ngữ cảnh).
  4. **Cập nhật 2 bước:** Cập nhật trọng số Student `θ ← θ − η·g` bằng AdamW, sau đó cập nhật mượt Teacher theo trung bình trượt mũ `φ ← α·θ + (1 − α)·φ`.
* [1] I. Shenfeld et al., "Self-Distillation Enables Continual Learning," arXiv:2601.19897v2, Fig. 2 & Algorithm 1, 2026. (Sơ đồ do nhóm tự vẽ lại).
* **Speaker Notes:** Sơ đồ tự vẽ minh họa trọn vẹn vòng lặp huấn luyện một bước của SDFT từ đầu vào đến cập nhật tham số. Hãy hình dung với một câu hỏi Hóa học cụ thể: đầu tiên Student chỉ nhận đề bài x và bốn lựa chọn, rồi tự viết ra quá trình lập luận y từ đầu đến cuối — giả sử đến bước thứ 50 Student bắt đầu tính sai công thức. Ngay sau đó, toàn bộ chuỗi y vừa sinh được ghép vào sau ngữ cảnh chứa đáp án mẫu c để đưa qua EMA Teacher (không lan truyền ngược gradient). Vì Teacher nhìn thấy đáp án mẫu c, tại đúng bước thứ 50 nơi Student đi chệch hướng, phân phối xác suất trên từ vựng của Teacher sẽ dồn vào token lập luận đúng, tạo ra tín hiệu hiệu chỉnh chính xác tại chỗ sai của Student. Ở bước cuối, hàm mất mát KL giải tích tổng hợp tín hiệu hiệu chỉnh trên toàn chuỗi (có che đi 3 token đầu tiên), truyền gradient để cập nhật bộ tham số theta của Student, rồi hòa trộn nhẹ theta vào phi của Teacher theo tỷ lệ alpha.

---

## Slide 10 — Phương pháp (Lớp 3a): Thành phần 1 — Teacher điều kiện hóa bằng Demonstration
* **Tiêu đề:** LỚP 3A: TEACHER ĐIỀU KIỆN HÓA BẰNG DEMONSTRATION
* **Cấu trúc What / Why / How:**
  * **Làm gì (What):** Biến chính mô hình đang huấn luyện thành Teacher `π_φ(· | x, c)` bằng cách chèn ví dụ minh họa `c` của từng câu hỏi vào prompt `Ctx_T(x, c)`.
  * **Vì sao cần (Why):** Tận dụng năng lực **học theo ngữ cảnh (In-Context Learning — ICL)** của LLM để biến các ví dụ rời rạc thành phân phối xác suất dày đặc ở mức token, mà không cần một mô hình thầy lớn hơn bên ngoài.
  * **Hoạt động ra sao (How — Template prompt 3 dòng):**
    `<Question>`
    `This is an example for a response to the question:`
    `<Demonstration>`
    `Now answer with a response of your own, including the thinking process:`
* **Cơ chế chống chép nguyên văn & bằng chứng chọn ngữ cảnh:**
  * Câu lệnh yêu cầu *"trả lời theo cách riêng kèm quá trình suy nghĩ"* ngăn Teacher lặp lại y nguyên chuỗi `c`, buộc nó tái cấu trúc lập luận phù hợp với phân phối gốc của mô hình.
  * Trên bài toán nạp kiến thức mới (Wikipedia 2025, đo bằng độ chính xác tuyệt đối — Strict Accuracy), việc đưa **cả bài viết nguồn lẫn đáp án mẫu** đạt **89%**, vượt xa chỉ đưa bài viết (**75%**) hoặc chỉ đưa đáp án (**37%**).
* [1] I. Shenfeld et al., "Self-Distillation Enables Continual Learning," arXiv:2601.19897v2, Sec. 3 & Appendix A.2, 2026.
* [11] T. Brown et al., "Language models are few-shot learners," NeurIPS, 2020.
* **Speaker Notes:** Thành phần đầu tiên trong kiến trúc SDFT là cơ chế tạo ra Teacher từ chính bản thân mô hình. Về mặt chức năng, Teacher không phải một mạng lớn hơn hay mô hình thương mại bên ngoài, mà là phiên bản EMA của chính Student được cung cấp thêm lời giải mẫu c riêng biệt cho từng câu hỏi x. Lý do thiết kế này là vì các mô hình ngôn ngữ tiền huấn luyện quy mô đủ lớn sở hữu năng lực In-Context Learning mạnh mẽ: khi nhìn thấy đáp án đúng trong ngữ cảnh, phân phối xác suất đầu ra của mô hình dịch chuyển rõ rệt về phía câu trả lời chuẩn xác mà không cần cập nhật trọng số. Cách thực hiện rất gọn qua mẫu prompt ba dòng: sau khi trình bày ví dụ, câu nhắc "hãy tự trả lời theo cách của bạn kèm quá trình suy nghĩ" ngăn mô hình chép vẹt chuỗi mẫu c. Thực nghiệm tách thành phần của tác giả trên bài toán nạp tri thức mới khẳng định rằng việc kết hợp cả ngữ cảnh nền và đáp án mẫu đạt độ chính xác tuyệt đối 89%, cao hơn hẳn 75% khi chỉ đưa văn bản nguồn và 37% khi chỉ đưa đáp án.

---

## Slide 11 — Phương pháp (Lớp 3b): Thành phần 2 — On-Policy Rollout & Hàm mất mát KL
* **Tiêu đề:** LỚP 3B: ON-POLICY ROLLOUT & ANALYTIC PER-TOKEN KL
* **Khối trái — Rollout On-policy (What / Why / How):**
  * **Làm gì & Vì sao:** Sinh duy nhất **1 chuỗi `y ~ π_θ(·|x)`** mỗi prompt để đảm bảo Student được sửa sai trực tiếp trên tiền tố do chính nó tạo ra (tránh tích tụ sai số off-policy).
  * **Vì sao không sinh nhiều mẫu?** Kiểm chứng thực nghiệm cho thấy tăng số rollout/prompt không cải thiện độ chính xác đáng kể nhưng làm chi phí tính toán tăng tuyến tính.
* **Khối phải — Công thức KL & 3 Bộ ước lượng Gradient:**
  * **Lý thuyết (Reverse KL phân rã theo token):** `∇_θ L_rev = E_{y~π_θ} [ ∑_t ∑_{v∈V} π_θ(v|y_<t,x) · log(π_θ(v|y_<t,x) / π_φ(v|y_<t,x,c)) · ∇_θ log π_θ(v|y_<t,x) ]`
  * **Thực tế cài đặt (Analytic Forward KL ở mức logit):** `L_fwd = ∑_t ∑_{v∈V} π_φ(v|y_<t,x,c) · log( π_φ(v|y_<t,x,c) / π_θ(v|y_<t,x) )` (chỉ truyền gradient vào `π_θ`).
  * **So sánh 3 bộ ước lượng KL (Bias | Variance | Chi phí):**
    1. *Sampled token-level partial:* Chệch (biased, bỏ qua tác động lên `y_>t`), phương sai cao, rẻ.
    2. *Full analytic per-token (được chọn):* Vẫn chệch cấp chuỗi, nhưng **phương sai thấp nhất**, tận dụng sẵn logits có từ forward pass → **ổn định và chính xác nhất**.
    3. *Rao-Blackwellized:* Không chệch (unbiased), phương sai thấp, nhưng tốn thêm tính toán tiền tố và không cho lợi ích thực tế vượt trội.
* [1] I. Shenfeld et al., "Self-Distillation Enables Continual Learning," arXiv:2601.19897v2, Sec. 3 & Appendix A.1, 2026.
* [10] Y. Tang and R. Munos, "On a few pitfalls in KL divergence gradient estimation for RL," arXiv:2506.09477, 2025.
* **Speaker Notes:** Thành phần thứ hai quy định cách lấy mẫu dữ liệu và hàm mất mát chưng cất. Để giữ chi phí hợp lý, Student chỉ lấy mẫu một chuỗi duy nhất cho mỗi câu hỏi thay vì lấy mẫu theo nhóm lớn như GRPO. Về hàm mất mát, dù phần dẫn chứng lý thuyết xuất phát từ Reverse KL ở cấp chuỗi và phân rã thành gradient từng token trên toàn bộ từ vựng V, tác giả ghi nhận ở phần triển khai thực tế rằng việc dùng Forward KL trên logit — tức lấy phân phối của Teacher làm trọng số kéo Student phủ lên các token đúng — lại đạt hiệu năng cao và ổn định hơn. Bên cạnh hướng KL, bài báo cũng so sánh ba cách ước lượng gradient phân kỳ KL. Cách thứ nhất chỉ lấy mẫu tại đúng token vừa sinh nên phương sai rất cao. Cách thứ ba dùng kỹ thuật Rao-Blackwellized để khử độ chệch cấp chuỗi (tức tác động của token hiện tại lên các bước tương lai y_>t), nhưng đòi hỏi tính tổng tiền tố đắt đỏ mà không đem lại cải thiện thực tế. Vì vậy, SDFT chọn bộ ước lượng giải tích toàn từ vựng tại từng bước (full analytic per-token) vì nó tận dụng ngay tensor logits sẵn có và cho phương sai thấp nhất.

---

## Slide 12 — Phương pháp (Lớp 3c): Thành phần 3 — EMA Teacher & Masking Token đầu
* **Tiêu đề:** LỚP 3C: EMA TEACHER & XỬ LÝ LEARNED ARTIFACTS
* **Khối trái — Tham số hóa Teacher (What / Why / How):**
  * **Làm gì:** Cập nhật trọng số Teacher sau mỗi bước bằng trung bình trượt mũ: `φ ← α·θ + (1 − α)·φ`.
  * **Vì sao không dùng mô hình cố định hay chính Student hiện tại?**
    * **Frozen Base (`α = 0`, `φ = θ₀`):** Huấn luyện ổn định nhưng Teacher không tiến bộ theo Student → trần hiệu năng bị kìm hãm.
    * **Live Student (`α = 1`, `φ = θ`):** Vòng phản hồi tự củng cố khuếch đại nhiễu xác suất nhỏ qua mỗi rollout → **mất ổn định và phân kỳ nhanh chóng** (độ chính xác sụp xuống gần 0 sau ~500 bước).
    * **EMA (`α ∈ {0.01, 0.02, 0.05}`; code chính thức dùng `0.01`):** Theo sát tiến trình học của Student đồng thời làm mượt nhiễu tức thời.
* **Khối phải — Learned Artifacts & Masking (`k = 3` tokens):**
  * **Hiện tượng:** Do Teacher nhìn thấy văn bản/ví dụ `c`, Teacher thường mở đầu bằng *"Based on the text..."* hoặc *"Following the example..."*.
  * **Hệ quả:** Student không hề được cấp `c` lúc test nhưng vẫn học lây các câu mở đầu vô nghĩa này.
  * **Cách xử lý:** Nhân mask `0` vào hàm loss KL ở **`k` token đầu tiên** của chuỗi trả lời (`k = 3` trong mã nguồn chính thức). Đây là giải pháp kinh nghiệm (*heuristic*) hiệu quả.
* [1] I. Shenfeld et al., "Self-Distillation Enables Continual Learning," arXiv:2601.19897v2, Sec. 5 & Appendix A.3, 2026.
* [20] I. Shenfeld, "Self-Distillation" (official repository), GitHub: `idanshen/Self-Distillation`, 2026.
* **Speaker Notes:** Thành phần thứ ba đảm bảo quá trình tự chưng cất hội tụ ổn định và sạch nhiễu. Một câu hỏi tự nhiên là vì sao không giữ nguyên mô hình gốc làm thầy hoặc lấy ngay trọng số hiện tại của Student làm thầy. Kết quả thực nghiệm kiểm chứng ba cấu hình thầy cho thấy: nếu đóng băng thầy ở trạng thái ban đầu (alpha = 0), mô hình học trò dừng tiến bộ sớm vì không khai thác được khả năng lập luận mới hình thành; ngược lại, nếu lấy chính trọng số Student đang thay đổi từng bước (alpha = 1), các dao động ngẫu nhiên trong log-xác suất bị vòng lặp on-policy khuếch đại liên hoàn, khiến hiệu năng tăng nhanh rồi sụp đổ hoàn toàn chỉ sau vài trăm lượt sinh. Do đó, cập nhật EMA với alpha nhỏ (0.01 trong code chính thức) tạo ra điểm cân bằng tối ưu. Ngoài ra, một lỗi tinh tế phát sinh khi Teacher đọc ngữ cảnh mẫu c là nó hay sinh các cụm từ quy chiếu như "Dựa theo ví dụ trên...". Nếu ép Student khớp các token đầu này, Student sẽ nói ra các cụm từ vô nghĩa khi kiểm thử không có ví dụ; việc che mất mát trên 3 token mở đầu triệt tiêu hiện tượng này mà không hại độ chính xác.

---

## Slide 13 — Phương pháp (Lớp 4a): Tính chất Lý thuyết — SDFT như một dạng Inverse RL
* **Tiêu đề:** LỚP 4A: CƠ SỞ LÝ THUYẾT — SDFT LÀ MỘT DẠNG INVERSE RL
* **Chuỗi suy diễn 5 bước (từ Trust-Region RL đến Gradient SDFT):**
  1. **Bài toán Trust-Region RL:** Cập nhật chính sách mới `π_{k+1}` sao cho tối đa phần thưởng kỳ vọng nhưng không lệch quá xa chính sách hiện tại `π_k`:
     `π_{k+1} = argmax_π E_{y~π}[ r(y, x) ] − β · D_KL( π(·|x) ‖ π_k(·|x) )`
  2. **Nghiệm dạng đóng (Tilted Distribution):** Nghiệm tối ưu thỏa mãn `π*_{k+1}(y | x) ∝ π_k(y | x) · exp( r(y, x) / β )`.
  3. **Rút biểu thức Reward:** Đảo ngược công thức trên, bỏ hằng số tuyến tính `β, C`: `r(y, x) = log π*_{k+1}(y | x) − log π_k(y | x)`.
  4. **Giả định In-Context (Cốt lõi):** Khi được điều kiện hóa bởi mẫu `c`, mô hình xấp xỉ chính sách tối ưu chưa biết: **`π*_{k+1}(y | x) ≈ π(y | x, c)`**.
     ⇒ **Phần thưởng nội tại từng token:** `r_t(y_t | y_<t, x, c) = log π(y_t | y_<t, x, c) − log π_k(y_t | y_<t, x)` (tổng theo `t` cho ra `r(y, x, c)`).
  5. **Tương đương Gradient:** `∇_θ J(π_k) = E_{y~π_k}[ r(y,x,c) · ∇_θ log π_k(y|x) ] = E_{y~π_k}[ log(π(y|x,c)/π_k(y|x)) · ∇_θ log π_k(y|x) ]` — **trùng khớp theo kỳ vọng với gradient của Reverse KL phân rã ở Slide 11**.
* **Ý nghĩa & Giới hạn lý thuyết:** Giải thích *vì sao* SDFT mang đặc tính chống quên của on-policy RL dù không có hàm thưởng; nhưng **chứng minh chỉ đúng với Reverse KL (không phải Forward KL thực tế)** và **đứng vững hoàn toàn trên Giả định 4**.
* [1] I. Shenfeld et al., "Self-Distillation Enables Continual Learning," arXiv:2601.19897v2, Sec. 3.1, 2026.
* [8] J. Schulman et al., "Trust region policy optimization," ICML, pp. 1889–1897, 2015.
* [9] R. Rafailov et al., "Direct preference optimization," NeurIPS, 2023.
* **Speaker Notes:** Lớp thứ tư đi sâu vào tính chất lý thuyết của SDFT. Tác giả xuất phát từ bài toán học tăng cường có vùng tin cậy (Trust-Region RL) chuẩn mực của Schulman et al., trong đó chính sách mới pi_{k+1} phải tối đa hóa phần thưởng kỳ vọng đồng thời chịu phạt phân kỳ KL để bám sát chính sách hiện tại pi_k. Theo kết quả đã biết trong DPO, nghiệm tối ưu pi*_{k+1} có dạng phân phối mũ nghiêng theo hàm thưởng. Khi đảo ngược phương trình, phần thưởng được biểu diễn trực tiếp qua hiệu log-xác suất giữa chính sách tối ưu và chính sách cũ. Trong IRL thông thường ta không biết pi*_{k+1}, nhưng SDFT đưa ra giả định In-Context: mô hình được cấp lời giải mẫu c chính là một xấp xỉ tốt cho pi*_{k+1}. Nhờ tính tự hồi quy của LLM, phần thưởng toàn chuỗi phân rã thành tổng phần thưởng tức thời tại từng token — trực giác là "Teacher ưa thích token này hơn Student bao nhiêu". Từ đó, phép lấy đạo hàm chính sách (policy gradient) cho thấy tối ưu phần thưởng nội tại này tương đương theo kỳ vọng với gradient của Reverse KL đã viết ở Slide 11. Giới hạn cần ghi nhớ là phép quy đổi này chỉ chính xác với Reverse KL và phụ thuộc hoàn toàn vào chất lượng In-Context của mô hình gốc.

---

## Slide 14 — Phương pháp (Lớp 4b): Kiểm chứng Thực nghiệm Giả định In-Context
* **Tiêu đề:** LỚP 4B: KIỂM CHỨNG 2 ĐIỀU KIỆN CỦA GIẢ ĐỊNH ICL
* **Bối cảnh kiểm chứng:** Bài toán gọi hàm API công cụ **ToolAlpaca** (nhận mô tả API + yêu cầu người dùng → sinh đúng lệnh gọi hàm) trên mô hình **`Qwen2.5-7B-Instruct`**.
* **Khối trái — Điều kiện 1: Tính Tối ưu (Optimality):**
  * *Yêu cầu:* Teacher `π(·|x, c)` phải đạt phần thưởng kỳ vọng gần tối đa như chính sách tối ưu `π*_{k+1}`.
  * *Kết quả đo của bài báo:*
    * Base model (không có mẫu `c`): chỉ giải đúng **42%** số ví dụ.
    * Teacher (được cấp mẫu `c`): đạt **100%** tỷ lệ thành công.
    * Kiểm tra thủ công **50/50** chuỗi lập luận của Teacher: cả lời gọi tool cuối lẫn chuỗi suy nghĩ trung gian (CoT) đều hợp lệ về ngữ nghĩa, không phải sao chép máy móc.
* **Khối phải — Điều kiện 2: Độ lệch Tối thiểu (Minimal Deviation):**
  * *Yêu cầu:* Theo Trust-Region RL, trong số các chính sách đạt thưởng tối ưu, Teacher phải nằm gần chính sách gốc `π₀` nhất theo nghĩa KL.
  * *Đo `D_KL(π ‖ π₀)` trên ToolAlpaca:*
    * Mô hình học bằng **SFT** lệch khỏi Base: **1.26 nats**.
    * **Teacher có mẫu `c`** lệch khỏi Base: chỉ **0.68 nats** (**giảm gần một nửa**).
* [1] I. Shenfeld et al., "Self-Distillation Enables Continual Learning," arXiv:2601.19897v2, Sec. 3.2, 2026.
* [13] Q. Tang et al., "ToolAlpaca: Generalized tool learning for language models with 3000 simulated cases," arXiv:2306.05301, 2023.
* **Speaker Notes:** Vì giả định In-Context không thể chứng minh thuần túy bằng đại số, tác giả tiến hành kiểm chứng thực nghiệm hai điều kiện cần thiết của nó trên tập bài toán ToolAlpaca — nơi mô hình phải đọc tài liệu API và tạo ra lệnh gọi hàm chính xác. Điều kiện thứ nhất là tính tối ưu (Optimality): mẫu c phải thực sự giúp Teacher đạt hiệu suất gần trần. Kết quả cho thấy Qwen2.5-7B-Instruct khi không có mẫu chỉ đúng 42%, nhưng khi ghép mẫu vào ngữ cảnh thì đạt tuyệt đối 100%, và kiểm tra bằng mắt 50 chuỗi lập luận cho thấy Teacher tự dựng lại quá trình suy nghĩ hợp lý thay vì chép lại nguyên văn. Điều kiện thứ hai là độ lệch tối thiểu (Minimal Deviation): trong bài toán Trust-Region, nghiệm tối ưu phải là phân phối gần chính sách hiện tại nhất mà vẫn giải được bài. Khi đo phân kỳ KL so với mô hình gốc ban đầu pi_0, mô hình sau khi fine-tune bằng SFT bị kéo lệch đi tới 1.26 nats, trong khi Teacher điều kiện hóa bởi mẫu chỉ cách mô hình gốc 0.68 nats — tức gần một nửa. Kết quả này củng cố niềm tin rằng Teacher đóng vai trò một mục tiêu tin cậy cho bước chưng cất on-policy.

---

## Slide 15 — Phương pháp (Lớp 5): Độ phức tạp Thời gian, Bộ nhớ & So sánh Tiền nhiệm
* **Tiêu đề:** LỚP 5: ĐỘ PHỨC TẠP TÍNH TOÁN & SO SÁNH PHƯƠNG PHÁP
* **Bảng so sánh chi phí mỗi bước (trên 1 mẫu dài câu hỏi `|x|`, mẫu `|c|`, sinh `T` tokens):**

| Phương pháp | Số rollout / prompt | Tính toán (FLOPs vs SFT) | Thời gian thực tế (Wall-clock) | Bộ nhớ VRAM & Số pha train |
| :--- | :---: | :---: | :---: | :--- |
| **SFT (Off-policy)** | 0 | `1.0×` (1 fwd+bwd trên `|x|+|c|`) | `1.0×` | 1 mô hình + optimizer (1 pha) |
| **SFT + Re-invoke** *(SFT trước rồi chưng cất ngược từ Base trên prompt chung)* | On-policy ở pha 2 | `> 2.5×` | `> 4.0×` | Student + Frozen Base (2 pha nối tiếp) |
| **GRPO** *(On-policy RL lấy mẫu theo nhóm `G`)* | `G ≥ 4–16` rollouts | Rất cao (nhân `G` lần sinh) | Rất cao | Policy + Ref + Reward (1 pha) |
| **SDFT (Bài báo — H200 + vLLM)** | **1 rollout** | **`~2.5×` FLOPs** | **`~4.0×` wall-clock** | Student + EMA Teacher (1 pha) |

* **Chi phí thực tế của SDFT:** Bước đắt nhất là sinh tự hồi quy `T` token (`y ~ π_θ`, bị nghẽn băng thông bộ nhớ), cộng 1 teacher forward trên `|x|+|c|+T` và 1 student fwd+bwd trên `|x|+T`.
* **Thích nghi bộ nhớ của nhóm:** Gắn 2 LoRA adapters (`default` & `teacher`) trên **cùng 1 base model 4-bit** → không tốn thêm VRAM chứa bản sao Teacher; nhưng trên T4 không có vLLM tối ưu 4-bit nên wall-clock dự kiến chậm hơn SFT **`~11×`**.
* [1] I. Shenfeld et al., "Self-Distillation Enables Continual Learning," arXiv:2601.19897v2, Sec. 5, 2026.
* [16] K. Lu and Thinking Machines Lab, "On-policy distillation," 2025.
* **Speaker Notes:** Lớp cuối cùng trong phân tích phương pháp là câu chuyện độ phức tạp về thời gian và không gian. Trong SFT chuẩn, mỗi mẫu chỉ cần đúng một lượt truyền xuôi và truyền ngược trên cặp (x, c), không tốn bước sinh chuỗi. Trái lại, mỗi bước của SDFT gồm ba công đoạn: Student tự sinh T token (vốn bị giới hạn bởi băng thông bộ nhớ GPU chứ không phải năng lực tính toán FLOPs), sau đó Teacher chạy truyền xuôi trên tổng độ dài |x|+|c|+T, và Student chạy truyền xuôi kèm truyền ngược trên độ dài |x|+T. Trên phần cứng H200 kết hợp engine sinh vLLM tốc độ cao, tác giả đo được SDFT tốn gấp khoảng 2.5 lần FLOPs và 4 lần thời gian thực tế so với SFT một pha. Tuy nhiên, nếu so với đối thủ chống quên Re-invoke (vốn phải chạy xong pha SFT rồi chạy thêm một pha chưng cất on-policy từ mô hình gốc để phục hồi năng lực) hoặc GRPO (phải sinh nhóm G từ 4 đến 16 chuỗi cho mỗi prompt), SDFT lại tiết kiệm hơn hẳn. Khi nhóm chạy tái tạo bằng PyTorch thuần trên GPU T4 miễn phí, tỷ lệ chậm này dự kiến tăng lên khoảng 11 lần do thiếu vLLM hỗ trợ QLoRA 4-bit.

---

## Slide 16 — Giao thức & Tập dữ liệu Thực nghiệm trong Paper
* **Tiêu đề:** PHƯƠNG PHÁP ĐÁNH GIÁ & TẬP DỮ LIỆU TRONG PAPER
* **Khối trái — 2 Bối cảnh & 4 Bộ dữ liệu (`Qwen2.5-7B-Instruct`):**
  * **Bối cảnh 1 — Học Kỹ năng (Skill Learning):**
    1. *Science Q&A:* Nhánh Hóa học cấp độ L-3 của **SciKnowEval** (chia 75% train / 5% val / 20% test; mẫu giải từ GPT-4o; chấm khớp chính xác đáp án trắc nghiệm).
    2. *Tool Use:* **ToolAlpaca** (split gốc của tác giả; mẫu có sẵn; chấm regex khớp lời gọi API bất kể thứ tự tham số).
    3. *Medical:* **HuatuoGPT-o1** (~20K câu tiếng Anh train, 1K câu test; chấm bằng giám khảo LLM `GPT-5-mini`).
  * **Bối cảnh 2 — Nạp Tri thức (Knowledge Acquisition):**
    * Tập bài viết Wikipedia về các thiên tai năm **2025** sau mốc cutoff (~200K tokens); sinh cặp hỏi-đáp bằng GPT-5, chấm bằng `GPT-5-mini`.
* **Khối phải — Các Baseline, Chỉ số Quên & Hyperparameters:**
  * **Baselines:** Base model; **SFT**; **DFT** *(SFT dùng importance sampling giả lập mẫu on-policy)*; **SFT + Re-invoke** *(chưng cất lại từ base sau SFT)*; **CPT** *(Continual Pre-training học dự đoán token tiếp theo trên văn bản thô)*; **Oracle RAG** *(RAG luôn cấp đúng bài viết)*.
  * **Đo Quên (6 benchmarks qua `lm-evaluation-harness`):** HellaSwag, TruthfulQA, MMLU, IFEval, Winogrande, HumanEval.
  * **Giao thức gốc:** 1x NVIDIA H200, Full FT (`bfloat16`), sweep `{lr, batch, epoch, α}`, chạy 3 seeds, giải mã tham lam (`temperature = 0`).
* [1] I. Shenfeld et al., "Self-Distillation Enables Continual Learning," arXiv:2601.19897v2, Sec. 4.1 & Appendix B, 2026.
* [12] K. Feng et al., "SciKnowEval," arXiv:2406.09098, 2024.
* [13] Q. Tang et al., "ToolAlpaca," arXiv:2306.05301, 2023.
* [14] L. Gao et al., "The language model evaluation harness," Zenodo, 2024.
* [15] Y. Wu et al., "On the generalization of SFT (DFT)," arXiv:2508.05629, 2025.
* **Speaker Notes:** Slide này tóm tắt toàn bộ giao thức thực nghiệm của bài báo. Tác giả sử dụng mô hình Qwen2.5-7B-Instruct làm nền tảng và đánh giá trên hai nhóm bài toán đại diện cho nhu cầu hậu huấn luyện: học kỹ năng mới và nạp tri thức mới. Ở học kỹ năng, ba bộ dữ liệu gồm Hóa học đại học (SciKnowEval L-3), gọi hàm API (ToolAlpaca) và suy luận y khoa (HuatuoGPT-o1) được chọn vì mô hình gốc chưa được tinh chỉnh chuyên sâu trên chúng. Ở nạp tri thức, tác giả thu thập các bài Wikipedia về thảm họa tự nhiên năm 2025 — chắc chắn nằm ngoài dữ liệu tiền huấn luyện. Để đo mức độ quên kiến thức cũ, mọi mô hình đều được kiểm tra qua bộ công cụ lm-evaluation-harness trên 6 benchmark tổng quát gồm suy luận thông thường (HellaSwag, Winogrande), độ trung thực (TruthfulQA), kiến thức đa ngành (MMLU), tuân thủ chỉ dẫn (IFEval) và lập trình (HumanEval). Toàn bộ thí nghiệm gốc chạy full fine-tuning trên GPU H200, chọn checkpoint tốt nhất trên tập validation và trung bình hóa qua 3 random seeds với greedy decoding.

---

## Slide 17 — Kết quả công bố: Biên Pareto ở Skill Learning
* **Tiêu đề:** KẾT QUẢ CÔNG BỐ: PARETO FRONTIER (SKILL LEARNING)
* **Hình ảnh trái:** Chèn hình tự vẽ `presentation/assets/pareto_paper_vs_repro.png`.
* **Khối phải — Số liệu chuẩn từ bài báo (`Qwen2.5-7B-Instruct`):**
  * **Mốc Base Model:** Trung bình 6 tác vụ cũ (**Prior Avg**) = **65.5%** (HellaSwag 62.0 | HumanEval 65.8 | IFEval 74.3 | MMLU 71.7 | TruthfulQA 47.9 | Winogrande 71.1).
  * **Science Q&A (Base = 32.1%):** SFT đạt `66.2%` nhưng Prior Avg tụt xuống `53.4%`; **SDFT** đạt **`70.2%`** và giữ Prior Avg ở **`64.5%`** (DFT: `54.8% / 60.2%`; Re-invoke: `66.0% / 60.2%`).
  * **Tool Use (Base = 42.9%):** SFT đạt `63.2% / 56.0%`; **SDFT** đạt **`70.6% / 65.4%`**.
  * **Medical (Base = 30.1%):** SFT đạt `35.5% / 60.2%`; **SDFT** đạt **`40.2% / 65.4%`**.
* **Kết luận:** SDFT nằm hoàn toàn ở góc trên-phải (Pareto dominance) — vừa học tác vụ mới giỏi hơn SFT, vừa giữ 98.5–99.8% năng lực cũ.
* [1] I. Shenfeld et al., "Self-Distillation Enables Continual Learning," arXiv:2601.19897v2, Sec. 4.2–4.3 & Table 5, 2026. (Biểu đồ do nhóm tự vẽ lại).
* **Speaker Notes:** Kết quả chính của bài báo ở nhóm bài toán Skill Learning được tổng hợp trực quan qua đồ thị đánh đổi Pareto, trong đó trục hoành là độ chính xác trên tác vụ mới và trục tung là điểm trung bình trên 6 tác vụ cũ. Một mô hình lý tưởng cần tiến sát góc trên-phải. Trên cả ba lĩnh vực, SFT đều bị tụt mạnh theo chiều dọc xuống vùng 53.4–60.2% so với mốc 65.5% của Base model. Các đối thủ như DFT (dùng trọng số importance sampling để giả lập on-policy trên dữ liệu tĩnh) hay SFT + Re-invoke (thêm một pha chưng cất hồi phục sau SFT) cải thiện được phần nào sự quên nhưng vẫn tụt 1.8 đến 5.3 điểm và đạt độ chính xác tác vụ mới thấp hơn. Duy nhất SDFT vừa đẩy độ chính xác tác vụ mới vượt SFT từ +4.0 đến +7.4 điểm phần trăm, vừa duy trì năng lực cũ ở mức 64.5% đến 65.4%, tức hầu như không quên. (Điểm ngôi sao xanh trên biểu đồ là vị trí chờ đối chiếu kết quả thực nghiệm của nhóm).

---

## Slide 18 — Kết quả công bố: Nạp Tri thức mới & Học nối tiếp 3 Tác vụ
* **Tiêu đề:** KẾT QUẢ CÔNG BỐ: NẠP TRI THỨC & HỌC NỐI TIẾP 3 TÁC VỤ
* **Hình ảnh trái:** Chèn hình tự vẽ `presentation/assets/sequential_cl_concept.png`.
* **Khối phải — Kết quả Nạp Tri thức (Wikipedia 2025) & Học tuần tự:**
  * **Định nghĩa 3 chỉ số Nạp Tri thức:**
    * *Strict Accuracy:* Đúng đầy đủ mọi chi tiết tham chiếu.
    * *Lenient Accuracy:* Chứa thông tin đúng và không có thông tin sai lệch.
    * *OOD Accuracy (Ngoài phân phối):* Câu hỏi gián tiếp đòi hỏi suy luận từ kiến thức đã nạp mà không nhắc thẳng tên sự kiện (vd: *"Quốc gia nào cần viện trợ quốc tế năm 2025?"*).
  * **So sánh kết quả (Base: 0 / 0 / 0%):**
    * **CPT** *(học tiếp trên văn bản thô)*: 9% Strict | 37% Lenient | 7% OOD.
    * **SFT** *(học cặp hỏi-đáp)*: 80% Strict | 95% Lenient | **80% OOD**.
    * **SDFT (Ours)**: **89% Strict** | **100% Lenient** | **98% OOD** (tiệm cận **Oracle RAG**: 91 / 100 / 100%).
  * **Học nối tiếp (Sequential CL):** Train lần lượt `Tool Use → Science → Medical` trên 1 model: SFT bị dao động và quên ngay tác vụ trước khi chuyển tác vụ; SDFT tích lũy ổn định cả 3 kỹ năng.
* [1] I. Shenfeld et al., "Self-Distillation Enables Continual Learning," arXiv:2601.19897v2, Sec. 4.2–4.3, Fig. 3 & Table 1, 2026. (Biểu đồ do nhóm tự vẽ lại).
* **Speaker Notes:** Hai thí nghiệm tiếp theo chứng minh SDFT thực sự giúp mô hình tiếp thu tri thức và tích lũy kỹ năng dài hạn. Ở thí nghiệm nạp tri thức thiên tai năm 2025, Continual Pre-training (CPT) trên văn bản thô thất bại nặng nề (chỉ đạt 9% strict và 7% OOD). SFT trên bộ câu hỏi – đáp đạt 80% câu hỏi trực tiếp nhưng giữ nguyên mức 80% cho câu hỏi gián tiếp ngoài phân phối (OOD), chứng tỏ mô hình thiên về học thuộc cặp hỏi-đáp bề mặt. Ngược lại, SDFT đạt 89% strict, 100% lenient và vọt lên 98% OOD — gần ngang bằng hệ thống Oracle RAG luôn được cấp sẵn bài viết đúng. Song song đó, biểu đồ bên trái minh họa kịch bản thử thách nhất là học tuần tự ba kỹ năng khác nhau trên cùng một bộ tham số: trong khi SFT làm điểm tác vụ cũ lao dốc ngay lập tức mỗi khi chuyển sang tác vụ mới, SDFT duy trì hiệu năng các tác vụ trước gần như đi ngang suốt 700 bước cập nhật.

---

## Slide 19 — Kết quả công bố: Kích thước Mô hình, Pass@k & Reasoning không cần CoT
* **Tiêu đề:** CÁC PHÁT HIỆN SÂU TỪ THỰC NGHIỆM KIỂM CHỨNG CỦA BÀI BÁO
* **3 thẻ phát hiện quan trọng:**
  * **1. Hiệu ứng Kích thước Mô hình (Science Q&A):**
    * `Qwen2.5-3B`: Năng lực ICL quá yếu để làm Teacher → SDFT **thua SFT `−3.3` điểm**.
    * `Qwen2.5-7B`: SDFT thắng **`+4.0` điểm** | `Qwen2.5-14B`: thắng **`+6.9` điểm**.
    * *(Quyết định then chốt: Nhóm phải giữ nguyên quy mô 7B khi tái tạo!).*
  * **2. Kiểm tra Pass@k (`k = 1` tới `k = 128`, đo xác suất có ít nhất 1 đáp án đúng trong `k` lần thử):**
    * Nhằm bác bỏ lo ngại **entropy collapse** *(hiện tượng mô hình chỉ co hẹp phân phối để tăng độ chính xác ở `k=1` mà không thực sự biết thêm lời giải mới)*.
    * Kết quả: SDFT vượt Base và SFT **đồng đều trên mọi `k` tới 128** → chứng minh học được kỹ năng thật.
  * **3. Huấn luyện Mô hình Suy luận không cần dữ liệu CoT (`Olmo-3-7B-Think` trên Medical):**
    * Base (`31.2%`, trung bình `4.612` tokens) → **SFT** tụt xuống **`23.5%`** (rút ngắn còn `3.273` tokens) → **SDFT** tăng vọt lên **`43.7%`** (giữ độ sâu `4.180` tokens).
* [1] I. Shenfeld et al., "Self-Distillation Enables Continual Learning," arXiv:2601.19897v2, Sec. 4.4–4.5, Fig. 5 & Table 2, 2026.
* **Speaker Notes:** Ba phân tích mở rộng của bài báo mang lại những hiểu biết thực tế quan trọng. Thứ nhất, vì SDFT dựa vào In-Context Learning để tạo tín hiệu Teacher, hiệu quả của nó tỷ lệ thuận với kích thước mô hình: ở cỡ 3B, khả năng học theo ngữ cảnh chưa đủ chín khiến SDFT thua SFT tới 3.3 điểm phần trăm, và chỉ đảo chiều thắng lợi rõ rệt ở 7B (+4.0) và 14B (+6.9). Đây là lý do sống còn khiến nhóm không hạ mô hình chính xuống 3B khi tái tạo. Thứ hai, để kiểm chứng mức tăng độ chính xác không phải do hiện tượng suy giảm entropy (entropy collapse — chỉ đẩy xác suất của một câu trả lời vốn đã biết lên cao), tác giả đo chỉ số Pass@k (tỷ lệ giải đúng khi cho phép lấy mẫu k lần với k lên tới 128). Đường cong SDFT nằm cao hơn SFT và Base trên toàn bộ dải k, khẳng định mô hình thực sự tiếp thu hành vi mới. Thứ ba, khi huấn luyện mô hình suy luận sâu Olmo-3-7B-Think trên dữ liệu y khoa chỉ có đáp án cuối, SFT phá hủy thói quen suy nghĩ dài, làm độ chính xác rớt từ 31.2% xuống 23.5%, trong khi SDFT giữ nguyên chiều sâu lập luận và nâng độ chính xác lên 43.7%.

---

## Slide 20 — Phạm vi Tái tạo (Core Scope) & Thích nghi Phần cứng
* **Tiêu đề:** PHẠM VI TÁI TẠO: CORE SCOPE & THÍCH NGHI PHẦN CỨNG
* **Khối trái — Lý do chọn Core Scope theo quy định môn học:**
  * Bài báo gốc chạy Full Fine-Tuning trên **NVIDIA H200 (141GB VRAM)** và quét lưới 54 cấu hình; ngoài ra tác vụ Medical cần gọi API trả phí `GPT-5-mini`, còn Knowledge Acquisition chưa công khai bộ dữ liệu.
  * Với ngân sách GPU miễn phí (**Kaggle T4 16GB/GPU**), nhóm chọn **Core Scope**:
    1. **E1 — Science Q&A (Headline):** Base, SFT, SDFT trên `Qwen2.5-7B-Instruct` (dữ liệu chính thức: `2.674` train / `507` test, tách `5%` validation; tập con `N_SCI = 352` theo đo đạc pilot).
    2. **E2 — Tool Use (ToolAlpaca):** Base, SFT, SDFT trên `Qwen2.5-7B-Instruct` (`4.046` train / `97` test; tập con `N_TOOL = 528` theo pilot).
    3. **E3 — Phân tích bổ sung (Effect of Model Size):** Chạy `1.5B / 3B / 7B` trên Science Q&A kèm đo độ chính xác Teacher có demo (`--with_demo`).
* **Khối phải — Thích nghi Phần cứng (4-bit QLoRA):**
  * Lượng tử hóa **4-bit NF4 (`fp16` compute)** kết hợp **LoRA `all-linear` (`r = 32`, `α_lora = 64`)** trên 7 nhóm chiếu `q, k, v, o, gate, up, down_proj`.
  * Hai adapter `default` (Student) và `teacher` (EMA `α = 0.01`) **dùng chung 1 base model lượng tử hóa** trong VRAM; `batch_size = 16`, `GEN_BS = 8`, `max_new_tokens = 1024`, `artifact_mask_tokens = 3`.
* [1] I. Shenfeld et al., "Self-Distillation Enables Continual Learning," arXiv:2601.19897v2, Sec. 4.4 & Appendix B.1, 2026.
* [17] E. J. Hu et al., "LoRA: Low-rank adaptation of large language models," ICLR, 2022.
* [18] T. Dettmers et al., "QLoRA: Efficient finetuning of quantized LLMs," NeurIPS, 2023.
* **Speaker Notes:** Tuân thủ quy định chọn phạm vi của môn học, nhóm lựa chọn Core Scope và giải trình rõ lý do kỹ thuật. Về mặt ngân sách, việc tái hiện toàn bộ lưới 54 siêu tham số bằng Full Fine-Tuning trên 7B hoặc gọi API GPT-5-mini nằm ngoài khả năng phần cứng miễn phí, đồng thời dữ liệu Wikipedia 2025 chưa được tác giả công bố. Do đó, Core Scope của nhóm tập trung vào hai bài toán cốt lõi có bộ chấm điểm tự động xác định là Science Q&A (E1) và Tool Use (E2) trên đúng mô hình 7B, cộng thêm một phân tích tự thiết kế (E3) khảo sát kích thước mô hình 1.5B, 3B và 7B để kiểm chứng sự đảo dấu giữa 3B và 7B. Để chạy vừa 16GB VRAM của GPU Kaggle T4, nhóm sử dụng kỹ thuật QLoRA 4-bit trên toàn bộ các lớp tuyến tính (all-linear, rank 32) và thiết kế hai adapter LoRA chia sẻ chung một mô hình nền. Dựa trên phiên chạy thử nghiệm (pilot) đo tốc độ thực tế trên T4 (50.6 giây/mẫu cho Science và 33.4 giây/mẫu cho Tool Use), nhóm chốt kích thước tập con N_SCI = 352 và N_TOOL = 528 để khớp giới hạn 10 giờ/phiên của Kaggle.

---

## Slide 21 — Kiến trúc Code Tự cài đặt & Kỷ luật Tracking W&B
* **Tiêu đề:** KIẾN TRÚC CÀI ĐẶT ĐỘC LẬP & THEO DÕI W&B
* **Khối trái — Tách bạch Tự cài đặt vs. Thư viện tái sử dụng:**
  * **Phần lõi nhóm tự cài đặt từ mô tả paper:**
    * `src/sdft_trainer.py`: Vòng lặp rollout on-policy theo batch, ghép prompt Teacher, tính Analytic Forward/Reverse KL trên từ vựng `V`, mask 3 token đầu, cập nhật EMA adapter, lưu resume mỗi 10 bước (`--max_hours 11`).
    * `src/answer_parsing.py` & `src/evaluate.py`: Trích xuất thẻ `<answer>`, phân tích JSON lồng nhau cho Tool Use.
  * **Phần phụ trợ tái sử dụng (có trích dẫn):** `torch`, `transformers`, `peft`, `bitsandbytes`, `datasets`, `lm-eval`, `wandb`.
  * **Dữ liệu tự động hóa:** Mirror từ dữ liệu chính thức của tác giả lên Hugging Face Hub và tải programmatic trong `src/data_loader.py` (`seed = 42`).
* **Khối phải — Bảng ánh xạ W&B Runs (Public Project `sdft-continual-learning`):**

| Job / Thí nghiệm | Script chạy | Cấu hình chính | Link W&B Run |
| :--- | :--- | :--- | :--- |
| `pilot` (Đo tốc độ T4) | `scripts/kaggle_jobs.sh` | 7B QLoRA, 24 mẫu/task | `cf7a0be7` (Sci) / `68998c9c` (Tool) |
| `e1_sft_base` (Science Base+SFT) | `scripts/kaggle_jobs.sh` | 7B, `N_SCI=352`, 1 epoch | `[Dán link W&B]` |
| `e1_sdft` (Science SDFT — TK A) | `scripts/kaggle_jobs.sh` | 7B, `N_SCI=352`, 2 epochs | `[Dán link W&B]` |
| `e2_sft` & `e2_sdft` (Tool Use) | `scripts/kaggle_jobs.sh` | 7B, `N_TOOL=528` | `[Dán link W&B]` |
| `e3_small` (1.5B / 3B Science) | `scripts/kaggle_jobs.sh` | `1.5B` & `3B`, `N_SCI=352` | `[Dán link W&B]` |

* **Kho mã nguồn công khai của nhóm:** `https://github.com/24122040-kin/distillation_enables_continual_learning_hcmus`
* [1] I. Shenfeld et al., "Self-Distillation Enables Continual Learning," arXiv:2601.19897v2, Algorithm 1 & Appendix A.1, 2026.
* [20] I. Shenfeld, "Self-Distillation" (chỉ dùng đối chiếu), GitHub: `idanshen/Self-Distillation`, 2026.
* **Speaker Notes:** Nhằm tuân thủ tuyệt đối quy tắc liêm chính của môn học, phần lõi của SDFT được nhóm tự viết hoàn toàn bằng PyTorch dựa trên Thuật toán 1 và Phụ lục A.1 của bài báo, chỉ dùng kho mã nguồn của tác giả để đối chiếu và xác minh sau cùng. Các thư viện chuẩn như Transformers, PEFT, BitsAndBytes và lm-eval được dùng cho tầng hạ tầng và đều được liệt kê minh bạch trong README. Dữ liệu chính thức của bài báo được lưu trữ trên Hugging Face và tải tự động trong lúc chạy. Mỗi bước thực nghiệm được điều phối qua một điểm vào duy nhất là scripts/kaggle_jobs.sh và ghi log đồng bộ lên Weights & Biases, giúp giảng viên truy vết lại toàn bộ đường cong mất mát, thời gian sinh chuỗi và siêu tham số của từng con số xuất hiện trên slide.

---

## Slide 22 — Kết quả Tái tạo: Độ chính xác Tác vụ mới (E1 & E2)
* **Tiêu đề:** KẾT QUẢ TÁI TẠO: ĐỘ CHÍNH XÁC TÁC VỤ MỚI (E1 & E2)
* **Bảng đối chiếu định lượng (từ `eval_results/test-*.json`, greedy `temperature=0`, `max_new_tokens=1024`):**

| Tác vụ / Phương pháp | Paper (%) | Nhóm tái tạo (%) | Sai lệch Δ (pp) | Sai số tương đối (%) | W&B Run Link / ID |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Science (E1):** Base (`7B`) | 32.1 | `[...]` | `[...]` | `[...]` | `[Link W&B]` |
| **Science (E1):** SFT (`best ep`) | 66.2 | `[...]` | `[...]` | `[...]` | `[Link W&B]` |
| **Science (E1):** SDFT (`best ep`) | **70.2** | `[...]` | `[...]` | `[...]` | `[Link W&B]` |
| **Tool Use (E2):** Base (`7B`) | 42.9 | `[...]` | `[...]` | `[...]` | `[Link W&B]` |
| **Tool Use (E2):** SFT (`best ep`) | 63.2 | `[...]` | `[...]` | `[...]` | `[Link W&B]` |
| **Tool Use (E2):** SDFT (`best ep`) | **70.6** | `[...]` | `[...]` | `[...]` | `[Link W&B]` |

* **Cách chấm điểm khớp chuẩn paper:** Science Q&A khớp chính xác chữ cái trong thẻ `<answer>...</answer>`; Tool Use khớp tập tên `Action` và JSON tham số (không phụ thuộc thứ tự khóa).
* **Nhận xét sau chạy (Điền khi có kết quả):** Kiểm tra xem xu hướng `SDFT > SFT > Base` có được bảo toàn với `N_SCI = 352` và `N_TOOL = 528` hay không.
* [1] I. Shenfeld et al., "Self-Distillation Enables Continual Learning," arXiv:2601.19897v2, Table 5, 2026.
* **Speaker Notes:** Slide này đặt kết quả tái tạo của nhóm cho hai thí nghiệm trọng tâm E1 (Science Q&A, tập test 507 câu) và E2 (Tool Use, tập test 97 câu) nằm song song với các con số công bố của bài báo. Để đảm bảo đánh giá công bằng, nhóm áp dụng đúng quy trình chọn epoch tốt nhất trên tập validation 5% trước khi đánh giá trên tập kiểm thử với greedy decoding và độ dài sinh tối đa 1024 token. Cách chấm điểm cũng tuân thủ chặt chẽ bài báo: phân tích thẻ đáp án cho trắc nghiệm Hóa học và giải mã JSON thực thụ cho các tham số lồng nhau trong lời gọi API ToolAlpaca. Mỗi con số tái tạo đều đi kèm độ lệch tuyệt đối tính theo điểm phần trăm (Delta), sai số tương đối tính theo phần trăm và đường dẫn tới W&B run tương ứng.

---

## Slide 23 — Kết quả Tái tạo: Đo lường Quên trên Đủ 6 Benchmarks (E1 & E2)
* **Tiêu đề:** KẾT QUẢ TÁI TẠO: ĐÁNH GIÁ QUÊN TRÊN 6 BENCHMARKS (`lm-eval`)
* **Bảng đối chiếu Đủ 6 Benchmarks (`--limit 0.25` trên Kaggle T4, kèm `Prior Avg` & Tỷ lệ bảo toàn):**

| Mô hình / Tác vụ | HellaSwag | HumanEval | IFEval | MMLU | TruthfulQA | Winogrande | Prior Avg (Paper → Team) | W&B Run ID |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Base (`7B`)** | 62.0 → `[...]` | 65.8 → `[...]` | 74.3 → `[...]` | 71.7 → `[...]` | 47.9 → `[...]` | 71.1 → `[...]` | 65.5 → `[...]` | `[W&B ID]` |
| **SFT (E1: Sci)** | 55.0 → `[...]` | 54.8 → `[...]` | 35.3 → `[...]` | 64.6 → `[...]` | 36.8 → `[...]` | 73.7 → `[...]` | 53.4 → `[...]` | `[W&B ID]` |
| **SDFT (E1: Sci)** | 60.9 → `[...]` | 68.9 → `[...]` | 66.8 → `[...]` | 70.7 → `[...]` | 46.5 → `[...]` | 73.1 → `[...]` | 64.5 → `[...]` | `[W&B ID]` |
| **SFT (E2: Tool)** | 57.3 → `[...]` | 50.0 → `[...]` | 49.8 → `[...]` | 70.2 → `[...]` | 37.5 → `[...]` | 73.1 → `[...]` | 56.0 → `[...]` | `[W&B ID]` |
| **SDFT (E2: Tool)** | 61.6 → `[...]` | 68.3 → `[...]` | 71.9 → `[...]` | 71.5 → `[...]` | 47.3 → `[...]` | 71.7 → `[...]` | 65.4 → `[...]` | `[W&B ID]` |

* **Ghi chú:** Đánh giá đủ cả 6 bộ kiểm tra qua `lm-evaluation-harness` (`--apply_chat_template`, `--limit 0.25` để khớp giới hạn giờ T4) và log tự động lên W&B (`prior-...`).
* [1] I. Shenfeld et al., "Self-Distillation Enables Continual Learning," arXiv:2601.19897v2, Table 5, 2026.
* [14] L. Gao et al., "The language model evaluation harness," Zenodo, 2024.
* **Speaker Notes:** Một trong những đóng góp quan trọng nhất cần tái kiểm chứng là khả năng chống quên kiến thức cũ của SDFT. Nhóm chạy bộ công cụ lm-evaluation-harness đầy đủ trên toàn bộ 6 benchmark giống hệt bài báo (HellaSwag, HumanEval, IFEval, MMLU, TruthfulQA và Winogrande) đối với Base model và các checkpoint tốt nhất của SFT, SDFT trên cả E1 và E2. Do tổng thời gian đánh giá toàn bộ 6 bộ benchmark trên T4 mất khoảng 4–5 giờ mỗi checkpoint, nhóm cấu hình tỷ lệ lấy mẫu `--limit 0.25` (25% số câu mỗi bộ) và ghi rõ sự đánh đổi này trong phần phân tích sai lệch. Khi đối chiếu, nhóm đặc biệt chú ý hai yếu tố: thứ nhất là khoảng cách quên giữa SFT và SDFT có giữ đúng chiều hướng (đặc biệt trên IFEval và HumanEval vốn bị SFT làm tụt mạnh nhất), và thứ hai là tác động giữ trọng số nền tự nhiên của QLoRA khiến mức sụt giảm của SFT có thể nhẹ hơn so với Full Fine-Tuning.

---

## Slide 24 — Kết quả Tái tạo: Phân tích E3 — Hiệu ứng Kích thước Mô hình (`1.5B / 3B / 7B`)
* **Tiêu đề:** PHÂN TÍCH BỔ SUNG (E3): KÍCH THƯỚC MÔ HÌNH & GIẢ ĐỊNH ICL
* **Bảng đối chiếu trên Science Q&A (Kiểm chứng kết quả `3B` vs `7B` của paper):**

| Kích thước Model | Base (%) | Teacher (`--with_demo`, %) | SFT (%) | SDFT (%) | Hiệu số (SDFT − SFT) | Paper báo cáo | W&B Run ID |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `Qwen2.5-1.5B` | `[...]` | `[...]` | `[...]` | `[...]` | `[...]` | — | `[W&B ID]` |
| `Qwen2.5-3B` | `[...]` | `[...]` | `[...]` | `[...]` | `[...]` | **−3.3 pp** | `[W&B ID]` |
| `Qwen2.5-7B` (E1) | `[...]` | `[...]` | `[...]` | `[...]` | `[...]` | **+4.0 pp** | `[W&B ID]` |

* **Giá trị của thí nghiệm tự thiết kế (E3):**
  * Không chỉ lặp lại `SFT` và `SDFT` ở `1.5B / 3B / 7B`, nhóm còn đo thêm **độ chính xác trực tiếp của Teacher** (`--with_demo` trên tập validation).
  * Giúp kiểm chứng nhân quả: Liệu sự đảo dấu từ âm (`3B`) sang dương (`7B`) có bắt nguồn trực tiếp từ việc Teacher nhỏ không đủ năng lực ICL để sinh tín hiệu đúng hay không?
* [1] I. Shenfeld et al., "Self-Distillation Enables Continual Learning," arXiv:2601.19897v2, Sec. 4.4 & Fig. 5 (Left), 2026.
* [19] A. Yang et al. (Qwen Team), "Qwen2.5 technical report," arXiv:2412.15115, 2024.
* **Speaker Notes:** Thay thế cho các phần bị lược bỏ trong Core Scope, nhóm thực hiện thí nghiệm E3 khảo sát ảnh hưởng của quy mô mô hình trên Science Q&A qua ba mức 1.5B, 3B và 7B. Kết quả của thí nghiệm này rất thú vị vì trong bài báo, tác giả báo cáo ở cỡ 3B thì SDFT thua SFT tới 3.3 điểm phần trăm, và chỉ đảo dấu thắng lợi +4.0 điểm ở 7B và +6.9 điểm ở 14B. Để kiểm tra sâu hơn nguyên nhân đằng sau hiện tượng này thay vì chỉ quan sát điểm cuối cùng, nhóm bổ sung phép đo trực tiếp độ chính xác của Teacher khi được cấp ví dụ mẫu trong ngữ cảnh (--with_demo) trên tập validation cho từng kích thước. Nếu Teacher của 1.5B và 3B đạt độ chính xác thấp dù đã nhìn thấy đáp án mẫu, ta có bằng chứng định lượng trực tiếp khẳng định giả định In-Context bị gãy ở mô hình nhỏ, nối liền kết quả tái tạo sang phần phân tích phê bình cho Poster cuối kỳ.

---

## Slide 25 — Phân tích Sai số (Deviation Analysis — Phần 1: Phần cứng & Thuật toán)
* **Tiêu đề:** PHÂN TÍCH SAI SỐ (DEVIATION ANALYSIS — 1/2)
* **Bảng phân rã: Sai lệch → Nguyên nhân → Hướng tác động kỳ vọng/thực tế → Bằng chứng:**
  1. **LoRA (`r=32 all-linear`) thay cho Full Fine-Tuning:**
     * *Nguyên nhân:* Giới hạn 16GB VRAM của Kaggle T4 không thể lưu optimizer Full FT cho 7B.
     * *Tác động:* LoRA đóng băng toàn bộ trọng số gốc nên **giảm quên tự nhiên cho cả SFT** → khoảng cách `Prior Avg` giữa SFT và SDFT bị thu hẹp lại so với Full FT; đồng thời trần độ chính xác tác vụ mới có thể thấp hơn vài điểm.
  2. **Xấp xỉ EMA trên hai ma trận LoRA riêng rẽ:**
     * *Nguyên nhân:* Cập nhật EMA độc lập lên `A` và `B`, trong khi trọng số thích nghi thực tế là tích `ΔW = B·A`.
     * *Tác động toán học:* `EMA(B) · EMA(A) ≠ EMA(B·A)` (tích của hai trung bình trượt khác trung bình trượt của tích) → Teacher chỉ là xấp xỉ gần đúng của EMA trọng số đầy đủ.
  3. **Lượng tử hóa 4-bit NF4 + tính toán `fp16` (thay cho `bfloat16`):**
     * *Tác động:* Làm nhiễu nhẹ phân phối logits trên 152k tokens và Base accuracy có thể xê dịch ±1–3%.
* [1] I. Shenfeld et al., "Self-Distillation Enables Continual Learning," arXiv:2601.19897v2, Appendix B.1 & Table 3, 2026.
* [17] E. J. Hu et al., ICLR, 2022. [18] T. Dettmers et al., NeurIPS, 2023.
* **Speaker Notes:** Theo hướng dẫn của môn học, phần phân tích sai số là nơi giải thích mọi chênh lệch giữa bản tái tạo và số liệu công bố dựa trên cơ chế thuật toán và điều kiện phần cứng. Ba yếu tố đầu tiên xuất phát trực tiếp từ thích nghi 4-bit QLoRA. Thứ nhất, việc chỉ cập nhật không gian con hạng thấp LoRA và giữ nguyên trọng số nền giúp bản thân mô hình SFT không bị phá hủy nhiều biểu diễn cũ như khi Full Fine-Tuning trên H200, khiến khoảng cách chống quên giữa SDFT và SFT có xu hướng co lại. Thứ hai, một sai lệch tinh tế về mặt đại số mà nhóm nhận ra khi tự cài đặt là việc áp EMA riêng cho từng ma trận LoRA A và B không bằng về mặt toán học với việc áp EMA lên ma trận tích delta W = BA; đây là phép xấp xỉ bắt buộc để tiết kiệm VRAM. Thứ ba, tính toán bằng fp16 và lượng tử hóa NF4 4-bit thay cho bfloat16 gốc có thể gây sai số làm tròn nhỏ trên phân phối xác suất đầu ra.

---

## Slide 26 — Phân tích Sai số (Deviation Analysis — Phần 2: Dữ liệu, Tốc độ & Sweep)
* **Tiêu đề:** PHÂN TÍCH SAI SỐ (DEVIATION ANALYSIS — 2/2)
* **3 nguồn sai số tiếp theo từ ngân sách thời gian & giao thức:**
  4. **Cắt tập con Train (`N_SCI=352`, `N_TOOL=528`) & Lấy mẫu `lm-eval` (`--limit 0.25`):**
     * *Đo thực tế từ Pilot TK A:* T4 chạy **`50.6 s/mẫu`** (Science) và **`33.4 s/mẫu`** (Tool Use) → 2 epochs đủ dữ liệu cần **71.4 giờ**, vượt xa giới hạn 12h/phiên của Kaggle.
     * *Tác động:* Tập train ít bước cập nhật hơn (22–33 steps/epoch với batch 16) nên mô hình có thể chưa hội tụ tới đỉnh như khi train đủ `2.674` / `4.046` mẫu; `lm-eval` lấy mẫu 25% có sai số chọn mẫu nhỏ.
  5. **Wall-clock Ratio (`~11×` trên T4 vs `~4×` trên H200+vLLM):**
     * Do thiếu engine vLLM tương thích tốt với QLoRA 4-bit trên T4, bước sinh rollout chiếm ~60–75% thời gian (`gen=192–262s`, `train=131–156s` trong pilot Science `cf7a0be7`), làm chi phí thời gian thực gấp ~11 lần SFT.
  6. **Không quét lưới 54 cấu hình & Chạy 1 Seed (`seed = 42`):**
     * Tác giả sweep LR/batch/epoch/alpha trên 3 seeds; nhóm chọn bộ siêu tham số neo (LR `1e-4` phù hợp LoRA, `α = 0.01`, `seed = 42`) nên có phương sai khởi tạo ngẫu nhiên.
* [1] I. Shenfeld et al., "Self-Distillation Enables Continual Learning," arXiv:2601.19897v2, Sec. 5 & Appendix B.2, 2026.
* **Speaker Notes:** Hai nguồn sai số tiếp theo đến từ giới hạn thời gian và chi phí tính toán. Từ phiên chạy đo tốc độ (pilot) trên GPU Kaggle T4 của Tài khoản A, chúng tôi đo được tốc độ thực tế là 50.6 giây mỗi mẫu ở Science Q&A và 33.4 giây mỗi mẫu ở Tool Use. Điều này có nghĩa để chạy đủ 2 epochs toàn bộ dữ liệu gốc sẽ tốn tới 71.4 giờ cho mỗi tác vụ, đồng thời làm tỷ lệ thời gian chạy SDFT trên SFT vọt lên khoảng 11 lần thay vì 4 lần như khi tác giả dùng H200 kết hợp vLLM. Để hoàn thành thực nghiệm trong phiên chạy tối đa 11 giờ của Kaggle, nhóm buộc phải lấy tập con lồng nhau N_SCI = 352 và N_TOOL = 528, dẫn tới số bước cập nhật gradient ít hơn. Cuối cùng, thay vì quét hàng chục tổ hợp siêu tham số qua 3 random seeds, nhóm chạy một cấu hình mục tiêu cố định ở seed 42 và dùng tập validation 5% chọn epoch tốt nhất. Toàn bộ con số thời gian và kích thước tập con này đều được log trên W&B để định lượng chính xác ảnh hưởng.

---

## Slide 27 — Đối chiếu Code gốc (`idanshen/Self-Distillation`)
* **Tiêu đề:** ĐỐI CHIẾU PAPER VS. CODE GỐC VS. BẢN TÁI TẠO CỦA NHÓM
* **Bảng so sánh 6 điểm bất nhất phát hiện được khi kiểm chứng code chính thức:**

| Khía cạnh | Paper công bố | Code chính thức (`idanshen/Self-Distillation`) | Bản tự cài đặt của nhóm |
| :--- | :--- | :--- | :--- |
| **1. Hàm KL Loss** | Lý thuyết (Eq. 1) dùng **Reverse KL** | README xác nhận mọi kết quả dùng **Forward KL** | Hỗ trợ cả `forward` (chính) & `reverse` |
| **2. Độ dài sinh tối đa** | Table 3 ghi **`2048`** tokens | Thực tế code đặt **`1024`** tokens | **`1024`** tokens |
| **3. Learning Rate** | Sweep `{5e-6, 1e-5, 5e-5}` | Mặc định `2e-5`, README dùng `5e-5` | `1e-4` (điều chỉnh cho LoRA) |
| **4. Warmup** | Table 3 ghi **`10` bước** cố định | Code dùng **`10%` tổng số bước** | `10` bước |
| **5. Rollout & IS** | Thuật toán 1 chỉ ghi *"Nếu cần, thêm IS"* | Dùng **vLLM** + Importance Sampling bù lệch | HF batched `generate()` thống nhất (không cần IS) |
| **6. Prompt Science** | Section 3 trình bày prompt 3 dòng chung | Thêm **system prompt** ép `<reasoning>`/`<answer>` | Khớp system prompt & parser thẻ `<answer>` |

* [1] I. Shenfeld et al., "Self-Distillation Enables Continual Learning," arXiv:2601.19897v2, 2026.
* [20] I. Shenfeld, "Self-Distillation" (chỉ dùng đối chiếu & xác minh), GitHub: `idanshen/Self-Distillation`, 2026.
* **Speaker Notes:** Một yêu cầu quan trọng của Lab 1 là đối chiếu bản tự cài đặt của nhóm với mã nguồn do tác giả phát hành (tuyệt đối không sao chép) để tìm ra các khác biệt ẩn. Qua kiểm tra kho mã nguồn idanshen/Self-Distillation, nhóm ghi nhận sáu điểm lệch đáng chú ý giữa bài báo in và code chạy thật. Thứ nhất và quan trọng nhất, dù phần chứng minh lý thuyết dùng Reverse KL, file README chính thức xác nhận mọi kết quả đều chạy bằng Forward KL theo từng token. Thứ hai, độ dài sinh tối đa thực tế là 1024 chứ không phải 2048 như Table 3. Thứ ba và tư là sự chênh lệch về learning rate mặc định và warmup (10% số bước thay vì 10 bước). Thứ năm, do code gốc dùng engine vLLM tách rời cho pha sinh và DeepSpeed/TRL cho pha train, nó cần thêm bước Importance Sampling (IS) để bù sai số số học giữa hai engine; trong khi bản PyTorch thống nhất của nhóm dùng cùng một mô hình nên không phát sinh lỗi lệch phân phối này. Thứ sáu, tác vụ Science trong code gốc còn chèn thêm system prompt bắt buộc xuất thẻ XML `<reasoning>` và `<answer>` mà bài báo chưa mô tả chi tiết.

---

## Slide 28 — Tổng kết các Điểm mạnh đã Kiểm chứng
* **Tiêu đề:** TỔNG KẾT CÁC ĐIỂM MẠNH CỦA PHƯƠNG PHÁP SDFT
* **3 điểm mạnh cốt lõi (gắn với bằng chứng lý thuyết & thực nghiệm):**
  1. **Học On-policy không cần Hàm thưởng tường minh:** Khai thác trực tiếp In-Context Learning để chuyển dữ liệu mẫu thành phần thưởng nội tại ở mức token (`log π_teacher − log π_student`), mở rộng phạm vi on-policy sang y khoa, API và nạp tri thức.
  2. **Cân bằng xuất sắc giữa Học mới và Bảo toàn Năng lực cũ:** Dẫn đầu biên Pareto trên 3 tác vụ kỹ năng (giữ `64.5–65.4%` trên 6 benchmark nền tảng so với Base `65.5%`) và cho phép học nối tiếp nhiều tác vụ mà không thoái lui như SFT.
  3. **Tổng quát hóa tri thức sâu (OOD) & Giữ chiều sâu Suy luận:** Đạt `98%` trên câu hỏi suy luận gián tiếp OOD (so với `80%` của SFT) và tránh sụp đổ độ dài suy luận trên mô hình Reasoning ngay cả khi chỉ có đáp án cuối.
* [1] I. Shenfeld et al., "Self-Distillation Enables Continual Learning," arXiv:2601.19897v2, Sec. 4–5, 2026.
* **Speaker Notes:** Tổng hợp lại toàn bộ giai đoạn đọc hiểu và tái tạo, nhóm đúc kết ba điểm mạnh có cơ sở vững chắc của SDFT. Điểm mạnh thứ nhất là tính thực dụng: SDFT bắc cầu thành công giữa dữ liệu mẫu phổ biến và cơ chế học on-policy vốn trước đây bị khóa chặt bởi yêu cầu phải có hàm thưởng tường minh hoặc dữ liệu cặp ưa thích. Điểm mạnh thứ hai là hiệu quả chống quên thảm khốc đã được chứng minh trên nhiều miền dữ liệu khác nhau, giúp mô hình vừa tiếp thu kỹ năng mới cao hơn SFT, vừa giữ lại gần như nguyên vẹn điểm số trên cả sáu bộ đánh giá năng lực tổng quát và hỗ trợ huấn luyện tuần tự. Điểm mạnh thứ ba là khả năng tổng quát hóa vượt khỏi việc học thuộc bề mặt: tỷ lệ chính xác 98% ở câu hỏi gián tiếp ngoài phân phối và việc bảo toàn chuỗi suy nghĩ trên mô hình reasoning từ dữ liệu chỉ có đáp án cuối là những lợi thế mà SFT truyền thống khó đạt được.

---

## Slide 29 — Cầu nối sang Phê bình Lý thuyết & Poster A2 (Checkpoint 2)
* **Tiêu đề:** CÂU HỎI MỞ & HƯỚNG PHÊ BÌNH LÝ THUYẾT CHO POSTER
* **3 điểm yếu lý thuyết nhóm đã nhận diện để thiết kế thí nghiệm kiểm chứng ở Lab 2:**
  1. **Khoảng cách Lý thuyết (Reverse KL) – Thực tế (Forward KL & Bộ ước lượng có độ chệch):**
     * Chứng minh tương đương Inverse RL chỉ đúng với Reverse KL cấp chuỗi; còn code thực tế dùng **Forward KL** kết hợp bộ ước lượng theo token bỏ qua tác động tới các token tương lai `y_>t`.
     * *Thí nghiệm đề xuất:* Chạy đối chiếu trực tiếp `forward` vs `reverse` KL bằng tham số `--kl_mode` đã cài đặt sẵn.
  2. **Sự mong manh của Giả định In-Context (`π* ≈ π(·|x, c)`):**
     * Giả định thất bại ở mô hình nhỏ (`3B` thua SFT `-3.3 pp`) và giả định ngầm rằng mẫu `c` luôn đúng, sạch.
     * *Thí nghiệm đề xuất:* Đưa dữ liệu mẫu bị nhiễu/sai một phần để vẽ đường cong suy giảm của Teacher và Student.
  3. **Chi phí Wall-clock thực tế & Cơ chế Heuristic che Artifact:**
     * Tốn `~4×` trên H200 nhưng lên tới **`~11×`** trên phần cứng phổ thông (T4), cộng thêm mẹo mask 3 token đầu chưa có cơ sở lý thuyết chặt chẽ.
* [1] I. Shenfeld et al., "Self-Distillation Enables Continual Learning," arXiv:2601.19897v2, Sec. 3.1, 4.4, 5 & Appendix A.1, 2026.
* **Speaker Notes:** Những phát hiện trong quá trình đọc sâu và tự cài đặt SDFT mở ra ba trục phê bình lý thuyết sắc bén cho giai đoạn Poster Workshop tiếp theo. Trục thứ nhất là sự bất nhất giữa lý thuyết và thực thi: phần chứng minh SDFT tương đương Inverse RL hoàn toàn dựa vào Reverse KL và đạo hàm cấp chuỗi không chệch, nhưng thuật toán chạy thực tế lại đổi sang Forward KL và dùng bộ ước lượng giải tích theo từng token vốn bỏ qua ảnh hưởng lên các bước tương lai. Nhờ cài sẵn tùy chọn chuyển đổi hai chế độ KL trong mã nguồn, nhóm sẽ kiểm chứng định lượng sự khác biệt này ở Lab 2. Trục thứ hai là tính dễ tổn thương của giả định In-Context: không chỉ thất bại ở mô hình nhỏ, SDFT còn có thể sụp đổ nếu dữ liệu mẫu chứa nhiễu hoặc sai lệch định dạng. Trục thứ ba là bài toán đánh đổi chi phí thực tế gấp 11 lần trên GPU tầm trung và tính tạm bợ của kỹ thuật mask 3 token đầu nhằm che giấu cụm từ quy chiếu ngữ cảnh của Teacher.

---

## Slide 30 — Kết luận & Tài nguyên Dự án
* **Tiêu đề:** KẾT LUẬN & TÀI NGUYÊN CÔNG KHAI
* **Tóm tắt thành quả Checkpoint 1 (Reading + Reproduction):**
  * Hoàn thành giải phẫu trọn vẹn 5 lớp phương pháp SDFT và viết lại hoàn toàn bằng ngôn ngữ của nhóm (tự đứng độc lập không cần mở bài báo).
  * Tự triển khai độc lập SDFT + QLoRA 4-bit trong **Core Scope** (E1 Science Q&A, E2 Tool Use, E3 Model Size + 6 benchmarks `lm-eval`).
  * Đo lường định lượng sai lệch so với bài báo gốc, phát hiện 6 điểm khác biệt ở code chính thức và xác định 3 hướng kiểm chứng cho Poster A2.
* **Liên kết kiểm chứng công khai:**
  * **GitHub Repository:** `https://github.com/24122040-kin/distillation_enables_continual_learning_hcmus`
  * **Weights & Biases Project:** `[Dán link W&B Project công khai]`
* **Speaker Notes:** Tóm lại, thông qua Milestone Reading và Lab 1 Reproduction, nhóm đã nắm vững cả cơ sở toán học lẫn các chi tiết kỹ thuật ẩn của thuật toán Self-Distillation Fine-Tuning (SDFT). Toàn bộ mã nguồn tự cài đặt, kịch bản tải dữ liệu tự động từ Hugging Face, các kiểm thử đơn vị cho bộ phân tích đáp án và nhật ký chạy thực nghiệm trên Weights & Biases đều được công khai đầy đủ tại kho GitHub của nhóm. Trong giai đoạn tiếp theo, nhóm sẽ khai thác tiếp các lỗ hổng lý thuyết đã chỉ ra — đặc biệt là khoảng cách giữa Reverse KL và Forward KL cùng độ nhạy của giả định In-Context trước dữ liệu nhiễu — để hoàn thành bài phân tích và trình bày trên Poster khổ A2 tại buổi bảo vệ cuối kỳ. Xin cảm ơn thầy và hội đồng đã theo dõi.
