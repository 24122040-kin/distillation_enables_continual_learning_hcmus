# Hướng dẫn lấy dữ liệu từ Hugging Face (`Baokin/sdft-official-data`)

Tài liệu hướng dẫn ngắn để nhóm sử dụng thống nhất tập dữ liệu đã mirror trên Hugging Face thay cho tải trực tiếp từ GitHub.

## 1. Thông tin dataset repo trên Hugging Face

- **Repo ID:** `Baokin/sdft-official-data`
- **URL:** https://huggingface.co/datasets/Baokin/sdft-official-data
- **Quyền truy cập:** **Public** — tất cả tài khoản Kaggle và máy cá nhân có thể tải trực tiếp qua `datasets` mà **không cần `HF_TOKEN`**.
- **Nguồn gốc dữ liệu:** Bản sao nguyên vẹn dữ liệu do tác giả bài báo *Self-Distillation Enables Continual Learning* (Shenfeld et al., 2026, `arXiv:2601.19897`) phát hành tại `github.com/idanshen/Self-Distillation` commit `d77573212fa0a3ae2eeb64b9b44db1c251f75e3e`.
- **Các config & kích thước split:**

| Config | Task trong `src/data_loader.py` | Split `train` | Split `test` | Nguồn gốc |
|---|---|---|---|---|
| `science` | `"science"` | 2.674 | 507 | SciKnowEval Chem L-3 + GPT-4o demonstrations |
| `tooluse` | `"tool_use"` | 4.046 | 97 | ToolAlpaca (split gốc của tác giả) |

## 2. Cấu trúc file & xử lý schema Parquet

Trên `Baokin/sdft-official-data` (`main`) gồm các file:
- `README.md` (khai báo YAML `configs` trỏ tới 4 file Parquet)
- `science-train.parquet`
- `science-test.parquet`
- `tooluse-train.parquet`
- `tooluse-test.parquet`

**Lưu ý kỹ thuật về schema (không cần sửa `src/data_loader.py`):**
- Dữ liệu gốc của tác giả có cột khác nhau giữa `train` và `test` (`science/train` chỉ có `messages`, `output_text` còn `science/test` chỉ có `prompt`, `answer`; `tooluse/train` có `golden_response` còn `tooluse/test` không có).
- Thư viện `datasets==4.3.0` yêu cầu mọi split trong cùng một config phải có chung Arrow schema. Vì vậy khi xuất Parquet lên Hugging Face, các cột chỉ thuộc về split kia được pad thêm giá trị `null` đúng kiểu Arrow, giữ nguyên 100% toàn bộ nội dung gốc.
- `src/data_loader.py` (`format_split`) chỉ đọc đúng các cột hợp lệ của từng split, nên đọc thẳng từ `Baokin/sdft-official-data` cho ra kết quả **khớp 100%** với dữ liệu gốc trên GitHub mà **không cần chỉnh sửa code trong `src/`**.

## 3. Ba cách sử dụng trong code và Kaggle

### Cách 1: Dùng biến môi trường `SDFT_HF_DATASET` (khuyến nghị)
`scripts/kaggle_jobs.sh` và Ô 1 trong `docs/huong_dan_tai_khoan_A.md` đã đặt sẵn:

```bash
export SDFT_HF_DATASET=Baokin/sdft-official-data
```

Khi biến này được đặt (hoặc trong Python có `os.environ["SDFT_HF_DATASET"] = "Baokin/sdft-official-data"`), mọi script (`sdft_trainer.py`, `sft_baseline.py`, `evaluate.py`) ở chế độ mặc định `--data_source auto` sẽ tự động tải từ Hugging Face.

### Cách 2: Truyền trực tiếp qua cờ CLI
```bash
python src/sdft_trainer.py --data_source hf --hf_dataset Baokin/sdft-official-data ...
python src/sft_baseline.py --data_source hf --hf_dataset Baokin/sdft-official-data ...
python src/evaluate.py     --data_source hf --hf_dataset Baokin/sdft-official-data ...
```

### Cách 3: Tải trực tiếp trong Python

```python
# Qua data_loader của dự án (tự tách 5% validation cố định theo data_seed=0)
from data_loader import get_dataset

train, val, test = get_dataset("science", source="hf", hf_dataset="Baokin/sdft-official-data")
# train=2540, val=134, test=507 (science)
# train=3844, val=202, test=97  (tool_use)
```

Hoặc đọc raw split bằng `datasets`:

```python
from datasets import load_dataset

sci_train = load_dataset("Baokin/sdft-official-data", "science", split="train")
sci_test = load_dataset("Baokin/sdft-official-data", "science", split="test")
tool_train = load_dataset("Baokin/sdft-official-data", "tooluse", split="train")
tool_test = load_dataset("Baokin/sdft-official-data", "tooluse", split="test")
```

## 4. Lệnh kiểm tra đối chiếu nhanh HF vs GitHub

Chạy tại thư mục gốc repo để xác nhận dữ liệu tải từ Hugging Face khớp hoàn toàn với GitHub:

```bash
python -c '
import sys; sys.path.insert(0, "src")
from data_loader import get_dataset
for task in ("science", "tool_use"):
    assert get_dataset(task, source="hf", hf_dataset="Baokin/sdft-official-data") == get_dataset(task, source="github")
    print(f"OK: {task} HF == GitHub")
'
```
