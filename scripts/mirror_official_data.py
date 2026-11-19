"""Mirror the authors' released datasets to the team's Hugging Face account.

The course requires experiments to read their data from Hugging Face or Kaggle.
The SDFT authors publish their data only inside their GitHub repository
(``idanshen/Self-Distillation``, ``data/``), so we copy it, unchanged, into one
HF dataset repo with two configurations::

    <hf_repo>  config "science"  splits train (2674) / test (507)
    <hf_repo>  config "tooluse"  splits train (4046) / test (97)

Run once, from a machine logged in with ``huggingface-cli login`` (write token)::

    python scripts/mirror_official_data.py --hf_repo Baokin/sdft-official-data

Then set ``SDFT_HF_DATASET=Baokin/sdft-official-data`` for every run.
Use ``--dry_run`` to download and print the splits without uploading.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from datasets import Dataset, DatasetDict, Features  # noqa: E402

from data_loader import HF_CONFIG_NAMES, OFFICIAL_COMMIT, OFFICIAL_REPO, load_raw_split  # noqa: E402

CARD = f"""---
license: other
configs:
- config_name: science
  data_files:
  - split: train
    path: science/train-*
  - split: test
    path: science/test-*
- config_name: tooluse
  data_files:
  - split: train
    path: tooluse/train-*
  - split: test
    path: tooluse/test-*
---

# SDFT official data (mirror)

Unmodified copy of the datasets released with *Self-Distillation Enables
Continual Learning* (Shenfeld et al., 2026, arXiv:2601.19897), taken from
`github.com/{OFFICIAL_REPO}` at commit `{OFFICIAL_COMMIT}`.
Mirrored by an HCMUS student team for a course reproduction project; all
credit belongs to the original authors and to the source benchmarks
(SciKnowEval, Feng et al. 2024; ToolAlpaca, Tang et al. 2023).

Note: Split-specific missing columns (`prompt`/`answer` on `science` train,
`messages`/`output_text` on `science` test, and `golden_response` on
`tooluse` test) are padded with `null` to satisfy Hugging Face's
single-schema-per-config requirement while preserving all original values
unmodified.
"""


def unify_splits(train_ds: Dataset, test_ds: Dataset) -> tuple[Dataset, Dataset]:
    """Pad split-specific missing columns with typed nulls so train and test share one schema."""
    merged = Features({**train_ds.features, **test_ds.features})

    def pad(ds: Dataset) -> Dataset:
        for col in merged:
            if col not in ds.column_names:
                ds = ds.add_column(col, [None] * len(ds))
        return ds.select_columns(list(merged.keys())).cast(merged)

    return pad(train_ds), pad(test_ds)


def main() -> None:
    """Download both tasks from the pinned GitHub commit and push them to the Hub."""
    p = argparse.ArgumentParser()
    p.add_argument("--hf_repo", required=True)
    p.add_argument("--private", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--cache_dir", default="./data_cache")
    args = p.parse_args()

    for task, config in HF_CONFIG_NAMES.items():
        raw_train = load_raw_split(task, "train", source="github", cache_dir=args.cache_dir)
        raw_test = load_raw_split(task, "test", source="github", cache_dir=args.cache_dir)
        uni_train, uni_test = unify_splits(raw_train, raw_test)
        dd = DatasetDict({"train": uni_train, "test": uni_test})
        print(config, dd)
        if not args.dry_run:
            dd.push_to_hub(args.hf_repo, config_name=config, private=args.private)

    if not args.dry_run:
        from huggingface_hub import HfApi

        HfApi().upload_file(
            path_or_fileobj=CARD.encode(), path_in_repo="README.md",
            repo_id=args.hf_repo, repo_type="dataset",
        )
        print(f"Done. export SDFT_HF_DATASET={args.hf_repo}")


if __name__ == "__main__":
    main()
