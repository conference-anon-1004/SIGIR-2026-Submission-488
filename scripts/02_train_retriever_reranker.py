from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

import pandas as pd

from src.data_io import DatasetPaths, load_triplets_csv, select_hard_triplets
from src.conrac_backbone import (
    # retriever training
    train_biencoder_from_parquet,
    TRIPLETS_PARQUET_DEFAULT,
    # reranker training
    build_reranker_training_pairs,
    finetune_reranker,
)


def main():
    paths = DatasetPaths(repo_root=REPO_ROOT)

    # -----------------------------
    # (A) Train Stage-2 bi-encoder (FT retriever)
    # -----------------------------
    TRAIN_CSV = paths.train_csv
    BI_ENCODER_OUT = REPO_ROOT / "models" / "bge-m3-rac-stage2-only"

    train_df = pd.read_csv(TRAIN_CSV)

    # IMPORTANT: you confirmed parquet indices are based on this filter + reset_index
    if "source" in train_df.columns:
        train_df = train_df[train_df["source"] == "original"].reset_index(drop=True)
    else:
        train_df = train_df.reset_index(drop=True)

    print("=" * 80)
    print("[02] Training Stage-2 bi-encoder (FT retriever)")
    print(f"Output dir: {BI_ENCODER_OUT}")
    print(f"Parquet:    {STAGE2_TRIPLETS_PARQUET_DEFAULT}")
    print("=" * 80)

    BI_ENCODER_OUT.mkdir(parents=True, exist_ok=True)

    # Train only if checkpoint does not exist (to avoid re-training by mistake)
    if not (BI_ENCODER_OUT / "modules.json").exists():
        _ = train_biencoder_from_parquet(
            train_df=train_df,
            output_dir=BI_ENCODER_OUT,
            parquet_path=STAGE2_TRIPLETS_PARQUET_DEFAULT,  # fixed ../dataset/Retriever/... path (resolved)
            base_model_name="BAAI/bge-m3",
            text_col="Content",
            batch_size=8,
            epochs=2,
            lr=1e-5,
            margin=0.2,
            seed=42,
            max_rows=None,  # set e.g. 50000 for quick debug
        )
    else:
        print(f"[Skip] Stage-2 bi-encoder already exists: {BI_ENCODER_OUT}")

    # -----------------------------
    # (B) Fine-tune reranker (FT reranker)
    # -----------------------------
    TRIPLETS_CSV = paths.triplets_csv
    OUT_RERANKER_DIR = REPO_ROOT / "outputs" / "models" / "bge-reranker-finetuned"

    print("=" * 80)
    print("[02] Fine-tuning Cross-Encoder reranker (FT reranker)")
    print(f"Triplets CSV: {TRIPLETS_CSV}")
    print(f"Output dir:   {OUT_RERANKER_DIR}")
    print("=" * 80)

    triplets_df = load_triplets_csv(TRIPLETS_CSV)
    hard_triplets = select_hard_triplets(triplets_df, min_hard=1000)
    print(f"Selected {len(hard_triplets)} triplets for reranker fine-tuning.")

    train_examples = build_reranker_training_pairs(hard_triplets)
    print(f"Created {len(train_examples)} training pairs.")

    out_dir = finetune_reranker(train_examples, output_path=OUT_RERANKER_DIR)
    print(f"Fine-tuned reranker saved to: {out_dir}")

    print("\n✅ Done. You can now run: python scripts/03_run_hybrid_eval_qwen.py")


if __name__ == "__main__":
    main()