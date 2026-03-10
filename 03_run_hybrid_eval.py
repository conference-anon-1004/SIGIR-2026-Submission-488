from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

import pandas as pd
from sklearn.metrics import classification_report
from sentence_transformers import SentenceTransformer

from src.data_io import DatasetPaths
from src.conrac_backbone import (
    GenerativeRAC,
    build_faiss_index,
    encode_train_test_with_biencoder,
    load_model,
    run_full_evaluation,
)
from src.conrac_se_hybrid import HybridGenerativeRAC, run_hybrid_evaluation


def main():
    paths = DatasetPaths(repo_root=REPO_ROOT)

    TRAIN_CSV = paths.train_csv
    TEST_CSV = paths.test_csv

    # trained checkpoints produced by 02 script
    BI_ENCODER_DIR = REPO_ROOT / "models" / "bge-m3-rac-stage2-only"
    RERANKER_DIR = REPO_ROOT / "outputs" / "models" / "bge-reranker-finetuned"

    if not (BI_ENCODER_DIR.exists() and (BI_ENCODER_DIR / "modules.json").exists()):
        raise FileNotFoundError(
            f"Stage-2 bi-encoder checkpoint not found: {BI_ENCODER_DIR}\n"
            "Run: python scripts/02_train_retriever_reranker.py"
        )

    if not RERANKER_DIR.exists():
        raise FileNotFoundError(
            f"Fine-tuned reranker not found: {RERANKER_DIR}\n"
            "Run: python scripts/02_train_retriever_reranker.py"
        )

    # Load train/test
    train_df = pd.read_csv(TRAIN_CSV)
    # IMPORTANT: keep consistent with index basis used for retrieval/training
    if "source" in train_df.columns:
        train_df = train_df[train_df["source"] == "original"].reset_index(drop=True)
    else:
        train_df = train_df.reset_index(drop=True)

    test_df = pd.read_csv(TEST_CSV)

    # Load bi-encoder
    bi_encoder = SentenceTransformer(str(BI_ENCODER_DIR))
    bi_encoder.max_seq_length = 8192

    # Encode + build FAISS index (FT retriever -> Top-K)
    E_train, E_test = encode_train_test_with_biencoder(bi_encoder, train_df, test_df)
    index = build_faiss_index(E_train)
    print("FAISS Index built.")

    # Load Qwen
    model_qwen, tokenizer_qwen = load_model()

    out_dir = REPO_ROOT / "outputs" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    # (A) Always-on RAC baseline (ConRAC decoding)
    rac_system = GenerativeRAC(
        bi_encoder_path=str(BI_ENCODER_DIR),
        reranker_path=str(RERANKER_DIR),
        train_df=train_df,
        train_embeddings=E_train,
        index=index,
    )

    qwen_results = run_full_evaluation(rac_system, test_df, E_test, model_qwen, tokenizer_qwen)
    qwen_out_csv = out_dir / "rac_qwen_finetuned_results.csv"
    qwen_results.to_csv(qwen_out_csv, index=False)

    valid_qwen = qwen_results[qwen_results["pred_label"].isin(["Secret", "Confidential", "Unclassified"])]
    print("\n=== Qwen Generative RAC Results ===")
    print(classification_report(valid_qwen["true_label"], valid_qwen["pred_label"], digits=4))
    print(f"Saved RAC results -> {qwen_out_csv}")

    # (B) Hybrid / Selective Escalation (ConRAC-SE)
    hybrid_rac = HybridGenerativeRAC(
        bi_encoder_path=str(BI_ENCODER_DIR),
        reranker_path=str(RERANKER_DIR),
        train_df=train_df,
        train_embeddings=E_train,
        index=index,
    )

    hybrid_results = run_hybrid_evaluation(hybrid_rac, test_df, E_test, model_qwen, tokenizer_qwen)
    hybrid_out_csv = out_dir / "hybrid_qwen_results.csv"
    hybrid_results.to_csv(hybrid_out_csv, index=False)

    print("\n=== Hybrid Conditional RAC Results (Qwen) ===")
    print(classification_report(hybrid_results["true_label"], hybrid_results["pred_label"], digits=4))
    print(f"Saved Hybrid results -> {hybrid_out_csv}")


if __name__ == "__main__":
    main()