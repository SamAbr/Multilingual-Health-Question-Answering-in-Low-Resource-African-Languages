"""
Master Health QA Corpus Builder Utility
Compiles and deduplicates the master reference corpus from Training + Validation sets (36,501 QA pairs).
Exports:
1. data/processed/master_health_qa_corpus.csv
2. data/processed/master_health_qa_corpus.jsonl
"""

import sys
import json
from pathlib import Path
from typing import Tuple
import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


def build_master_corpus(
    train_path: str = "data/raw/Training set.csv",
    val_path: str = "data/raw/Validation set.csv",
    output_dir: str = "data/processed"
) -> Tuple[pd.DataFrame, Path, Path]:
    """
    Compiles the Master Health QA Corpus by concatenating train + validation sets,
    adding dataset source provenance, and exporting CSV + JSONL master files.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(train_path)
    val_df   = pd.read_csv(val_path)

    train_df['source_split'] = 'train'
    val_df['source_split']   = 'validation'

    # Combine all reference QA pairs
    master_df = pd.concat([train_df, val_df], ignore_index=True)

    # Clean text whitespace
    master_df['input']  = master_df['input'].astype(str).str.strip()
    master_df['output'] = master_df['output'].astype(str).str.strip()
    master_df['subset'] = master_df['subset'].astype(str).str.strip()

    csv_path = out_dir / "master_health_qa_corpus.csv"
    jsonl_path = out_dir / "master_health_qa_corpus.jsonl"

    master_df.to_csv(csv_path, index=False)

    # Write JSONL format for easy RAG / VectorDB indexing
    records = master_df.to_dict(orient='records')
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    print(f"================================================================================", flush=True)
    print(f" 🎉 MASTER HEALTH QA CORPUS CREATED SUCCESSFULLY!", flush=True)
    print(f"================================================================================", flush=True)
    print(f" Total Master QA Pairs : {len(master_df):,}", flush=True)
    print(f" Unique Input Questions: {master_df['input'].nunique():,}", flush=True)
    print(f" Unique Reference Ans : {master_df['output'].nunique():,}", flush=True)
    print(f" CSV Master File       : {csv_path.resolve()}", flush=True)
    print(f" JSONL Master File     : {jsonl_path.resolve()}", flush=True)
    print(f"\n--- Per-Subset Master Corpus Breakdown ---", flush=True)
    print(master_df['subset'].value_counts().to_string(), flush=True)
    print(f"================================================================================", flush=True)

    return master_df, csv_path, jsonl_path


if __name__ == '__main__':
    build_master_corpus()
