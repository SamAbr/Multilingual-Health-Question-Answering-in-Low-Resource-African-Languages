"""
Ceiling Measurement & Retrieval Diagnostic Tool for Multilingual Health QA
Measures:
1. Fraction of validation rows with near-duplicate retrieval matches (>0.85, 0.70-0.85, 0.50-0.70, <0.50)
2. Per-subset ROUGE-1 and ROUGE-L on exact retrieved references vs model generation
3. Theoretical upper-bound competition leaderboard score ceiling
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from rouge_utils import compute_rouge_metrics


def measure_retrieval_ceiling(
    val_df: pd.DataFrame,
    retriever,
    similarity_thresholds: List[float] = [0.85, 0.70, 0.50]
) -> pd.DataFrame:
    """
    Computes real empirical ceiling of retrieval on the validation dataset.
    """
    results = []
    val_df = val_df.copy()

    sim_scores = []
    retrieved_answers = []

    for i, row in val_df.iterrows():
        q_text = str(row['input'])
        subset = str(row['subset'])
        ans, sim = retriever.get_top1(q_text, subset, exclude_exact=False)
        sim_scores.append(sim)
        retrieved_answers.append(ans)

    val_df['sim_score'] = sim_scores
    val_df['retrieved_ans'] = retrieved_answers

    # Categorize into similarity bins
    def get_bin(score):
        if score >= 0.85:
            return 'High (>=0.85)'
        elif score >= 0.70:
            return 'Medium-High (0.70-0.85)'
        elif score >= 0.50:
            return 'Medium (0.50-0.70)'
        else:
            return 'Low (<0.50)'

    val_df['sim_bin'] = val_df['sim_score'].apply(get_bin)

    print("\n" + "="*80)
    print(" 📊 REAL RETRIEVAL CEILING MEASUREMENT REPORT (Validation Set: 6,686 rows)")
    print("="*80)

    # 1. Overall Similarity Bin Distribution
    bin_counts = val_df['sim_bin'].value_counts()
    bin_pcts = (bin_counts / len(val_df) * 100).round(2)
    
    bin_df = pd.DataFrame({'Count': bin_counts, 'Pct (%)': bin_pcts})
    print("\n--- 1. Validation Similarity Bin Breakdown ---")
    print(bin_df.to_string())

    # 2. ROUGE per Similarity Bin
    print("\n--- 2. Ground Truth ROUGE Scores per Similarity Bin ---")
    bin_rouge_list = []
    for bin_name in ['High (>=0.85)', 'Medium-High (0.70-0.85)', 'Medium (0.50-0.70)', 'Low (<0.50)']:
        sub_group = val_df[val_df['sim_bin'] == bin_name]
        if len(sub_group) > 0:
            m = compute_rouge_metrics(sub_group['retrieved_ans'].tolist(), sub_group['output'].tolist())
            bin_rouge_list.append({
                'Bin': bin_name,
                'Count': len(sub_group),
                'Pct (%)': round(len(sub_group) / len(val_df) * 100, 2),
                'ROUGE-1 F1': round(m['rouge1_f1'], 4),
                'ROUGE-L F1': round(m['rougeL_f1'], 4),
            })
    print(pd.DataFrame(bin_rouge_list).to_string(index=False))

    # 3. Per-Subset High-Similarity Match Fraction (>=0.70)
    print("\n--- 3. Per-Subset Near-Duplicate Match Fraction (Similarity >= 0.70) ---")
    subset_ceilings = []
    for subset, group in val_df.groupby('subset'):
        high_sim_count = (group['sim_score'] >= 0.70).sum()
        pct_high = round(high_sim_count / len(group) * 100, 2)
        
        m_retr = compute_rouge_metrics(group['retrieved_ans'].tolist(), group['output'].tolist())
        subset_ceilings.append({
            'Subset': subset,
            'Total Val Count': len(group),
            'Match Count (>=0.70)': high_sim_count,
            'Match Pct (%)': pct_high,
            'Full Retrieval ROUGE-L': round(m_retr['rougeL_f1'], 4)
        })
    subset_df = pd.DataFrame(subset_ceilings).sort_values('Total Val Count', ascending=False)
    print(subset_df.to_string(index=False))

    print("\n" + "="*80)
    return val_df
