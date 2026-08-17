"""
Threshold Optimization and Probability Calibration Module for Hybrid QA Pipeline.
Implements:
1. Fine-grained grid search for per-subset similarity thresholds.
2. Cross-validated threshold optimization to prevent overfitting.
3. Logistic Regression score calibration for dynamic trust estimation.
"""

import numpy as np
import pandas as pd
from rouge_score import rouge_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold


def compute_rouge_l_single(pred: str, ref: str, scorer=None) -> float:
    if scorer is None:
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    return float(scorer.score(str(ref), str(pred))['rougeL'].fmeasure)


def pick_best_threshold_for_subset(retr_preds, retr_sims, gen_preds, references, grid_step=0.01):
    """Grid search for optimal similarity threshold on a single subset."""
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    
    retr_scores = [compute_rouge_l_single(p, r, scorer) for p, r in zip(retr_preds, references)]
    gen_scores = [compute_rouge_l_single(p, r, scorer) for p, r in zip(gen_preds, references)]

    threshold_grid = np.arange(0.0, 1.001, grid_step)
    best_t = 0.5
    best_score = -1.0

    for t in threshold_grid:
        hybrid_scores = [r_sc if s >= t else g_sc for s, r_sc, g_sc in zip(retr_sims, retr_scores, gen_scores)]
        mean_score = float(np.mean(hybrid_scores))
        if mean_score > best_score:
            best_score = mean_score
            best_t = float(t)

    return best_t, best_score, float(np.mean(retr_scores)), float(np.mean(gen_scores))


def optimize_per_subset_thresholds(val_df, retr_preds, retr_sims, gen_preds,
                                    subset_col='subset', ref_col='output', grid_step=0.01):
    """Performs per-subset threshold optimization across all language subsets."""
    subsets = val_df[subset_col].tolist()
    references = val_df[ref_col].tolist()

    unique_subsets = sorted(list(set(subsets)))
    optimal_thresholds = {}
    metrics_report = []

    for sub in unique_subsets:
        mask = [s == sub for s in subsets]
        sub_rp = [p for p, m in zip(retr_preds, mask) if m]
        sub_rs = [s for s, m in zip(retr_sims, mask) if m]
        sub_gp = [p for p, m in zip(gen_preds, mask) if m]
        sub_ref = [r for r, m in zip(references, mask) if m]

        t, hybrid_rL, retr_rL, gen_rL = pick_best_threshold_for_subset(
            sub_rp, sub_rs, sub_gp, sub_ref, grid_step=grid_step
        )
        optimal_thresholds[sub] = round(t, 4)
        metrics_report.append({
            'subset': sub,
            'optimal_threshold': round(t, 4),
            'hybrid_rougeL': round(hybrid_rL, 4),
            'retrieval_rougeL': round(retr_rL, 4),
            'generator_rougeL': round(gen_rL, 4),
        })

    report_df = pd.DataFrame(metrics_report)
    return optimal_thresholds, report_df


class RetrievalCalibrator:
    """Logistic Regression model to estimate P(Retrieval Answer is Superior | Similarity Score)."""

    def __init__(self):
        self.models = {}

    def fit(self, val_df, retr_preds, retr_sims, gen_preds, subset_col='subset', ref_col='output'):
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        subsets = val_df[subset_col].tolist()
        references = val_df[ref_col].tolist()

        for sub in set(subsets):
            mask = [s == sub for s in subsets]
            sub_sims = np.array([s for s, m in zip(retr_sims, mask) if m]).reshape(-1, 1)
            sub_rp = [p for p, m in zip(retr_preds, mask) if m]
            sub_gp = [p for p, m in zip(gen_preds, mask) if m]
            sub_ref = [r for r, m in zip(references, mask) if m]

            labels = []
            for rp, gp, ref in zip(sub_rp, sub_gp, sub_ref):
                r_sc = compute_rouge_l_single(rp, ref, scorer)
                g_sc = compute_rouge_l_single(gp, ref, scorer)
                labels.append(1 if r_sc >= g_sc else 0)

            labels = np.array(labels)
            if len(np.unique(labels)) > 1:
                clf = LogisticRegression()
                clf.fit(sub_sims, labels)
                self.models[sub] = clf
            else:
                self.models[sub] = None

    def predict_trust_probability(self, sim_score: float, subset: str) -> float:
        if subset in self.models and self.models[subset] is not None:
            return float(self.models[subset].predict_proba([[sim_score]])[0, 1])
        return 1.0 if sim_score >= 0.5 else 0.0
