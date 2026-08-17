"""
Retrieval Module for Multilingual Health QA in Low-Resource African Languages.
Supports:
1. SubsetRAGRetriever: Sub-word / Char n-gram TF-IDF retrieval per language subset.
2. DenseSentenceRetriever: Multilingual dense sentence embedding retrieval via Transformers/PyTorch.
3. HybridRetriever: Weighted ensemble combining TF-IDF lexical overlap and Dense semantic similarity.
"""

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class SubsetRAGRetriever:
    """TF-IDF retriever operating per language subset with character and word n-grams."""

    def __init__(self, train_df, question_col='input', answer_col='output', group_col='subset', ngram_range=(1, 3)):
        self.question_col = question_col
        self.answer_col = answer_col
        self.group_col = group_col
        self.models = {}

        for group_val, group_data in train_df.groupby(group_col):
            questions = group_data[question_col].astype(str).tolist()
            answers = group_data[answer_col].astype(str).tolist()
            
            # Use combined word and character n-grams for low-resource African language morphosyntax
            vec = TfidfVectorizer(analyzer='char_wb', ngram_range=ngram_range, max_features=25000, min_df=1)
            X = vec.fit_transform(questions)
            
            self.models[group_val] = {
                'vectorizer': vec,
                'X': X,
                'answers': answers,
                'questions': questions,
            }

    def get_top1(self, question: str, group_val: str, exclude_exact: bool = False):
        """Returns (best_answer, similarity_score)."""
        if group_val not in self.models:
            return "", 0.0

        m = self.models[group_val]
        q_vec = m['vectorizer'].transform([str(question)])
        sims = cosine_similarity(q_vec, m['X']).flatten()

        if len(sims) == 0 or np.max(sims) == 0.0:
            return "", 0.0

        top_indices = sims.argsort()[::-1]
        best_idx = top_indices[0]

        # In training mode, prevent exact match self-retrieval
        if exclude_exact and len(top_indices) > 1 and sims[best_idx] > 0.999:
            best_idx = top_indices[1]

        best_score = float(sims[best_idx])
        best_answer = m['answers'][best_idx]
        return best_answer, best_score

    def get_context(self, question: str, group_val: str, exclude_exact: bool = False, min_score: float = 0.1) -> str:
        ans, score = self.get_top1(question, group_val, exclude_exact=exclude_exact)
        if score < min_score:
            return ""
        return ans


class DenseSentenceRetriever:
    """Dense sentence-embedding retriever using PyTorch and HuggingFace Transformers."""

    def __init__(self, train_df, model_name='sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
                 question_col='input', answer_col='output', group_col='subset', device=None):
        self.question_col = question_col
        self.answer_col = answer_col
        self.group_col = group_col
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        self.models = {}

        try:
            from transformers import AutoTokenizer, AutoModel
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.encoder = AutoModel.from_pretrained(model_name).to(self.device)
            self.encoder.eval()
            self._has_dense = True
        except Exception as e:
            print(f"[WARN] DenseSentenceRetriever failed to load {model_name}: {e}. Falling back to disabled.")
            self._has_dense = False
            return

        for group_val, group_data in train_df.groupby(group_col):
            questions = group_data[question_col].astype(str).tolist()
            answers = group_data[answer_col].astype(str).tolist()
            embeddings = self._encode_batch(questions)

            self.models[group_val] = {
                'embeddings': embeddings,  # Normalized NumPy array (N, D)
                'answers': answers,
                'questions': questions,
            }

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def _encode_batch(self, texts, batch_size=32):
        if not self._has_dense:
            return None

        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors='pt'
            ).to(self.device)

            with torch.no_grad():
                model_output = self.encoder(**encoded)
                embeddings = self._mean_pooling(model_output, encoded['attention_mask'])
                # L2 normalize embeddings
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                all_embeddings.append(embeddings.cpu().numpy())

        return np.vstack(all_embeddings)

    def get_top1(self, question: str, group_val: str, exclude_exact: bool = False):
        """Returns (best_answer, similarity_score)."""
        if not self._has_dense or group_val not in self.models:
            return "", 0.0

        m = self.models[group_val]
        q_emb = self._encode_batch([str(question)], batch_size=1)  # (1, D)
        sims = (q_emb @ m['embeddings'].T).flatten()  # Cosine similarity for normalized vectors

        if len(sims) == 0:
            return "", 0.0

        top_indices = sims.argsort()[::-1]
        best_idx = top_indices[0]

        if exclude_exact and len(top_indices) > 1 and sims[best_idx] > 0.999:
            best_idx = top_indices[1]

        best_score = float(sims[best_idx])
        best_answer = m['answers'][best_idx]
        return best_answer, best_score


class HybridRetriever:
    """Combines lexical TF-IDF and semantic Dense sentence embeddings into a hybrid score."""

    def __init__(self, train_df, question_col='input', answer_col='output', group_col='subset',
                 dense_model_name='sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
                 alpha=0.5, enable_dense=True):
        self.alpha = alpha  # Weight for TF-IDF score: alpha * tfidf + (1 - alpha) * dense
        self.tfidf_retriever = SubsetRAGRetriever(train_df, question_col, answer_col, group_col)
        self.enable_dense = enable_dense

        if enable_dense:
            self.dense_retriever = DenseSentenceRetriever(
                train_df, model_name=dense_model_name, question_col=question_col,
                answer_col=answer_col, group_col=group_col
            )
        else:
            self.dense_retriever = None

    def get_top1(self, question: str, group_val: str, exclude_exact: bool = False):
        ans_tfidf, score_tfidf = self.tfidf_retriever.get_top1(question, group_val, exclude_exact=exclude_exact)
        
        if not self.enable_dense or self.dense_retriever is None or not self.dense_retriever._has_dense:
            return ans_tfidf, score_tfidf

        ans_dense, score_dense = self.dense_retriever.get_top1(question, group_val, exclude_exact=exclude_exact)

        # Hybrid scoring
        if ans_tfidf == ans_dense:
            # High-confidence match: both TF-IDF and Dense point to the exact same reference answer
            combined_score = self.alpha * score_tfidf + (1 - self.alpha) * score_dense + 0.1
            return ans_tfidf, min(combined_score, 1.0)
        else:
            # Picks the answer from whichever model has the higher score
            if score_tfidf >= score_dense:
                return ans_tfidf, score_tfidf
            else:
                return ans_dense, score_dense

    def get_context(self, question: str, group_val: str, exclude_exact: bool = False, min_score: float = 0.1) -> str:
        ans, score = self.get_top1(question, group_val, exclude_exact=exclude_exact)
        if score < min_score:
            return ""
        return ans
