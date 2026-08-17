# Multilingual Health Question Answering in Low-Resource African Languages

This repository contains code, notebooks, and models for building end-to-end question-answering systems in low-resource African languages (Akan, Amharic, Luganda, Swahili, and regional English variants).

---

## 📁 Directory & Project Structure

```
Multilingual-Health-Question-Answering-in-Low-Resource-African-Languages/
├── data/
│   ├── raw/                  # Original raw Zindi datasets
│   │   ├── Training set.csv  # Training dataset
│   │   ├── Validation set.csv# Validation dataset
│   │   └── Test set.csv      # Test evaluation dataset
│   └── processed/            # Preprocessed & augmented datasets
├── notebooks/                # Jupyter Notebooks for analysis & cloud training
│   ├── colab_nllb_health_qa.ipynb                     # Cloud/Colab automated clone & run notebook
│   └── multilingual_health_qa_starter_notebook.ipynb # Interactive analysis & hybrid modeling notebook
├── src/                      # Modular Python scripts & pipeline algorithms
│   ├── __init__.py
│   ├── nllb_pipeline.py      # NLLB-200 generation, per-language batch sampler & callback
│   ├── retrieval.py          # TF-IDF, Multilingual Dense Sentence Embeddings & Hybrid RAG
│   └── threshold_optimizer.py# Per-subset threshold grid optimization & score calibration
├── models/
│   └── checkpoints/          # Model checkpoints & fine-tuned weights
├── submissions/              # Formatted CSV submission outputs for Zindi
├── LICENSE
└── README.md
```

---

## 🚀 Getting Started

### 1. Requirements & Dependencies
Ensure you have PyTorch, Transformers, Sentence-Transformers, Datasets, and evaluate installed:
```bash
pip install torch transformers sentence-transformers datasets evaluate scikit-learn pandas numpy rouge-score peft accelerate
```

### 2. Running in Cloud / Google Colab / Docker Environments
Open `notebooks/colab_nllb_health_qa.ipynb` in Google Colab or your cloud Jupyter notebook server. It includes an automated, zero-git fallback repository downloader that works in any Docker container.

### 3. Running the NLLB-200 Training & Hybrid RAG Pipeline

To perform a fast dry-run verification:
```bash
python src/nllb_pipeline.py --dry_run
```

To run full fine-tuning with Hybrid Lexical + Dense RAG retrieval enabled:
```bash
python src/nllb_pipeline.py --model_name facebook/nllb-200-distilled-600M --use_dense_rag --epochs 3 --batch_size 8
```

Submissions will be automatically generated and saved to `submissions/submission_nllb.csv`.