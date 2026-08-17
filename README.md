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
├── notebooks/                # Jupyter Notebooks for analysis & prototyping
│   └── multilingual_health_qa_starter_notebook.ipynb  # Comprehensive starter notebook
├── src/                      # Modular Python scripts & algorithms
│   ├── __init__.py
│   └── nllb_pipeline.py      # NLLB-200 seq2seq, RAG retriever & PEFT fine-tuning pipeline
├── models/
│   └── checkpoints/          # Model checkpoints & fine-tuned weights
│       └── nllb-health-qa-checkpoint/
├── submissions/              # Formatted CSV submission outputs for Zindi
├── LICENSE
└── README.md
```

---

## 🚀 Getting Started

### 1. Requirements & Dependencies
Ensure you have PyTorch, Transformers, Datasets, evaluate, and scikit-learn installed:
```bash
pip install torch transformers datasets evaluate scikit-learn pandas numpy rouge_score peft
```

### 2. Running the Starter Notebook
Launch Jupyter Notebook or Jupyter Lab and open the notebook in `notebooks/`:
```bash
jupyter notebook notebooks/multilingual_health_qa_starter_notebook.ipynb
```
The notebook automatically sets up `BASE_DIR`, creates required output directories (`submissions/`, `models/checkpoints/`), and verifies dataset availability in `data/raw/`.

### 3. Running the NLLB-200 Training & Generation Pipeline

To perform a fast dry-run verification:
```bash
python src/nllb_pipeline.py --dry_run
```

To run full fine-tuning with RAG retrieval enabled:
```bash
python src/nllb_pipeline.py --model_name facebook/nllb-200-distilled-600M --use_rag --epochs 3 --batch_size 8
```

Submissions will be automatically generated and saved to `submissions/submission_nllb.csv`.