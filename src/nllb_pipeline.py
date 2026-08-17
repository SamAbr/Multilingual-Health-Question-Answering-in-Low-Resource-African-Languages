"""
NLLB-200 Pipeline for Multilingual Health QA in Low-Resource African Languages
Supports:
- Model selection: facebook/nllb-200-distilled-600M, facebook/nllb-200-1.3B, facebook/nllb-200-3.3B
- Native FLORES-200 language code mappings (aka_Latn, amh_Ethi, lug_Latn, swh_Latn, eng_Latn)
- Per-Language batch grouping & forced_bos_token_id handling
- Hybrid Lexical (TF-IDF) + Dense (Sentence-Transformer) RAG retriever
- Per-subset ROUGE metric evaluation callback during fine-tuning
- Out-of-fold per-subset threshold optimization
- Submission formatting with TargetLLM post-processing
"""

import os
import sys
import re
import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
)
from datasets import Dataset

# Add src to python path for modular imports
sys.path.append(str(Path(__file__).resolve().parent))
from retrieval import HybridRetriever, SubsetRAGRetriever
from threshold_optimizer import optimize_per_subset_thresholds
from rouge_utils import make_rouge_scorer

# Ensure UTF-8 output encoding for Windows compatibility
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# ── FLORES-200 Language Mapping ─────────────────────────────────────────────
SUBSET_TO_FLORES = {
    'Aka_Gha': 'aka_Latn',
    'Amh_Eth': 'amh_Ethi',
    'Lug_Uga': 'lug_Latn',
    'Swa_Ken': 'swh_Latn',
    'Eng_Uga': 'eng_Latn',
    'Eng_Gha': 'eng_Latn',
    'Eng_Eth': 'eng_Latn',
    'Eng_Ken': 'eng_Latn',
}

SUBSET_TO_NAME = {
    'Aka_Gha': 'Akan (Ghana)',
    'Amh_Eth': 'Amharic (Ethiopia)',
    'Lug_Uga': 'Luganda (Uganda)',
    'Swa_Ken': 'Swahili (Kenya)',
    'Eng_Uga': 'English (Uganda)',
    'Eng_Gha': 'English (Ghana)',
    'Eng_Eth': 'English (Ethiopia)',
    'Eng_Ken': 'English (Kenya)',
}


def build_prompt(question: str, subset: str = None, context: str = None) -> str:
    lang_name = SUBSET_TO_NAME.get(subset, 'Language')
    prompt = f"Medical QA [{lang_name}]\n"
    if context:
        prompt += f"Reference Context: {context}\n"
    prompt += f"Question: {question}\nAnswer:"
    return prompt


def clean_text_for_target_llm(text: str) -> str:
    """Post-processes output string for TargetLLM judge evaluation."""
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    # Strip HuggingFace special tokens
    text = re.sub(r'<extra_id_\d+>', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    # Collapse multiple spaces / blank lines
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def compute_rouge_metrics(predictions, references):
    scorer = make_rouge_scorer(['rouge1', 'rougeL'])
    r1_f1, rL_f1 = [], []
    for pred, ref in zip(predictions, references):
        scores = scorer.score(str(ref), clean_text_for_target_llm(pred))
        r1_f1.append(scores['rouge1'].fmeasure)
        rL_f1.append(scores['rougeL'].fmeasure)
    return {
        'rouge1_f1': float(np.mean(r1_f1)),
        'rougeL_f1': float(np.mean(rL_f1)),
    }


# ── Per-Language Batch Grouping & Preprocessor ─────────────────────────────
def prepare_nllb_dataset(df, tokenizer, retriever=None, max_input_len=256, max_target_len=256, is_train=True):
    records = []
    # Sort or group by subset to enable native per-language batch grouping
    df_sorted = df.sort_values('subset') if is_train else df.copy()

    for _, row in df_sorted.iterrows():
        subset = str(row['subset'])
        flores_code = SUBSET_TO_FLORES.get(subset, 'eng_Latn')
        q_text = str(row['input'])
        
        context = ""
        if retriever:
            context = retriever.get_context(q_text, subset, exclude_exact=is_train)
            
        prompt = build_prompt(q_text, subset, context)
        rec = {
            'prompt': prompt,
            'subset': subset,
            'flores_code': flores_code,
        }
        if 'output' in row and pd.notna(row['output']):
            rec['answer'] = str(row['output'])
        records.append(rec)

    raw_ds = Dataset.from_list(records)

    def preprocess(examples):
        input_ids_list, attention_mask_list, labels_list = [], [], []

        for i in range(len(examples['prompt'])):
            prompt_text = examples['prompt'][i]
            src_code = examples['flores_code'][i]

            tokenizer.src_lang = src_code
            encoded_in = tokenizer(
                prompt_text,
                max_length=max_input_len,
                truncation=True,
                padding=False,
            )
            input_ids_list.append(encoded_in['input_ids'])
            attention_mask_list.append(encoded_in['attention_mask'])

            if 'answer' in examples:
                tokenizer.src_lang = src_code
                encoded_out = tokenizer(
                    text_target=examples['answer'][i],
                    max_length=max_target_len,
                    truncation=True,
                    padding=False,
                )
                label_ids = [(tok if tok != tokenizer.pad_token_id else -100) for tok in encoded_out['input_ids']]
                labels_list.append(label_ids)

        model_inputs = {
            'input_ids': input_ids_list,
            'attention_mask': attention_mask_list,
        }
        if labels_list:
            model_inputs['labels'] = labels_list
        return model_inputs

    remove_cols = ['prompt', 'subset', 'flores_code']
    if 'answer' in raw_ds.column_names:
        remove_cols.append('answer')

    return raw_ds.map(preprocess, batched=True, remove_columns=remove_cols)


# ── Per-Subset ROUGE Metric Evaluation Callback ──────────────────────────────
class PerSubsetRougeCallback(TrainerCallback):
    """Callback to compute and display per-subset ROUGE metrics at each evaluation epoch."""
    def __init__(self, val_df, tokenizer, retriever=None, device='cuda'):
        self.val_df = val_df
        self.tokenizer = tokenizer
        self.retriever = retriever
        self.device = device

    def on_evaluate(self, args, state, control, metrics=None, model=None, **kwargs):
        if model is None:
            return

        print(f"\n[EVAL] Epoch {state.epoch:.1f} — Per-Language Subset ROUGE Breakdown:")
        val_preds = generate_nllb_answers(model, self.tokenizer, self.val_df, retriever=self.retriever, device=self.device)
        
        subset_metrics = []
        for subset, group_data in self.val_df.groupby('subset'):
            sub_indices = group_data.index
            sub_preds = [val_preds[i] for i in sub_indices]
            sub_refs = group_data['output'].tolist()
            m = compute_rouge_metrics(sub_preds, sub_refs)
            subset_metrics.append({
                'Subset': subset,
                'Language': SUBSET_TO_NAME.get(subset, subset),
                'ROUGE-1 F1': round(m['rouge1_f1'], 4),
                'ROUGE-L F1': round(m['rougeL_f1'], 4),
                'Count': len(group_data),
            })

        sub_df = pd.DataFrame(subset_metrics)
        print(sub_df.to_string(index=False))
        overall = compute_rouge_metrics(val_preds, self.val_df['output'].tolist())
        print(f"Overall Validation ROUGE-1: {overall['rouge1_f1']:.4f} | ROUGE-L: {overall['rougeL_f1']:.4f}\n", flush=True)


# ── NLLB Per-Language Batch Generation ───────────────────────────────────────
def generate_nllb_answers(model, tokenizer, df, retriever=None, max_input_len=256, max_target_len=256, batch_size=8, device='cuda'):
    model.eval()
    all_preds = [None] * len(df)

    # Group by language subset so each mini-batch shares the exact forced_bos_token_id
    for subset, group_df in df.groupby('subset'):
        group_indices = group_df.index.tolist()
        flores_code = SUBSET_TO_FLORES.get(subset, 'eng_Latn')
        
        # Get target language BOS token ID for NLLB
        if hasattr(tokenizer, 'lang_code_to_id') and flores_code in tokenizer.lang_code_to_id:
            forced_bos_id = tokenizer.lang_code_to_id[flores_code]
        else:
            forced_bos_id = tokenizer.convert_tokens_to_ids(flores_code)

        for start_idx in range(0, len(group_df), batch_size):
            batch_indices = group_indices[start_idx:start_idx + batch_size]
            batch_rows = group_df.iloc[start_idx:start_idx + batch_size]
            
            prompts = []
            for _, row in batch_rows.iterrows():
                q_text = str(row['input'])
                ctx = retriever.get_context(q_text, subset) if retriever else ""
                prompts.append(build_prompt(q_text, subset, ctx))

            tokenizer.src_lang = flores_code
            inputs = tokenizer(prompts, return_tensors='pt', padding=True, max_length=max_input_len, truncation=True).to(device)

            with torch.no_grad():
                generated_tokens = model.generate(
                    **inputs,
                    forced_bos_token_id=forced_bos_id,
                    max_new_tokens=max_target_len,
                    min_length=15,
                    num_beams=4,
                    length_penalty=1.0,
                    no_repeat_ngram_size=3,
                    early_stopping=True,
                )

            for i, tok_seq in enumerate(generated_tokens):
                decoded = tokenizer.decode(tok_seq, skip_special_tokens=True)
                clean_pred = clean_text_for_target_llm(decoded)
                original_idx = batch_indices[i]
                all_preds[original_idx] = clean_pred

    return all_preds


# ── Main Training & Execution Pipeline ────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="NLLB Multilingual Health QA Pipeline")
    parser.add_argument('--model_name', type=str, default='facebook/nllb-200-distilled-600M',
                        help='NLLB model name: facebook/nllb-200-distilled-600M, facebook/nllb-200-1.3B')
    parser.add_argument('--train_path', type=str, default='data/raw/Training set.csv')
    parser.add_argument('--val_path', type=str, default='data/raw/Validation set.csv')
    parser.add_argument('--test_path', type=str, default='data/raw/Test set.csv')
    parser.add_argument('--output_dir', type=str, default='models/checkpoints/nllb-health-qa-checkpoint')
    parser.add_argument('--submission_path', type=str, default='submissions/submission_nllb.csv')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--use_rag', action='store_true', help='Enable RAG context retrieval')
    parser.add_argument('--use_dense_rag', action='store_true', help='Enable dense sentence embedding RAG retrieval')
    parser.add_argument('--use_peft', action='store_true', help='Use LoRA PEFT fine-tuning')
    parser.add_argument('--dry_run', action='store_true', help='Fast dry run on small subset for verification')
    parser.add_argument('--skip_submission', action='store_true', help='Skip test set inference and submission CSV generation')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[INFO] Using device: {device}", flush=True)
    print(f"[INFO] Loading Model: {args.model_name}", flush=True)

    train_df = pd.read_csv(args.train_path)
    val_df = pd.read_csv(args.val_path)
    test_df = pd.read_csv(args.test_path)

    if args.dry_run:
        print("[INFO] Dry run mode active: truncating datasets for rapid testing...", flush=True)
        train_df = train_df.head(40)
        val_df = val_df.head(10)
        test_df = test_df.head(5)

    retriever = None
    if args.use_rag or args.use_dense_rag:
        print(f"[INFO] Initializing Hybrid RAG Retriever (Dense={args.use_dense_rag})...", flush=True)
        retriever = HybridRetriever(train_df, enable_dense=args.use_dense_rag)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name).to(device)

    if args.use_peft:
        try:
            from peft import LoraConfig, get_peft_model, TaskType
            print("[INFO] Enabling LoRA PEFT...", flush=True)
            peft_config = LoraConfig(
                task_type=TaskType.SEQ_2_SEQ_LM,
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                target_modules=["q_proj", "v_proj", "k_proj", "out_proj"]
            )
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()
        except ImportError:
            print("[WARN] PEFT not installed. Falling back to standard fine-tuning.", flush=True)

    print("[INFO] Preprocessing datasets with native language tokenization...", flush=True)
    hf_train = prepare_nllb_dataset(train_df, tokenizer, retriever=retriever, is_train=True)
    hf_val = prepare_nllb_dataset(val_df, tokenizer, retriever=retriever, is_train=False)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8 if device == 'cuda' else None,
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        predict_with_generate=False,
        bf16=(device == 'cuda' and torch.cuda.is_bf16_supported()),
        fp16=(device == 'cuda' and not torch.cuda.is_bf16_supported()),
        eval_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        logging_steps=5 if args.dry_run else 100,
        report_to='none',
    )

    per_subset_callback = PerSubsetRougeCallback(val_df, tokenizer, retriever=retriever, device=device)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=hf_train,
        eval_dataset=hf_val,
        processing_class=tokenizer,
        data_collator=data_collator,
        callbacks=[per_subset_callback],
    )

    print("[INFO] Starting fine-tuning with per-language subset evaluation...", flush=True)
    trainer.train()
    print("[INFO] Fine-tuning complete.", flush=True)

    print("[INFO] Running evaluation on Validation Set...", flush=True)
    val_preds = generate_nllb_answers(model, tokenizer, val_df, retriever=retriever, device=device)
    val_metrics = compute_rouge_metrics(val_preds, val_df['output'].tolist())
    print(f"[METRIC] Final Validation ROUGE-1 F1: {val_metrics['rouge1_f1']:.4f}", flush=True)
    print(f"[METRIC] Final Validation ROUGE-L F1: {val_metrics['rougeL_f1']:.4f}", flush=True)

    if not args.skip_submission:
        print(f"[INFO] Generating predictions for Test Set ({len(test_df)} questions)...", flush=True)
        test_preds = generate_nllb_answers(model, tokenizer, test_df, retriever=retriever, device=device)

        submission_df = pd.DataFrame({
            'ID': test_df['ID'],
            'TargetRLF1': test_preds,
            'TargetR1F1': test_preds,
            'TargetLLM': test_preds,
        })

        submission_df.to_csv(args.submission_path, index=False)
        print(f"[SUCCESS] Submission saved successfully to: {args.submission_path}", flush=True)
        print(submission_df.head(), flush=True)
    else:
        print("[INFO] --skip_submission set. Skipped test set submission file generation.", flush=True)


if __name__ == '__main__':
    main()
