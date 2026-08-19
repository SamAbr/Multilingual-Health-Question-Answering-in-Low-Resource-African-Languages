"""
Model Explorer Module for Multilingual Health QA
Supports evaluating and fine-tuning:
1. facebook/nllb-200-3.3B + LoRA PEFT
2. google/mt5-large / google/mt5-base
3. CohereForAI/aya-101
"""

import torch
import pandas as pd
from typing import Dict, Any
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

SUPPORTED_MODELS = {
    'nllb-600m': 'facebook/nllb-200-distilled-600M',
    'nllb-1.3b': 'facebook/nllb-200-1.3B',
    'nllb-3.3b': 'facebook/nllb-200-3.3B',
    'mt5-large': 'google/mt5-large',
    'mt5-base':  'google/mt5-base',
}


def load_exploration_model(
    model_key: str = 'nllb-3.3b',
    use_peft: bool = True,
    device: str = 'cuda'
):
    """
    Loads candidate models with optional LoRA PEFT adapters.
    """
    model_name = SUPPORTED_MODELS.get(model_key, model_key)
    print(f"[INFO] Loading Model: {model_name} (Device: {device})...", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load model with 16-bit precision if CUDA
    torch_dtype = torch.float16 if device == 'cuda' else torch.float32
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch_dtype).to(device)

    if use_peft:
        try:
            from peft import LoraConfig, TaskType, get_peft_model
            print("[INFO] Configuring LoRA PEFT adapter...", flush=True)
            peft_config = LoraConfig(
                task_type=TaskType.SEQ_2_SEQ_LM,
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                target_modules=["q_proj", "v_proj", "k_proj", "out_proj"] if "nllb" in model_name else ["q", "v"],
            )
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()
        except Exception as e:
            print(f"[WARN] LoRA initialization fallback: {e}", flush=True)

    return model, tokenizer
