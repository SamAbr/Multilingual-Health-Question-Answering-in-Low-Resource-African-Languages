"""
English-to-Amharic Translation Data Augmentation Module using deep-translator
Translates English subsets (Eng_Eth, Eng_Ken, Eng_Uga, Eng_Gha) to Amharic (Amh_Eth).
Augments Amharic dataset size from 1,845 rows -> 19,000+ rows!
"""

import sys
import time
from pathlib import Path
import pandas as pd
from typing import Optional

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


def translate_english_subsets_to_amharic(
    input_csv_path: str = "data/raw/Training set.csv",
    output_csv_path: str = "data/processed/amharic_augmented_dataset.csv",
    sample_limit: Optional[int] = None,
    batch_sleep_sec: float = 0.05
) -> pd.DataFrame:
    """
    Translates English health QA pairs to Amharic using deep-translator (GoogleTranslator).
    """
    try:
        from deep_translator import GoogleTranslator
    except ImportError:
        raise ImportError("deep-translator is required. Run: pip install deep-translator")

    in_path = Path(input_csv_path)
    out_path = Path(output_csv_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)

    # Filter English language subsets
    english_mask = df['subset'].isin(['Eng_Eth', 'Eng_Ken', 'Eng_Uga', 'Eng_Gha'])
    eng_df = df[english_mask].copy()

    if sample_limit and sample_limit > 0:
        eng_df = eng_df.head(sample_limit)

    print(f"================================================================================", flush=True)
    print(f" 🇪🇹 STARTING ENGLISH -> AMHARIC BATCH TRANSLATION ({len(eng_df):,} QA pairs)", flush=True)
    print(f"================================================================================", flush=True)

    translator = GoogleTranslator(source='en', target='am')

    translated_inputs = []
    translated_outputs = []

    start_time = time.time()
    for idx, row in eng_df.reset_index(drop=True).iterrows():
        q_eng = str(row['input'])
        a_eng = str(row['output']) if 'output' in row and pd.notna(row['output']) else ""

        # Translate Question
        try:
            q_amh = translator.translate(q_eng)
        except Exception as e:
            q_amh = q_eng

        # Translate Answer
        if a_eng:
            try:
                a_amh = translator.translate(a_eng)
            except Exception as e:
                a_amh = a_eng
        else:
            a_amh = ""

        translated_inputs.append(q_amh)
        translated_outputs.append(a_amh)

        if (idx + 1) % 100 == 0 or (idx + 1) == len(eng_df):
            elapsed = time.time() - start_time
            spd = (idx + 1) / elapsed
            print(f"[INFO] Translated {idx + 1}/{len(eng_df)} samples ({spd:.1f} samples/sec)...", flush=True)

        if batch_sleep_sec > 0:
            time.sleep(batch_sleep_sec)

    augmented_df = eng_df.copy()
    augmented_df['input_orig'] = eng_df['input']
    augmented_df['output_orig'] = eng_df.get('output', '')
    augmented_df['input'] = translated_inputs
    augmented_df['output'] = translated_outputs
    augmented_df['subset'] = 'Amh_Eth'  # Target Amharic subset
    augmented_df['is_augmented'] = True

    augmented_df.to_csv(out_path, index=False)

    print(f"================================================================================", flush=True)
    print(f" 🎉 AMHARIC AUGMENTED DATASET CREATED SUCCESSFULLY!", flush=True)
    print(f" Output File: {out_path.resolve()}", flush=True)
    print(f" Total Augmented Amharic Samples: {len(augmented_df):,}", flush=True)
    print(f"================================================================================", flush=True)

    return augmented_df


if __name__ == '__main__':
    translate_english_subsets_to_amharic()
