# dataset.py
# ─────────────────────────────────────────────────────────
# Loads, validates, and tokenizes the training dataset.
# Run this directly to preview the data: python dataset.py
# ─────────────────────────────────────────────────────────

import json
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict

from config import (
    DATA_FILE, BASE_MODEL_NAME, MAX_SEQ_LENGTH,
    LABEL2ID, TEST_SPLIT, RANDOM_SEED
)


# ── 1. Load raw JSON data ──────────────────────────────────
def load_raw_data(filepath: str = DATA_FILE) -> pd.DataFrame:
    """
    Loads the JSON dataset and returns a clean pandas DataFrame.
    Expected JSON format: [{"text": "...", "label": "safe|injection"}, ...]
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset not found at {filepath}\n"
            "Make sure data/sample_data.json exists."
        )

    with open(filepath, "r", encoding="utf-8") as f:
        raw = json.load(f)

    df = pd.DataFrame(raw)

    # ── Validate columns ──────────────────────────────────
    assert "text"  in df.columns, "Missing 'text' column in dataset."
    assert "label" in df.columns, "Missing 'label' column in dataset."

    # ── Drop nulls ────────────────────────────────────────
    df = df.dropna(subset=["text", "label"])
    df["text"]  = df["text"].str.strip()
    df["label"] = df["label"].str.strip().str.lower()

    # ── Encode labels to integers ─────────────────────────
    df["label_id"] = df["label"].map(LABEL2ID)
    if df["label_id"].isna().any():
        bad = df[df["label_id"].isna()]["label"].unique().tolist()
        raise ValueError(f"Unknown labels found: {bad}. Expected: {list(LABEL2ID.keys())}")

    print(f"✅ Loaded {len(df)} samples")
    print(df["label"].value_counts().to_string())
    return df


# ── 2. Split into train / validation ──────────────────────
def split_data(df: pd.DataFrame):
    train_df, val_df = train_test_split(
        df,
        test_size=TEST_SPLIT,
        random_state=RANDOM_SEED,
        stratify=df["label"]       # keep class balance in both splits
    )
    print(f"\n📊 Train: {len(train_df)} | Validation: {len(val_df)}")
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


# ── 3. Tokenize with HuggingFace tokenizer ────────────────
def tokenize_dataset(train_df: pd.DataFrame, val_df: pd.DataFrame) -> DatasetDict:
    """
    Converts DataFrames → HuggingFace DatasetDict with tokenized inputs.
    """
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

    def to_hf_dataset(df: pd.DataFrame) -> Dataset:
        return Dataset.from_dict({
            "text":   df["text"].tolist(),
            "labels": df["label_id"].tolist(),
        })

    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
        )

    raw_datasets = DatasetDict({
        "train": to_hf_dataset(train_df),
        "validation": to_hf_dataset(val_df),
    })

    tokenized = raw_datasets.map(tokenize_fn, batched=True)
    tokenized = tokenized.remove_columns(["text"])
    tokenized.set_format("torch")

    print(f"\n🔤 Tokenization complete. Max length: {MAX_SEQ_LENGTH}")
    print(f"   Features: {list(tokenized['train'].features.keys())}")
    return tokenized, tokenizer


# ── Entry point: run to preview data ──────────────────────
if __name__ == "__main__":
    df = load_raw_data()
    print("\n── Sample rows ──────────────────────────────────────")
    print(df[["text", "label"]].sample(6, random_state=42).to_string(index=False))
    train_df, val_df = split_data(df)
    tokenized, _ = tokenize_dataset(train_df, val_df)
    print("\n✅ Dataset pipeline working correctly!")
