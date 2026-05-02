# train.py
# ─────────────────────────────────────────────────────────────────
# Fine-tunes DistilBERT on the injection detection dataset.
# Run: python train.py
# Output: saved model → ./saved_model/
# ─────────────────────────────────────────────────────────────────

import os
import numpy as np
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

from config import (
    BASE_MODEL_NAME, MODEL_SAVE_DIR, EPOCHS,
    BATCH_SIZE, LEARNING_RATE, LABEL2ID, ID2LABEL,
    RANDOM_SEED
)
from dataset import load_raw_data, split_data, tokenize_dataset


# ── 1. Metrics function ───────────────────────────────────────────
def compute_metrics(eval_pred):
    """
    Called by Trainer after every epoch.
    Returns accuracy, precision, recall, F1.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary"
    )
    acc = accuracy_score(labels, predictions)

    return {
        "accuracy":  round(acc, 4),
        "f1":        round(f1, 4),
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
    }


# ── 2. Main training function ─────────────────────────────────────
def train():
    print("=" * 60)
    print("  🛡️  Prompt Injection Detector — Training")
    print("=" * 60)

    # ── Load & prepare data ───────────────────────────────────────
    print("\n📂 Step 1: Loading dataset...")
    df = load_raw_data()
    train_df, val_df = split_data(df)
    tokenized_datasets, tokenizer = tokenize_dataset(train_df, val_df)

    # ── Load pre-trained model ────────────────────────────────────
    print(f"\n🤖 Step 2: Loading base model: {BASE_MODEL_NAME}")
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_NAME,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    # ── Training arguments ────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=MODEL_SAVE_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_dir=os.path.join(MODEL_SAVE_DIR, "logs"),
        logging_steps=10,
        seed=RANDOM_SEED,
        report_to="none",              # disable wandb/mlflow
        push_to_hub=False,
    )

    # ── Data collator (handles dynamic padding) ───────────────────
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # ── Trainer ───────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # ── Train ─────────────────────────────────────────────────────
    print(f"\n🚀 Step 3: Training for {EPOCHS} epochs...")
    print(f"   Batch size : {BATCH_SIZE}")
    print(f"   LR         : {LEARNING_RATE}")
    print(f"   Train size : {len(tokenized_datasets['train'])}")
    print(f"   Val size   : {len(tokenized_datasets['validation'])}")

    trainer.train()

    # ── Evaluate ──────────────────────────────────────────────────
    print("\n📊 Step 4: Final evaluation...")
    results = trainer.evaluate()
    print("\n── Evaluation Results ─────────────────────────────────")
    for k, v in results.items():
        print(f"  {k:<30} {v}")

    # ── Confusion matrix ──────────────────────────────────────────
    preds_output = trainer.predict(tokenized_datasets["validation"])
    preds = np.argmax(preds_output.predictions, axis=-1)
    labels = preds_output.label_ids

    print("\n── Confusion Matrix ───────────────────────────────────")
    cm = confusion_matrix(labels, preds)
    print(f"              Predicted Safe  Predicted Injection")
    print(f"  Actual Safe      {cm[0][0]:<5}          {cm[0][1]}")
    print(f"  Actual Inject    {cm[1][0]:<5}          {cm[1][1]}")

    print("\n── Classification Report ──────────────────────────────")
    print(classification_report(labels, preds, target_names=["safe", "injection"]))

    # ── Save model ────────────────────────────────────────────────
    print(f"\n💾 Step 5: Saving model to {MODEL_SAVE_DIR}...")
    trainer.save_model(MODEL_SAVE_DIR)
    tokenizer.save_pretrained(MODEL_SAVE_DIR)

    print("\n✅ Training complete! Model saved.")
    print(f"   Run: python test_detector.py")
    print(f"   Or:  uvicorn api:app --reload")


# ── Entry point ───────────────────────────────────────────────────
if __name__ == "__main__":
    train()
