import argparse
import sys

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from common.data import RANDOM_STATE, load_splits
from common.model_meta import write_metadata
from common.paths import AI_MODEL_DIR, AI_MODEL_META, RESULTS_DIR

MAX_LENGTH = 512


class NewsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        # Unpadded; DataCollatorWithPadding pads each batch to its own longest example.
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, preds), "f1": f1_score(labels, preds)}


def train_ai_model(base_model, train_samples, epochs, min_accuracy):
    print("Loading processed data...")
    try:
        train_df, test_df = load_splits()
    except FileNotFoundError as e:
        sys.exit(str(e))

    # Subsample the training split only, to keep CPU training time reasonable.
    # The test split is the same one the ML model is scored on.
    if train_samples and train_samples < len(train_df):
        train_df = train_df.sample(train_samples, random_state=RANDOM_STATE)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    print("Encoding data...")
    train_encodings = tokenizer(train_df["total_content"].tolist(), truncation=True, max_length=MAX_LENGTH)
    test_encodings = tokenizer(test_df["total_content"].tolist(), truncation=True, max_length=MAX_LENGTH)

    train_dataset = NewsDataset(train_encodings, train_df["label_num"].tolist())
    test_dataset = NewsDataset(test_encodings, test_df["label_num"].tolist())

    print("Initializing model...")
    model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=2)

    training_args = TrainingArguments(
        output_dir=str(RESULTS_DIR),
        num_train_epochs=epochs,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=100,
        weight_decay=0.01,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="no",
        seed=RANDOM_STATE,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    print("Training model (this will take time)...")
    trainer.train()

    metrics = trainer.evaluate()
    accuracy = metrics["eval_accuracy"]
    print(f"Test accuracy: {accuracy:.4f}  F1: {metrics['eval_f1']:.4f}")

    # Don't overwrite a working model with a worse one.
    if accuracy < min_accuracy:
        sys.exit(f"Accuracy {accuracy:.4f} is below --min-accuracy {min_accuracy}; model not saved.")

    print("Saving AI model...")
    model.save_pretrained(AI_MODEL_DIR)
    tokenizer.save_pretrained(AI_MODEL_DIR)
    # save_pretrained writes model.safetensors as owner-only (0600); make the files readable
    # by the non-root user the Docker image runs as.
    for path in AI_MODEL_DIR.iterdir():
        if path.is_file():
            path.chmod(0o644)
    write_metadata(
        AI_MODEL_META,
        metrics={"accuracy": accuracy, "f1": metrics["eval_f1"], "test_size": len(test_df)},
        train_size=len(train_df),
        base_model=base_model,
        epochs=epochs,
    )
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT for fake news detection.")
    parser.add_argument("--base-model", default="distilbert-base-uncased")
    parser.add_argument(
        "--train-samples", type=int, default=2000, help="Training examples to use (0 = full training split)."
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "--min-accuracy", type=float, default=0.80, help="Refuse to save the model if test accuracy is below this."
    )
    args = parser.parse_args()
    train_ai_model(args.base_model, args.train_samples, args.epochs, args.min_accuracy)
