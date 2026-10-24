import argparse
import sys

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline

from common.data import load_splits
from common.model_meta import write_metadata
from common.paths import ML_MODEL, ML_MODEL_META


def train_ml_model(min_accuracy):
    print("Loading processed data...")
    try:
        train_df, test_df = load_splits()
    except FileNotFoundError as e:
        sys.exit(str(e))

    # Vectorizer and classifier are saved together so they can never get out of sync.
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=5000)),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )

    print("Training TF-IDF + Logistic Regression model...")
    pipeline.fit(train_df["cleaned_content"], train_df["label_num"])

    print("Evaluating model...")
    y_true = test_df["label_num"]
    y_pred = pipeline.predict(test_df["cleaned_content"])
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["FAKE", "REAL"]))
    print("Confusion Matrix (rows = true FAKE/REAL):")
    print(confusion_matrix(y_true, y_pred))

    # Don't overwrite a working model with a worse one.
    if accuracy < min_accuracy:
        sys.exit(f"Accuracy {accuracy:.4f} is below --min-accuracy {min_accuracy}; model not saved.")

    print("Saving model...")
    ML_MODEL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, ML_MODEL)
    write_metadata(
        ML_MODEL_META,
        metrics={"accuracy": accuracy, "f1": f1_score(y_true, y_pred), "test_size": len(test_df)},
        train_size=len(train_df),
    )
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the TF-IDF + Logistic Regression model.")
    parser.add_argument(
        "--min-accuracy", type=float, default=0.85, help="Refuse to save the model if test accuracy is below this."
    )
    args = parser.parse_args()
    train_ml_model(args.min_accuracy)
