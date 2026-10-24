import sys

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

from common.data import load_splits
from common.paths import ML_MODEL, MODELS_DIR


def train_ml_model():
    print("Loading processed data...")
    try:
        train_df, test_df = load_splits()
    except FileNotFoundError as e:
        sys.exit(str(e))

    # Vectorizer and classifier are saved together so they can never get out of sync.
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000)),
        ("clf", LogisticRegression(max_iter=1000)),
    ])

    print("Training TF-IDF + Logistic Regression model...")
    pipeline.fit(train_df["cleaned_content"], train_df["label_num"])

    print("Evaluating model...")
    y_pred = pipeline.predict(test_df["cleaned_content"])
    print(f"Accuracy: {accuracy_score(test_df['label_num'], y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(test_df["label_num"], y_pred, target_names=["FAKE", "REAL"]))
    print("Confusion Matrix (rows = true FAKE/REAL):")
    print(confusion_matrix(test_df["label_num"], y_pred))

    print("Saving model...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, ML_MODEL)
    print("Done!")


if __name__ == "__main__":
    train_ml_model()
