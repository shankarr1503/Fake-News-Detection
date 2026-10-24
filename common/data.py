"""Loading the processed dataset and the train/test split shared by both models."""
import pandas as pd
from sklearn.model_selection import train_test_split

from common.paths import PROCESSED_DATA

LABEL_REAL = 1
LABEL_FAKE = 0
TEST_SIZE = 0.2
RANDOM_STATE = 42


def load_splits():
    """Return (train_df, test_df). Fixed seed + stratified, so both models are scored on the same rows."""
    if not PROCESSED_DATA.exists():
        raise FileNotFoundError(
            f"{PROCESSED_DATA} not found. Run `python -m scripts.preprocess` first."
        )
    df = pd.read_csv(PROCESSED_DATA)
    df = df.dropna(subset=["total_content", "cleaned_content", "label_num"])
    df["label_num"] = df["label_num"].astype(int)
    return train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df["label_num"]
    )
