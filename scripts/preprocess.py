import sys

import pandas as pd

from common.labels import LABEL_FAKE, LABEL_REAL
from common.paths import PROCESSED_DATA, RAW_DATA
from common.text import clean_text, combine_title_text


def main():
    if not RAW_DATA.exists():
        sys.exit(f"{RAW_DATA} not found. Run `python -m scripts.download_data` first.")

    print("Loading dataset...")
    df = pd.read_csv(RAW_DATA)
    print(f"Dataset loaded. Shape: {df.shape}")

    missing = {"title", "text", "label"} - set(df.columns)
    if missing:
        sys.exit(f"Dataset is missing required columns: {sorted(missing)}")

    df["total_content"] = [combine_title_text(t, x) for t, x in zip(df["title"], df["text"], strict=True)]

    print("Cleaning text...")
    df["cleaned_content"] = df["total_content"].apply(clean_text)

    unknown = set(df["label"].dropna()) - {"REAL", "FAKE"}
    if unknown:
        sys.exit(f"Unexpected label values: {sorted(unknown)}")
    df["label_num"] = df["label"].map({"REAL": LABEL_REAL, "FAKE": LABEL_FAKE})

    print("Saving processed data...")
    df.to_csv(PROCESSED_DATA, index=False)
    print("Preprocessing complete!")


if __name__ == "__main__":
    main()
