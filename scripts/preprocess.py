import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import os

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Pre-load stopwords
STOP_WORDS = set(stopwords.words('english'))

def clean_text_fast(text):
    if not isinstance(text, str):
        return ""
    # Lowercase and remove punctuation/numbers in one go
    text = text.lower()
    text = re.sub(r'[%s\d+]' % re.escape(string.punctuation), '', text)
    # Tokenize and remove stopwords
    tokens = text.split()
    filtered_tokens = [word for word in tokens if word not in STOP_WORDS]
    
    return " ".join(filtered_tokens)

if __name__ == "__main__":
    try:
        print("Checking for news.csv...")
        if not os.path.exists("data/news.csv"):
            print("news.csv not found!")
            exit(1)
            
        print("Loading dataset...")
        df = pd.read_csv("data/news.csv")
        print(f"Dataset loaded. Shape: {df.shape}")
        
        # Combine title and text
        df['total_content'] = df['title'].fillna('') + " " + df['text'].fillna('')
        
        print("Cleaning text...")
        df['cleaned_content'] = df['total_content'].apply(clean_text_fast)
        
        # Map labels
        if 'label' in df.columns:
            df['label_num'] = df['label'].apply(lambda x: 1 if x == 'REAL' else 0)
        
        print("Saving processed data...")
        df.to_csv("data/processed_news.csv", index=False)
        print("Preprocessing complete!")
    except Exception as e:
        print(f"Error: {e}")
