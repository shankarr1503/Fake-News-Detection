import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import os

def train_ml_model():
    print("Loading processed data...")
    if not os.path.exists("data/processed_news.csv"):
        print("Processed data not found. Please run preprocess.py first.")
        return
        
    df = pd.read_csv("data/processed_news.csv")
    df = df.dropna(subset=['cleaned_content', 'label_num'])
    
    X = df['cleaned_content']
    y = df['label_num']
    
    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Vectorizing text using TF-IDF...")
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    
    print("Training Logistic Regression model...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, y_train)
    
    print("Evaluating model...")
    y_pred = model.predict(X_test_tfidf)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("Saving model and vectorizer...")
    if not os.path.exists("models"):
        os.makedirs("models")
        
    with open("models/ml_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("models/tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(tfidf_vectorizer, f)
    
    print("Done!")

if __name__ == "__main__":
    train_ml_model()
