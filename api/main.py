from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import re
import string
import nltk
from nltk.corpus import stopwords
import uvicorn
import os

app = FastAPI()

# Load ML model and vectorizer
try:
    with open("models/ml_model.pkl", "rb") as f:
        ml_model = pickle.load(f)
    with open("models/tfidf_vectorizer.pkl", "rb") as f:
        tfidf_vectorizer = pickle.load(f)
    print("ML models loaded successfully.")
except Exception as e:
    print(f"Error loading ML models: {e}")

# Load AI model and tokenizer
try:
    ai_model_path = "models/ai_model"
    if os.path.exists(ai_model_path):
        ai_tokenizer = DistilBertTokenizerFast.from_pretrained(ai_model_path)
        ai_model = DistilBertForSequenceClassification.from_pretrained(ai_model_path)
        ai_model.eval()
        print("AI models loaded successfully.")
    else:
        print("AI model path not found. AI predictions will be unavailable.")
except Exception as e:
    print(f"Error loading AI models: {e}")

STOP_WORDS = set(stopwords.words('english'))

def clean_text_fast(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[%s\d+]' % re.escape(string.punctuation), '', text)
    tokens = text.split()
    filtered_tokens = [word for word in tokens if word not in STOP_WORDS]
    return " ".join(filtered_tokens)

class NewsInput(BaseModel):
    title: str
    text: str

@app.post("/predict")
async def predict(news: NewsInput):
    combined_text = f"{news.title} {news.text}"
    cleaned_text = clean_text_fast(combined_text)
    
    # ML Prediction
    ml_pred = "Unknown"
    ml_prob = 0.0
    if ml_model and tfidf_vectorizer:
        vectorized_text = tfidf_vectorizer.transform([cleaned_text])
        ml_prob = ml_model.predict_proba(vectorized_text)[0][1]
        ml_pred = "REAL" if ml_prob > 0.5 else "FAKE"
    
    # AI Prediction
    ai_pred = "Unknown"
    ai_prob = 0.0
    if 'ai_model' in globals() and ai_model:
        inputs = ai_tokenizer(combined_text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = ai_model(**inputs)
            scores = torch.nn.functional.softmax(outputs.logits, dim=1)
            ai_prob = scores[0][1].item()
            ai_pred = "REAL" if ai_prob > 0.5 else "FAKE"
            
    return {
        "ml": {"label": ml_pred, "confidence": ml_prob if ml_pred == "REAL" else 1 - ml_prob},
        "ai": {"label": ai_pred, "confidence": ai_prob if ai_pred == "REAL" else 1 - ai_prob}
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
