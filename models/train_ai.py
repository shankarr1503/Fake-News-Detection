import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
import os

class NewsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def train_ai_model():
    print("Loading processed data...")
    if not os.path.exists("data/processed_news.csv"):
        print("Processed data not found.")
        return
        
    df = pd.read_csv("data/processed_news.csv")
    df = df.dropna(subset=['total_content', 'label_num']).sample(2000) # Use a subset for faster training demonstration
    
    texts = df['total_content'].tolist()
    labels = df['label_num'].tolist()
    
    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2)
    
    print("Loading tokenizer...")
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    
    print("Encoding data...")
    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)
    
    train_dataset = NewsDataset(train_encodings, train_labels)
    val_dataset = NewsDataset(val_encodings, val_labels)
    
    print("Initializing model...")
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
    
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=1,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="steps",
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    
    print("Training model (this will take time)...")
    trainer.train()
    
    print("Saving AI model...")
    model.save_pretrained("models/ai_model")
    tokenizer.save_pretrained("models/ai_model")
    print("Done!")

if __name__ == "__main__":
    train_ai_model()
