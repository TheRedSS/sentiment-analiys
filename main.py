import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import pickle
import random

def create_simple_data():
    """Basit örnek veri oluştur"""
    emotions = ['joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust']
    
    sample_texts = [
        "I am so happy today!",
        "This makes me very sad.",
        "I'm extremely angry about this.",
        "I'm scared of what might happen.",
        "Wow, that's surprising!",
        "This is disgusting.",
        "I love this movie!",
        "I feel terrible today.",
        "This is outrageous!",
        "I'm terrified of spiders.",
        "Amazing news!",
        "This is revolting."
    ]
    
    texts = []
    labels = []
    
    for _ in range(1000):
        text = random.choice(sample_texts)
        emotion = random.choice(emotions)
        texts.append(text)
        labels.append([emotion])
    
    return texts, labels

def main():
    """Basit duygu analizi modeli"""
    print("Basit Duygu Analizi Modeli")
    
    # Veri oluştur
    texts, labels = create_simple_data()
    
    # Multi-label encoding
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(labels)
    
    # Tokenizer
    tokenizer = Tokenizer(num_words=1000, oov_token='<OOV>')
    tokenizer.fit_on_texts(texts)
    
    # Sequence'e çevir
    sequences = tokenizer.texts_to_sequences(texts)
    X = pad_sequences(sequences, maxlen=50, padding='post')
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model oluştur
    model = Sequential([
        Embedding(1000, 64, input_length=50),
        LSTM(64, return_sequences=True),
        LSTM(32),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(len(mlb.classes_), activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Model eğit
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
    
    # Model kaydet
    model.save('simple_model.h5')
    
    # Tokenizer ve mlb kaydet
    with open('simple_tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
    with open('simple_mlb.pkl', 'wb') as f:
        pickle.dump(mlb, f)
    
    print("Basit model eğitimi tamamlandı!")

if __name__ == "__main__":
    main()
