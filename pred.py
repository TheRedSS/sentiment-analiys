import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

def load_simple_model():
    """Basit modeli yükle"""
    try:
        model = tf.keras.models.load_model('simple_model.h5')
        with open('simple_tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)
        with open('simple_mlb.pkl', 'rb') as f:
            mlb = pickle.load(f)
        return model, tokenizer, mlb
    except:
        print("Model dosyaları bulunamadı. Önce main.py çalıştırın.")
        return None, None, None

def predict_emotion(text, model, tokenizer, mlb):
    """Tek metin için duygu tahmini"""
    # Metni işle
    text = text.lower()
    
    # Sequence'e çevir
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=50, padding='post')
    
    # Tahmin yap
    prediction = model.predict(padded)
    
    # En yüksek duyguyu bul
    max_idx = np.argmax(prediction[0])
    emotion = mlb.classes_[max_idx]
    confidence = prediction[0][max_idx]
    
    return emotion, confidence

def main():
    """Ana fonksiyon"""
    print("Basit Duygu Analizi Tahmini")
    
    # Model yükle
    model, tokenizer, mlb = load_simple_model()
    
    if model is None:
        return
    
    # Test metinleri
    test_texts = [
        "I am very happy today!",
        "This makes me sad.",
        "I'm angry about this situation.",
        "I'm scared of the dark.",
        "Wow, that's amazing!",
        "This is disgusting."
    ]
    
    print("\nTahmin Sonuçları:")
    print("=" * 30)
    
    for text in test_texts:
        emotion, confidence = predict_emotion(text, model, tokenizer, mlb)
        print(f"Metin: {text}")
        print(f"Duygu: {emotion} (Güven: {confidence:.3f})")
        print("-" * 30)

if __name__ == "__main__":
    main()
