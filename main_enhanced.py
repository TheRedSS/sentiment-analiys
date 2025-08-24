import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, GRU, Dense, Dropout, LayerNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from text_processor import process_text, augment_text
import random

# GPU kullanımını optimize et
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

class EnhancedSentimentAnalyzer:
    def __init__(self, max_words=10000, max_len=100, embedding_dim=256):
        self.max_words = max_words
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.tokenizer = None
        self.mlb = None
        self.model = None
        
    def load_data(self, sample_size=10000):
        """Go Emotions veri setini yükle"""
        print("Veri seti yükleniyor...")
        
        # Veri setini oku
        try:
            df = pd.read_csv('go_emotions_dataset.csv')
            print("Gerçek veri seti yüklendi.")
            
            # Veri setini temizle
            df = df.dropna()
            
            # Örnek sayısını sınırla
            if len(df) > sample_size:
                df = df.sample(n=sample_size, random_state=42)
            
            # Metinleri işle
            print("Metinler işleniyor...")
            df['processed_text'] = df['text'].apply(process_text)
            
            # Duygu etiketlerini işle
            emotion_columns = [col for col in df.columns if col not in ['text', 'processed_text']]
            emotions = df[emotion_columns].values
            
            return df['processed_text'].values, emotions
            
        except FileNotFoundError:
            print("Veri seti bulunamadı. Örnek veri oluşturuluyor...")
            return self.create_sample_data(sample_size)
    
    def create_sample_data(self, sample_size=10000):
        """Örnek veri oluştur"""
        print("Örnek veri oluşturuluyor...")
        
        # 27 duygu sınıfı
        emotions = [
            'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
            'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
            'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
            'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
            'relief', 'remorse', 'sadness', 'surprise', 'neutral'
        ]
        
        # Örnek metinler
        sample_texts = [
            "I love this movie so much!",
            "This is absolutely terrible.",
            "I'm so excited about the new project!",
            "I feel really sad today.",
            "This makes me angry.",
            "I'm grateful for your help.",
            "I'm confused about this situation.",
            "I'm proud of my achievements.",
            "I'm nervous about the presentation.",
            "I'm optimistic about the future.",
            "I'm disappointed with the results.",
            "I'm curious about what happens next.",
            "I'm embarrassed by my mistake.",
            "I'm relieved that it's over.",
            "I'm surprised by the news.",
            "I'm annoyed by the noise.",
            "I'm caring about your feelings.",
            "I'm disgusted by this behavior.",
            "I'm fearful of what might happen.",
            "I'm grieving the loss.",
            "I'm hopeful for better days.",
            "I'm jealous of their success.",
            "I'm lonely without you.",
            "I'm nostalgic for the past.",
            "I'm overwhelmed by work.",
            "I'm peaceful in nature.",
            "I'm satisfied with the outcome."
        ]
        
        texts = []
        emotion_labels = []
        
        for _ in range(sample_size):
            text = random.choice(sample_texts)
            # Rastgele 1-3 duygu seç
            num_emotions = random.randint(1, 3)
            selected_emotions = random.sample(emotions, num_emotions)
            
            texts.append(text)
            emotion_labels.append(selected_emotions)
        
        # Metinleri işle
        processed_texts = [process_text(text) for text in texts]
        
        # Multi-label encoding
        mlb = MultiLabelBinarizer()
        emotion_vectors = mlb.fit_transform(emotion_labels)
        
        return processed_texts, emotion_vectors
    
    def augment_data(self, texts, labels, num_augmentations=3):
        """Veri artırma uygula"""
        print("Veri artırma uygulanıyor...")
        
        augmented_texts = []
        augmented_labels = []
        
        for text, label in zip(texts, labels):
            # Orijinal veri
            augmented_texts.append(text)
            augmented_labels.append(label)
            
            # Artırılmış veri
            augmented_versions = augment_text(text, num_augmentations)
            for aug_text in augmented_versions[1:]:  # İlk eleman orijinal metin
                augmented_texts.append(aug_text)
                augmented_labels.append(label)
        
        return np.array(augmented_texts), np.array(augmented_labels)
    
    def prepare_data(self, texts, labels):
        """Veriyi model için hazırla"""
        print("Veri hazırlanıyor...")
        
        # Tokenizer oluştur ve eğit
        self.tokenizer = Tokenizer(num_words=self.max_words, oov_token='<OOV>')
        self.tokenizer.fit_on_texts(texts)
        
        # Metinleri sequence'e çevir
        sequences = self.tokenizer.texts_to_sequences(texts)
        X = pad_sequences(sequences, maxlen=self.max_len, padding='post')
        
        # Labels zaten doğru formatta (numpy array), sadece MultiLabelBinarizer'ı ayarla
        if isinstance(labels, list):
            # Eğer labels liste ise (örnek veri için)
            self.mlb = MultiLabelBinarizer()
            y = self.mlb.fit_transform(labels)
        else:
            # Eğer labels numpy array ise (gerçek veri için)
            y = labels
            # MultiLabelBinarizer için sınıf isimlerini ayarla - veri setinden al
            emotion_names = [
                'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
                'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
                'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
                'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
                'relief', 'remorse', 'sadness', 'surprise', 'neutral'
            ]
            self.mlb = MultiLabelBinarizer()
            self.mlb.fit([emotion_names])  # Sınıf isimlerini ayarla
        
        return X, y
    
    def create_model(self):
        """Gelişmiş model mimarisi oluştur"""
        print("Model oluşturuluyor...")
        
        model = Sequential([
            # Embedding katmanı
            Embedding(self.max_words, self.embedding_dim, input_length=self.max_len),
            
            # İlk Bidirectional GRU katmanı
            Bidirectional(GRU(256, return_sequences=True)),
            LayerNormalization(),
            Dropout(0.3),
            
            # İkinci Bidirectional GRU katmanı
            Bidirectional(GRU(128, return_sequences=True)),
            LayerNormalization(),
            Dropout(0.3),
            
            # Üçüncü Bidirectional GRU katmanı
            Bidirectional(GRU(64, return_sequences=False)),
            LayerNormalization(),
            Dropout(0.3),
            
            # Dense katmanları
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(28, activation='sigmoid')  # 28 duygu sınıfı
        ])
        
        # Model derle
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_model(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        """Modeli eğit"""
        print("Model eğitiliyor...")
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        model_checkpoint = ModelCheckpoint(
            'best_model.h5',
            monitor='val_loss',
            save_best_only=True
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7
        )
        
        # Model eğitimi
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, model_checkpoint, reduce_lr],
            verbose=1
        )
        
        return history
    
    def evaluate_model(self, X_test, y_test):
        """Modeli değerlendir"""
        print("Model değerlendiriliyor...")
        
        # Tahminler
        predictions = self.model.predict(X_test)
        
        # Threshold uygula
        threshold = 0.5
        predictions_binary = (predictions > threshold).astype(int)
        
        # Metrikler
        accuracy = accuracy_score(y_test, predictions_binary)
        auc_scores = []
        
        for i in range(y_test.shape[1]):
            try:
                auc = roc_auc_score(y_test[:, i], predictions[:, i])
                auc_scores.append(auc)
            except:
                auc_scores.append(0.5)
        
        mean_auc = np.mean(auc_scores)
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Mean AUC: {mean_auc:.4f}")
        
        return accuracy, mean_auc, predictions
    
    def plot_training_history(self, history):
        """Eğitim geçmişini görselleştir"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss grafiği
        ax1.plot(history.history['loss'], label='Training Loss')
        ax1.plot(history.history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy grafiği
        ax2.plot(history.history['accuracy'], label='Training Accuracy')
        ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model(self):
        """Modeli kaydet"""
        print("Model kaydediliyor...")
        
        # Model kaydet
        self.model.save('sentiment_analysis_model.h5')
        
        # Tokenizer kaydet
        with open('tokenizer.pkl', 'wb') as f:
            pickle.dump(self.tokenizer, f)
        
        # Multi-label binarizer kaydet
        with open('mlb.pkl', 'wb') as f:
            pickle.dump(self.mlb, f)
        
        print("Model başarıyla kaydedildi!")
    
    def test_custom_data(self):
        """Özel test verileriyle modeli test et"""
        print("Özel test verileriyle model test ediliyor...")
        
        custom_texts = [
            "I am feeling very happy today!",
            "This situation makes me extremely angry.",
            "I'm so grateful for your help.",
            "I'm feeling quite sad and disappointed.",
            "This is absolutely amazing and exciting!",
            "I'm nervous about the upcoming presentation.",
            "I'm proud of what we accomplished together.",
            "This is disgusting and unacceptable.",
            "I'm curious about what will happen next.",
            "I'm relieved that everything worked out."
        ]
        
        # Metinleri işle
        processed_texts = [process_text(text) for text in custom_texts]
        
        # Sequence'e çevir
        sequences = self.tokenizer.texts_to_sequences(processed_texts)
        X_custom = pad_sequences(sequences, maxlen=self.max_len, padding='post')
        
        # Tahmin yap
        predictions = self.model.predict(X_custom)
        
        # Sonuçları göster
        print("\nÖzel Test Sonuçları:")
        print("=" * 50)
        
        for i, (text, pred) in enumerate(zip(custom_texts, predictions)):
            print(f"\n{i+1}. Metin: {text}")
            print("Tespit edilen duygular:")
            
            # En yüksek 3 duyguyu göster
            top_indices = np.argsort(pred)[-3:][::-1]
            for idx in top_indices:
                emotion = self.mlb.classes_[idx]
                confidence = pred[idx]
                if confidence > 0.3:  # Threshold
                    print(f"  - {emotion}: {confidence:.3f}")

def main():
    """Ana fonksiyon"""
    print("🚀 Gelişmiş Duygu Analizi Modeli Başlatılıyor...")
    
    # Model oluştur
    analyzer = EnhancedSentimentAnalyzer()
    
    # Veri yükle
    texts, labels = analyzer.load_data(sample_size=10000)
    
    if texts is None:
        print("Veri yüklenemedi. Program sonlandırılıyor.")
        return
    
    # Veri artırma
    augmented_texts, augmented_labels = analyzer.augment_data(texts, labels, num_augmentations=3)
    
    # Veriyi hazırla
    X, y = analyzer.prepare_data(augmented_texts, augmented_labels)
    
    # Train-test split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    print(f"Eğitim seti boyutu: {X_train.shape}")
    print(f"Validasyon seti boyutu: {X_val.shape}")
    print(f"Test seti boyutu: {X_test.shape}")
    
    # Model oluştur
    analyzer.model = analyzer.create_model()
    
    # Model eğit
    history = analyzer.train_model(X_train, y_train, X_val, y_val, epochs=50, batch_size=32)
    
    # Model değerlendir
    accuracy, auc, predictions = analyzer.evaluate_model(X_test, y_test)
    
    # Eğitim geçmişini görselleştir
    analyzer.plot_training_history(history)
    
    # Modeli kaydet
    analyzer.save_model()
    
    # Özel test verileriyle test et
    analyzer.test_custom_data()
    
    print("\n✅ Model eğitimi tamamlandı!")
    print(f"📊 Final Accuracy: {accuracy:.4f}")
    print(f"📊 Final AUC: {auc:.4f}")

if __name__ == "__main__":
    main()
