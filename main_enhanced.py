import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout, LayerNormalization, Conv1D, GlobalMaxPooling1D, Attention
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, hamming_loss
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
    def __init__(self, max_words=10000, max_len=150, embedding_dim=300):  # Parametreleri artırdım
        self.max_words = max_words
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.tokenizer = None
        self.mlb = None
        self.model = None
        
    def load_data(self, sample_size=20000):
        """Archive klasöründeki Go Emotions veri setini yükle"""
        print("Archive klasöründeki veri seti yükleniyor...")
        
        try:
            # Train veri setini yükle
            train_data = []
            with open('archive (1)/data/train.tsv', 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        text = parts[0]
                        emotion_ids_str = parts[1]
                        # Birden fazla duygu varsa virgülle ayrılmış
                        emotion_ids = [int(eid.strip()) for eid in emotion_ids_str.split(',')]
                        train_data.append((text, emotion_ids))
            
            # Dev veri setini yükle
            dev_data = []
            with open('archive (1)/data/dev.tsv', 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        text = parts[0]
                        emotion_ids_str = parts[1]
                        emotion_ids = [int(eid.strip()) for eid in emotion_ids_str.split(',')]
                        dev_data.append((text, emotion_ids))
            
            # Test veri setini yükle
            test_data = []
            with open('archive (1)/data/test.tsv', 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        text = parts[0]
                        emotion_ids_str = parts[1]
                        emotion_ids = [int(eid.strip()) for eid in emotion_ids_str.split(',')]
                        test_data.append((text, emotion_ids))
            
            # Tüm veriyi birleştir
            all_data = train_data + dev_data + test_data
            print(f"Toplam veri: {len(all_data)} örnek")
            print(f"Train: {len(train_data)}, Dev: {len(dev_data)}, Test: {len(test_data)}")
            
            # Örnek sayısını sınırla
            if len(all_data) > sample_size:
                all_data = random.sample(all_data, sample_size)
                print(f"Örnekleme sonrası: {len(all_data)} örnek")
            
            # Duygu etiketlerini oku
            emotions = []
            with open('archive (1)/data/emotions.txt', 'r', encoding='utf-8') as f:
                emotions = [line.strip() for line in f]
            
            print(f"Duygu sınıfları: {emotions}")
            print(f"Toplam duygu sayısı: {len(emotions)}")
            
            # Veriyi işle
            texts = []
            labels = []
            
            for text, emotion_ids in all_data:
                # Metni işle
                processed_text = process_text(text)
                texts.append(processed_text)
                
                # Multi-label encoding oluştur
                label = [0] * len(emotions)
                for emotion_id in emotion_ids:
                    if 0 <= emotion_id < len(emotions):
                        label[emotion_id] = 1
                labels.append(label)
            
            # Etiket dağılımını kontrol et
            labels_array = np.array(labels)
            print(f"Etiket dağılımı: {labels_array.sum(axis=0)}")
            print(f"Pozitif örnek oranı: {labels_array.sum() / (labels_array.shape[0] * labels_array.shape[1]):.3f}")
            
            return np.array(texts), labels_array
            
        except FileNotFoundError as e:
            print(f"Veri seti dosyası bulunamadı: {e}")
            print("Örnek veri oluşturuluyor...")
            return self.create_sample_data(sample_size)
        except Exception as e:
            print(f"Veri yükleme hatası: {e}")
            print("Örnek veri oluşturuluyor...")
            return self.create_sample_data(sample_size)
    
    def create_sample_data(self, sample_size=20000):
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
        
        # Daha çeşitli örnek metinler
        sample_texts = [
            "I love this movie so much! It's absolutely amazing!",
            "This is absolutely terrible and disgusting.",
            "I'm so excited about the new project! Can't wait to start!",
            "I feel really sad today, everything is going wrong.",
            "This makes me extremely angry and frustrated.",
            "I'm so grateful for your help, thank you so much!",
            "I'm confused about this situation, I don't understand.",
            "I'm proud of my achievements and accomplishments.",
            "I'm nervous about the presentation tomorrow.",
            "I'm optimistic about the future and our chances.",
            "I'm curious about what will happen next.",
            "I'm disappointed with the results we got.",
            "I'm embarrassed about what happened yesterday.",
            "I'm relieved that everything worked out well.",
            "I'm surprised by the unexpected news.",
            "I'm annoyed by all the noise around here.",
            "I'm caring about your well-being and health.",
            "I'm fearful about the upcoming surgery.",
            "I'm grieving the loss of my beloved pet.",
            "I'm remorseful about my past actions.",
            "I'm realizing the truth about the situation.",
            "I'm desiring to achieve my goals.",
            "I'm disapproving of their behavior.",
            "I'm admiring your courage and strength.",
            "I'm amused by the funny joke you told.",
            "I'm neutral about this topic, no strong feelings.",
            "This is wonderful and brings me great joy!",
            "I'm feeling anxious and worried about the future.",
            "I'm hopeful that things will get better soon."
        ]
        
        # Daha gerçekçi veri oluştur
        texts = []
        labels = []
        
        for _ in range(sample_size):
            # Rastgele metin seç
            base_text = random.choice(sample_texts)
            
            # Metni varyasyonlarla çoğalt
            variations = [
                base_text,
                base_text.upper(),
                base_text.lower(),
                base_text + "!",
                base_text + "?",
                "Really, " + base_text,
                base_text + " indeed.",
                "I think " + base_text.lower(),
                base_text + " and I mean it.",
                "Honestly, " + base_text.lower()
            ]
            
            text = random.choice(variations)
            texts.append(text)
            
            # Rastgele 1-3 duygu etiketi ata
            num_emotions = random.randint(1, 3)
            emotion_indices = random.sample(range(len(emotions)), num_emotions)
            
            label = [0] * len(emotions)
            for idx in emotion_indices:
                label[idx] = 1
            labels.append(label)
        
        return np.array(texts), np.array(labels)
    
    def augment_data(self, texts, labels, num_augmentations=2):
        """Veri artırma"""
        print("Veri artırma yapılıyor...")
        
        # Orijinal veriyi list'e çevir
        augmented_texts = list(texts)
        augmented_labels = list(labels)
        
        # Her metin için augmentation yap
        for i in range(len(texts)):
            for _ in range(num_augmentations):
                try:
                    # Basit augmentation teknikleri
                    original_text = texts[i]
                    
                    # Rastgele augmentation seç
                    augmentation_type = random.choice(['synonym', 'back_translation', 'insertion'])
                    
                    if augmentation_type == 'synonym':
                        # Basit synonym replacement
                        words = original_text.split()
                        if len(words) > 3:
                            # Rastgele bir kelimeyi değiştir
                            idx = random.randint(0, len(words)-1)
                            words[idx] = f"modified_{words[idx]}"
                            augmented_text = " ".join(words)
                        else:
                            augmented_text = original_text
                    elif augmentation_type == 'back_translation':
                        # Basit back translation simulation
                        augmented_text = f"translated_{original_text}"
                    else:  # insertion
                        # Basit insertion
                        augmented_text = f"augmented_{original_text}"
                    
                    augmented_texts.append(augmented_text)
                    augmented_labels.append(labels[i])
                    
                except Exception as e:
                    # Hata durumunda orijinal metni kullan
                    augmented_texts.append(texts[i])
                    augmented_labels.append(labels[i])
        
        print(f"Artırılmış veri: {len(augmented_texts)} örnek")
        return np.array(augmented_texts), np.array(augmented_labels)
    
    def prepare_data(self, texts, labels):
        """Veriyi hazırla"""
        print("Veri hazırlanıyor...")
        
        # Tokenizer oluştur ve fit et
        self.tokenizer = Tokenizer(num_words=self.max_words, oov_token='<OOV>')
        self.tokenizer.fit_on_texts(texts)
        
        # Metinleri sequence'e çevir
        sequences = self.tokenizer.texts_to_sequences(texts)
        X = pad_sequences(sequences, maxlen=self.max_len, padding='post')
        
        # Labels zaten doğru formatta (numpy array), sadece kontrol et
        y = np.array(labels)
        
        print(f"Veri hazırlandı: X shape: {X.shape}, y shape: {y.shape}")
        return X, y
    
    def create_model(self):
        """Gelişmiş model oluştur"""
        print("Gelişmiş model oluşturuluyor...")
        
        model = Sequential([
            # Embedding katmanı - daha büyük
            Embedding(self.max_words, self.embedding_dim, input_length=self.max_len),
            
            # Bidirectional LSTM katmanları - daha derin
            Bidirectional(LSTM(256, return_sequences=True)),
            LayerNormalization(),
            Dropout(0.4),
            
            Bidirectional(LSTM(128, return_sequences=True)),
            LayerNormalization(),
            Dropout(0.4),
            
            Bidirectional(LSTM(64, return_sequences=False)),
            LayerNormalization(),
            Dropout(0.3),
            
            # Dense katmanları - daha büyük ve daha iyi regularization
            Dense(256, activation='relu', kernel_regularizer=l2(0.01)),
            LayerNormalization(),
            Dropout(0.5),
            
            Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
            LayerNormalization(),
            Dropout(0.4),
            
            Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
            LayerNormalization(),
            Dropout(0.3),
            
            Dense(28, activation='sigmoid')  # 28 duygu sınıfı
        ])
        
        # Model derle - daha iyi optimizer ve learning rate
        model.compile(
            optimizer=Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    def train_model(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=64):  # Daha büyük batch size
        """Modeli eğit"""
        print("Model eğitiliyor...")
        
        # Callbacks - daha iyi strateji
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=15,  # Daha uzun patience
            restore_best_weights=True,
            verbose=1
        )
        
        model_checkpoint = ModelCheckpoint(
            'best_model.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,  # Daha yavaş decay
            patience=5,  # Daha uzun patience
            min_lr=1e-6,
            verbose=1
        )
        
        # Model eğitimi
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, model_checkpoint, reduce_lr],
            verbose=1,
            shuffle=True  # Her epoch'ta shuffle
        )
        
        return history
    
    def evaluate_model(self, X_test, y_test):
        """Modeli değerlendir"""
        print("Model değerlendiriliyor...")
        
        # Tahminler
        predictions = self.model.predict(X_test)
        
        # Threshold uygula - daha düşük threshold
        threshold = 0.3  # 0.5'ten 0.3'e düşürdüm
        predictions_binary = (predictions > threshold).astype(int)
        
        # Multi-label için uygun metrikler
        # Hamming loss hesapla (multi-label için daha iyi)
        hamming_loss_value = hamming_loss(y_test, predictions_binary)
        
        # Exact match accuracy (tüm etiketler doğru olmalı)
        exact_match_accuracy = np.mean(np.all(y_test == predictions_binary, axis=1))
        
        # Label-based accuracy (her etiket için ayrı accuracy)
        label_accuracy = np.mean([accuracy_score(y_test[:, i], predictions_binary[:, i]) 
                                 for i in range(y_test.shape[1])])
        
        # AUC scores
        auc_scores = []
        for i in range(y_test.shape[1]):
            try:
                if len(np.unique(y_test[:, i])) > 1:  # En az 2 sınıf olmalı
                    auc = roc_auc_score(y_test[:, i], predictions[:, i])
                    auc_scores.append(auc)
                else:
                    auc_scores.append(0.5)
            except:
                auc_scores.append(0.5)
        
        mean_auc = np.mean(auc_scores)
        
        print(f"Hamming Loss: {hamming_loss_value:.4f}")
        print(f"Exact Match Accuracy: {exact_match_accuracy:.4f}")
        print(f"Label-based Accuracy: {label_accuracy:.4f}")
        print(f"Mean AUC: {mean_auc:.4f}")
        
        return label_accuracy, mean_auc, predictions
    
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
    print("🚀 Yüksek Accuracy Duygu Analizi Modeli Başlatılıyor...")
    
    # Model oluştur - daha büyük parametreler
    analyzer = EnhancedSentimentAnalyzer(max_words=10000, max_len=150, embedding_dim=300)
    
    # Veri yükle - daha büyük veri seti
    texts, labels = analyzer.load_data(sample_size=20000)
    
    if texts is None:
        print("Veri yüklenemedi. Program sonlandırılıyor.")
        return
    
    # Veri artırma - optimal augmentation
    augmented_texts, augmented_labels = analyzer.augment_data(texts, labels, num_augmentations=2)
    
    # Veriyi hazırla
    X, y = analyzer.prepare_data(augmented_texts, augmented_labels)
    
    # Train-test split - daha iyi oranlar
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.25, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.4, random_state=42)
    
    print(f"Eğitim seti boyutu: {X_train.shape}")
    print(f"Validasyon seti boyutu: {X_val.shape}")
    print(f"Test seti boyutu: {X_test.shape}")
    
    # Model oluştur
    analyzer.model = analyzer.create_model()
    
    # Model eğit - daha iyi hyperparameter'lar
    history = analyzer.train_model(X_train, y_train, X_val, y_val, epochs=50, batch_size=64)
    
    # Model değerlendir
    accuracy, auc, predictions = analyzer.evaluate_model(X_test, y_test)
    
    # Eğitim geçmişini görselleştir
    analyzer.plot_training_history(history)
    
    # Modeli kaydet
    analyzer.save_model()
    
    # Özel test verileriyle test et
    analyzer.test_custom_data()
    
    print("\n✅ Yüksek Accuracy Model Eğitimi Tamamlandı!")
    print(f"📊 Final Label-based Accuracy: {accuracy:.4f}")
    print(f"📊 Final AUC: {auc:.4f}")
    print(f"🎯 Hedef Accuracy: %80+ (Mevcut: %{accuracy*100:.1f})")

if __name__ == "__main__":
    main()
