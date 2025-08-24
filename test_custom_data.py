import numpy as np
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from text_processor import process_text
from tensorflow.keras.preprocessing.sequence import pad_sequences

class CustomDataTester:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.mlb = None
        self.max_len = 100
        
    def load_model(self):
        """Eğitilmiş modeli yükle"""
        try:
            # Model yükle
            self.model = tf.keras.models.load_model('sentiment_analysis_model.h5')
            
            # Tokenizer yükle
            with open('tokenizer.pkl', 'rb') as f:
                self.tokenizer = pickle.load(f)
            
            # Multi-label binarizer yükle
            with open('mlb.pkl', 'rb') as f:
                self.mlb = pickle.load(f)
                
            print("✅ Model başarıyla yüklendi!")
            return True
            
        except FileNotFoundError as e:
            print(f"❌ Model dosyası bulunamadı: {e}")
            print("Lütfen önce 'main_enhanced.py' scriptini çalıştırın.")
            return False
        except Exception as e:
            print(f"❌ Model yükleme hatası: {e}")
            return False
    
    def predict_emotions(self, texts):
        """Metinlerdeki duyguları tahmin et"""
        if self.model is None:
            print("❌ Model yüklenmemiş!")
            return None
        
        # Metinleri işle
        processed_texts = [process_text(text) for text in texts]
        
        # Sequence'e çevir
        sequences = self.tokenizer.texts_to_sequences(processed_texts)
        X = pad_sequences(sequences, maxlen=self.max_len, padding='post')
        
        # Tahmin yap
        predictions = self.model.predict(X)
        
        return predictions
    
    def analyze_predictions(self, texts, predictions, threshold=0.3):
        """Tahminleri analiz et"""
        results = []
        
        for i, (text, pred) in enumerate(zip(texts, predictions)):
            # En yüksek duyguları bul
            top_indices = np.argsort(pred)[-3:][::-1]
            
            detected_emotions = []
            for idx in top_indices:
                emotion = self.mlb.classes_[idx]
                confidence = pred[idx]
                if confidence > threshold:
                    detected_emotions.append({
                        'emotion': emotion,
                        'confidence': confidence
                    })
            
            results.append({
                'text': text,
                'emotions': detected_emotions,
                'predictions': pred
            })
        
        return results
    
    def create_heatmap(self, texts, predictions):
        """Duygu tahminleri için heatmap oluştur"""
        # Duygu isimlerini al
        emotion_names = self.mlb.classes_
        
        # Heatmap verisi oluştur
        heatmap_data = predictions.T  # Transpose to get emotions as rows
        
        # Plot oluştur
        plt.figure(figsize=(15, 10))
        sns.heatmap(heatmap_data, 
                   xticklabels=[f"Text {i+1}" for i in range(len(texts))],
                   yticklabels=emotion_names,
                   cmap='YlOrRd',
                   annot=True,
                   fmt='.2f',
                   cbar_kws={'label': 'Confidence Score'})
        
        plt.title('Duygu Analizi Sonuçları - Heatmap', fontsize=16, fontweight='bold')
        plt.xlabel('Metinler', fontsize=12)
        plt.ylabel('Duygular', fontsize=12)
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('custom_data_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_emotion_frequency_chart(self, results):
        """Duygu frekans grafiği oluştur"""
        # Duygu sayılarını hesapla
        emotion_counts = {}
        
        for result in results:
            for emotion_info in result['emotions']:
                emotion = emotion_info['emotion']
                if emotion in emotion_counts:
                    emotion_counts[emotion] += 1
                else:
                    emotion_counts[emotion] = 1
        
        # Grafik oluştur
        if emotion_counts:
            emotions = list(emotion_counts.keys())
            counts = list(emotion_counts.values())
            
            plt.figure(figsize=(12, 8))
            bars = plt.bar(emotions, counts, color='skyblue', edgecolor='navy', alpha=0.7)
            
            # Bar değerlerini göster
            for bar, count in zip(bars, counts):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        str(count), ha='center', va='bottom', fontweight='bold')
            
            plt.title('Tespit Edilen Duygu Frekansları', fontsize=16, fontweight='bold')
            plt.xlabel('Duygular', fontsize=12)
            plt.ylabel('Frekans', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig('emotion_frequency.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    def save_results(self, results):
        """Sonuçları dosyaya kaydet"""
        with open('custom_test_results.txt', 'w', encoding='utf-8') as f:
            f.write("🎭 Özel Veri Duygu Analizi Sonuçları\n")
            f.write("=" * 50 + "\n\n")
            
            for i, result in enumerate(results, 1):
                f.write(f"{i}. Metin: {result['text']}\n")
                f.write("Tespit edilen duygular:\n")
                
                if result['emotions']:
                    for emotion_info in result['emotions']:
                        f.write(f"  - {emotion_info['emotion']}: {emotion_info['confidence']:.3f}\n")
                else:
                    f.write("  - Belirgin duygu tespit edilemedi\n")
                
                f.write("\n")
            
            f.write("=" * 50 + "\n")
            f.write("📊 Özet İstatistikler:\n")
            
            # Toplam duygu sayısı
            total_emotions = sum(len(result['emotions']) for result in results)
            f.write(f"Toplam tespit edilen duygu sayısı: {total_emotions}\n")
            
            # En sık tespit edilen duygu
            emotion_counts = {}
            for result in results:
                for emotion_info in result['emotions']:
                    emotion = emotion_info['emotion']
                    emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            
            if emotion_counts:
                most_common = max(emotion_counts, key=emotion_counts.get)
                f.write(f"En sık tespit edilen duygu: {most_common} ({emotion_counts[most_common]} kez)\n")
        
        print("✅ Sonuçlar 'custom_test_results.txt' dosyasına kaydedildi!")
    
    def test_custom_data(self):
        """Özel test verileriyle modeli test et"""
        print("🎯 Özel Veri Testi Başlatılıyor...")
        
        # Özel test metinleri
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
        
        # Model yükle
        if not self.load_model():
            return
        
        # Tahmin yap
        print("🔍 Duygu analizi yapılıyor...")
        predictions = self.predict_emotions(custom_texts)
        
        if predictions is None:
            return
        
        # Sonuçları analiz et
        results = self.analyze_predictions(custom_texts, predictions)
        
        # Sonuçları göster
        print("\n📋 Analiz Sonuçları:")
        print("=" * 50)
        
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Metin: {result['text']}")
            print("Tespit edilen duygular:")
            
            if result['emotions']:
                for emotion_info in result['emotions']:
                    print(f"  - {emotion_info['emotion']}: {emotion_info['confidence']:.3f}")
            else:
                print("  - Belirgin duygu tespit edilemedi")
        
        # Görselleştirmeler oluştur
        print("\n📊 Görselleştirmeler oluşturuluyor...")
        self.create_heatmap(custom_texts, predictions)
        self.create_emotion_frequency_chart(results)
        
        # Sonuçları kaydet
        self.save_results(results)
        
        print("\n✅ Özel veri testi tamamlandı!")
        print("📁 Oluşturulan dosyalar:")
        print("  - custom_data_heatmap.png")
        print("  - emotion_frequency.png")
        print("  - custom_test_results.txt")

def main():
    """Ana fonksiyon"""
    tester = CustomDataTester()
    tester.test_custom_data()

if __name__ == "__main__":
    main()
