# Enhanced Sentiment Analysis with Data Augmentation

Bu proje, Go Emotions veri seti kullanarak 27 farklı duygu türünü sınıflandıran gelişmiş bir duygu analizi modeli içerir. Model, veri artırma (Data Augmentation) teknikleri kullanılarak geliştirilmiştir.

## 🚀 Özellikler

- **27 Duygu Sınıfı**: admiration, amusement, anger, annoyance, approval, caring, confusion, curiosity, desire, disappointment, disapproval, disgust, embarrassment, excitement, fear, gratitude, grief, joy, love, nervousness, optimism, pride, realization, relief, remorse, sadness, surprise, neutral
- **Gelişmiş Veri Artırma**: Synonym replacement, random insertion, random swap, random deletion
- **Bidirectional GRU Model**: Çok katmanlı derin öğrenme modeli
- **Kapsamlı Değerlendirme**: Accuracy, AUC, Precision, Recall metrikleri
- **Görselleştirme**: Training history ve sonuç analizi grafikleri
- **Özel Veri Testi**: Kendi verilerinizle model testi

## 📁 Dosya Yapısı

```
sentiment-analysis/
├── main_enhanced.py          # Gelişmiş model eğitimi
├── text_processor.py         # Metin işleme ve veri artırma
├── test_custom_data.py       # Özel veri testi
├── main.py                   # Orijinal model
├── pred.py                   # Orijinal tahmin
├── requirements.txt          # Gerekli kütüphaneler
├── README.md                 # Bu dosya
├── sentiment_analysis_model.h5  # Eğitilmiş model
├── tokenizer.pkl            # Tokenizer
├── mlb.pkl                  # Label encoder
└── best_model.h5            # En iyi model (eğitim sırasında)
```

## 🛠️ Kurulum

1. **Gerekli kütüphaneleri yükleyin:**
```bash
pip install -r requirements.txt
```

2. **NLTK verilerini indirin:**
```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
```

## 🎯 Kullanım

### 1. Model Eğitimi

Gelişmiş veri artırma teknikleriyle modeli eğitin:

```bash
python main_enhanced.py
```

Bu script:
- 10,000 örnek yükler
- Veri artırma teknikleri uygular
- Gelişmiş model mimarisi kullanır
- Training history grafiklerini oluşturur
- 10 özel test verisiyle modeli test eder

### 2. Özel Veri Testi

Kendi verilerinizle modeli test edin:

```bash
python test_custom_data.py
```

Bu script:
- Eğitilmiş modeli yükler
- 10 özel test verisiyle tahmin yapar
- Detaylı sonuç analizi sunar
- Görselleştirmeler oluşturur
- Sonuçları dosyaya kaydeder

### 3. Kendi Verilerinizi Test Etmek

`test_custom_data.py` dosyasındaki `custom_texts` listesini değiştirin:

```python
custom_texts = [
    "Your custom text here",
    "Another custom text",
    # ... daha fazla metin
]
```

## 📊 Veri Artırma Teknikleri

### 1. Synonym Replacement (Eş Anlamlı Kelime Değiştirme)
Kelimeleri WordNet kullanarak eş anlamlılarıyla değiştirir
- Örnek: "happy" → "joyful"

### 2. Random Insertion (Rastgele Ekleme)
Metindeki kelimeleri rastgele konumlara ekler
- Örnek: "I am happy" → "I am happy am"

### 3. Random Swap (Rastgele Değiştirme)
Kelime çiftlerini rastgele değiştirir
- Örnek: "I am happy" → "am I happy"

### 4. Random Deletion (Rastgele Silme)
Kelimeleri belirli olasılıkla siler
- Örnek: "I am very happy" → "I am happy"

### 5. Random Shuffle (Rastgele Karıştırma)
Kelimeleri rastgele sıralar
- Örnek: "I am happy" → "happy I am"

## 🏗️ Model Mimarisi

```
Embedding (256d) → SpatialDropout (0.3)
    ↓
Bidirectional GRU (256) → LayerNormalization
    ↓
Bidirectional GRU (128) → LayerNormalization
    ↓
Bidirectional GRU (64) → LayerNormalization
    ↓
Dense (128) → Dropout (0.5)
    ↓
Dense (64) → Dropout (0.3)
    ↓
Dense (27, sigmoid)  # 27 duygu sınıfı
```

## 📈 Metrikler

Model şu metriklerle değerlendirilir:

- **Accuracy**: Genel doğruluk oranı
- **AUC**: ROC eğrisi altındaki alan
- **Precision**: Kesinlik
- **Recall**: Duyarlılık

## 📊 Çıktılar

Eğitim sonrası oluşturulan dosyalar:

- `training_history.png`: Eğitim geçmişi grafikleri
- `custom_data_heatmap.png`: Özel veri sonuçları heatmap
- `emotion_frequency.png`: Duygu frekans grafiği
- `custom_test_results.txt`: Detaylı test sonuçları

## 🔧 Özelleştirme

### Veri Artırma Parametreleri
`text_processor.py` dosyasında:
- `num_augmentations`: Her metin için oluşturulacak artırılmış örnek sayısı
- `threshold`: Duygu tespiti için güven eşiği

### Model Parametreleri
`main_enhanced.py` dosyasında:
- `epochs`: Eğitim epoch sayısı
- `batch_size`: Batch boyutu
- `learning_rate`: Öğrenme oranı

## 🚨 Sorun Giderme

### Model Yükleme Hatası
```
✗ Error loading model. Please run main_enhanced.py first.
```
**Çözüm**: Önce `main_enhanced.py` scriptini çalıştırın.

### NLTK Veri Hatası
```
LookupError: Resource stopwords not found
```
**Çözüm**: NLTK verilerini indirin:
```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
```

### Bellek Hatası
**Çözüm**: Batch size'ı küçültün veya veri seti boyutunu azaltın.

## 📝 Örnek Kullanım

```python
# Basit tahmin örneği
import tensorflow as tf
import pickle

# Model yükle
model = tf.keras.models.load_model("sentiment_analysis_model.h5")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
with open("mlb.pkl", "rb") as f:
    mlb = pickle.load(f)

# Metin işle
from text_processor import process_text
text = "I am feeling very happy today!"
clean_text = process_text(text)

# Tahmin yap
sequences = tokenizer.texts_to_sequences([clean_text])
X = tf.keras.preprocessing.sequence.pad_sequences(sequences, padding="post", maxlen=100)
predictions = model.predict(X)

# Sonuçları göster
top_indices = np.argsort(predictions[0])[-3:][::-1]
for idx in top_indices:
    emotion = mlb.classes_[idx]
    confidence = predictions[0][idx]
    print(f"{emotion}: {confidence:.3f}")
```

## 🤝 Katkıda Bulunma

1. Bu repository'yi fork edin
2. Feature branch oluşturun (`git checkout -b feature/AmazingFeature`)
3. Değişikliklerinizi commit edin (`git commit -m 'Add some AmazingFeature'`)
4. Branch'inizi push edin (`git push origin feature/AmazingFeature`)
5. Pull Request oluşturun

## 📄 Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için `LICENSE` dosyasına bakın.

## 📞 İletişim

Proje Linki: [https://github.com/yourusername/sentiment-analysis](https://github.com/yourusername/sentiment-analysis)

---

⭐ Bu projeyi beğendiyseniz yıldız vermeyi unutmayın!
