# 🧠 Çok Sınıflı Duygu Analizi Projesi

Bu proje, **Go Emotions** veri seti kullanarak 27 farklı duygu türünü sınıflandıran gelişmiş bir yapay zeka modeli sunar. Bidirectional GRU tabanlı derin öğrenme mimarisi ve veri artırma teknikleri kullanılarak geliştirilmiştir.

## 🌟 Proje Özellikleri

### 🎯 Ana Özellikler
- **27 Duygu Sınıfı**: admiration, amusement, anger, annoyance, approval, caring, confusion, curiosity, desire, disappointment, disapproval, disgust, embarrassment, excitement, fear, gratitude, grief, joy, love, nervousness, optimism, pride, realization, relief, remorse, sadness, surprise, neutral
- **Gelişmiş Veri Artırma**: Synonym replacement, random insertion, random swap, random deletion teknikleri
- **Bidirectional GRU Modeli**: Çok katmanlı derin öğrenme mimarisi
- **Kapsamlı Değerlendirme**: Accuracy, AUC, Precision, Recall metrikleri
- **Görselleştirme**: Eğitim geçmişi ve sonuç analizi grafikleri
- **Özel Veri Testi**: Kendi metinlerinizle model testi

### 🔧 Teknik Özellikler
- **GPU Desteği**: TensorFlow GPU optimizasyonu
- **Otomatik NLTK Kurulumu**: Gerekli dil işleme araçları
- **Model Kaydetme**: En iyi modelin otomatik kaydedilmesi
- **Early Stopping**: Aşırı öğrenmeyi önleme
- **Learning Rate Scheduling**: Dinamik öğrenme oranı ayarlama

## 📁 Proje Yapısı

```
sentiment-analiys/
├── 📄 main_enhanced.py          # Gelişmiş model eğitimi ve ana script
├── 📄 text_processor.py         # Metin işleme ve veri artırma sınıfı
├── 📄 test_custom_data.py       # Özel veri testi scripti
├── 📄 main.py                   # Orijinal basit model
├── 📄 pred.py                   # Orijinal tahmin scripti
├── 📄 download_dataset.py       # Veri seti indirme scripti
├── 📄 requirements.txt          # Python bağımlılıkları
├── 📄 README.md                 # Proje dokümantasyonu
├── 📊 go_emotions_dataset.csv   # Go Emotions veri seti
├── 🎯 best_model.h5            # En iyi eğitilmiş model
├── 🔧 tokenizer.pkl            # Metin tokenizer'ı
└── 🏷️ mlb.pkl                  # Multi-label binarizer
```

## 🚀 Hızlı Başlangıç

### 1. Ortam Kurulumu

```bash
# Gerekli kütüphaneleri yükleyin
pip install -r requirements.txt

# NLTK verilerini indirin (otomatik olarak yapılır)
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt'); nltk.download('averaged_perceptron_tagger')"
```

### 2. Model Eğitimi

```bash
# Gelişmiş modeli eğitin
python main_enhanced.py
```

Bu script şunları yapar:
- ✅ 10,000 örnek veri yükler
- ✅ Gelişmiş veri artırma teknikleri uygular
- ✅ Bidirectional GRU modelini eğitir
- ✅ Eğitim geçmişi grafiklerini oluşturur
- ✅ 10 özel test verisiyle modeli değerlendirir

### 3. Özel Veri Testi

```bash
# Kendi verilerinizle test edin
python test_custom_data.py
```

## 🧮 Veri Artırma Teknikleri

### 1. Synonym Replacement (Eş Anlamlı Değiştirme)
```python
# Örnek: "happy" → "joyful"
text = "I am happy today"
augmented = "I am joyful today"
```

### 2. Random Insertion (Rastgele Ekleme)
```python
# Örnek: "I am happy" → "I am happy am"
text = "I am happy"
augmented = "I am happy am"
```

### 3. Random Swap (Rastgele Değiştirme)
```python
# Örnek: "I am happy" → "am I happy"
text = "I am happy"
augmented = "am I happy"
```

### 4. Random Deletion (Rastgele Silme)
```python
# Örnek: "I am very happy" → "I am happy"
text = "I am very happy"
augmented = "I am happy"
```

## 🏗️ Model Mimarisi

```
📥 Input Text
    ↓
🔤 Embedding Layer (256d)
    ↓
🚫 SpatialDropout (0.3)
    ↓
🔄 Bidirectional GRU (256 units)
    ↓
📏 LayerNormalization
    ↓
🔄 Bidirectional GRU (128 units)
    ↓
📏 LayerNormalization
    ↓
🔄 Bidirectional GRU (64 units)
    ↓
📏 LayerNormalization
    ↓
🧠 Dense Layer (128 units)
    ↓
🚫 Dropout (0.5)
    ↓
🧠 Dense Layer (64 units)
    ↓
🚫 Dropout (0.3)
    ↓
🎯 Output Layer (27 units, sigmoid)
```

## 📊 Performans Metrikleri

Model şu metriklerle değerlendirilir:

| Metrik | Açıklama |
|--------|----------|
| **Accuracy** | Genel doğruluk oranı |
| **AUC** | ROC eğrisi altındaki alan |
| **Precision** | Kesinlik (yanlış pozitif oranı) |
| **Recall** | Duyarlılık (yanlış negatif oranı) |

## 🎨 Çıktı Dosyaları

Eğitim sonrası oluşturulan görselleştirmeler:

- 📈 `training_history.png`: Eğitim ve doğrulama metrikleri
- 🗺️ `custom_data_heatmap.png`: Özel veri sonuçları heatmap
- 📊 `emotion_frequency.png`: Duygu dağılım grafiği
- 📝 `custom_test_results.txt`: Detaylı test sonuçları

## 💻 Kullanım Örnekleri

### Basit Tahmin

```python
import tensorflow as tf
import pickle
import numpy as np

# Model ve tokenizer yükle
model = tf.keras.models.load_model("best_model.h5")
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
X = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=100)
predictions = model.predict(X)

# En yüksek 3 duyguyu göster
top_indices = np.argsort(predictions[0])[-3:][::-1]
for idx in top_indices:
    emotion = mlb.classes_[idx]
    confidence = predictions[0][idx]
    print(f"🎭 {emotion}: {confidence:.3f}")
```

### Özel Veri Testi

```python
# test_custom_data.py dosyasını düzenleyin
custom_texts = [
    "Bu film gerçekten harika!",
    "Çok üzgünüm bu durum için.",
    "Yeni projeden heyecan duyuyorum!",
    "Bu davranış beni kızdırıyor.",
    "Yardımın için minnettarım."
]
```

## ⚙️ Özelleştirme

### Veri Artırma Parametreleri

`text_processor.py` dosyasında:

```python
# Her metin için oluşturulacak artırılmış örnek sayısı
num_augmentations = 3

# Duygu tespiti için güven eşiği
threshold = 0.3
```

### Model Parametreleri

`main_enhanced.py` dosyasında:

```python
# Eğitim parametreleri
epochs = 50
batch_size = 32
learning_rate = 0.001
max_words = 10000
max_len = 100
embedding_dim = 256
```

## 🚨 Sorun Giderme

### ❌ Model Yükleme Hatası
```
Error loading model. Please run main_enhanced.py first.
```
**Çözüm**: Önce `python main_enhanced.py` komutunu çalıştırın.

### ❌ NLTK Veri Hatası
```
LookupError: Resource stopwords not found
```
**Çözüm**: NLTK verilerini manuel olarak indirin:
```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
```

### ❌ Bellek Hatası
**Çözüm**: 
- Batch size'ı küçültün: `batch_size = 16`
- Veri seti boyutunu azaltın: `sample_size = 5000`

### ❌ GPU Hatası
**Çözüm**: CPU kullanımına geçin:
```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
```

## 🔬 Teknik Detaylar

### Veri İşleme Pipeline
1. **Metin Temizleme**: Küçük harfe çevirme, özel karakter kaldırma
2. **Tokenization**: Kelime bazlı tokenization
3. **Stop Word Removal**: Gereksiz kelimeleri kaldırma
4. **Lemmatization**: Kelime köklerini bulma
5. **Veri Artırma**: 4 farklı teknik uygulama

### Model Optimizasyonu
- **Early Stopping**: Aşırı öğrenmeyi önleme
- **Model Checkpoint**: En iyi modeli kaydetme
- **ReduceLROnPlateau**: Öğrenme oranını dinamik ayarlama
- **Layer Normalization**: Eğitim stabilizasyonu

## 🤝 Katkıda Bulunma

1. 🍴 Bu repository'yi fork edin
2. 🌿 Feature branch oluşturun: `git checkout -b feature/YeniOzellik`
3. 💾 Değişikliklerinizi commit edin: `git commit -m 'Yeni özellik eklendi'`
4. 📤 Branch'inizi push edin: `git push origin feature/YeniOzellik`
5. 🔄 Pull Request oluşturun

## 📄 Lisans

Bu proje **MIT lisansı** altında lisanslanmıştır. Detaylar için `LICENSE` dosyasına bakın.

## 📞 İletişim

- 🌐 **GitHub**: [Proje Linki](https://github.com/yourusername/sentiment-analiys)
- 📧 **E-posta**: your.email@example.com
- 💬 **Issues**: GitHub Issues sayfasını kullanın


*Bu proje, doğal dil işleme ve duygu analizi alanında gelişmiş teknikler kullanarak 27 farklı duygu türünü sınıflandırabilen güçlü bir yapay zeka modeli sunar.*
