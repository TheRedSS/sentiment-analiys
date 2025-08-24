import requests
import pandas as pd
import os

def download_go_emotions_dataset():
    """Go Emotions veri setini indir"""
    print("Go Emotions veri seti indiriliyor...")
    
    # Go Emotions veri seti URL'leri
    urls = {
        'train': 'https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data/train.tsv',
        'dev': 'https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data/dev.tsv',
        'test': 'https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data/test.tsv'
    }
    
    # Duygu etiketleri
    emotions = [
        'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
        'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
        'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
        'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
        'relief', 'remorse', 'sadness', 'surprise', 'neutral'
    ]
    
    all_data = []
    
    for split_name, url in urls.items():
        try:
            print(f"{split_name} veri seti indiriliyor...")
            response = requests.get(url)
            response.raise_for_status()
            
            # TSV dosyasını oku
            lines = response.text.strip().split('\n')
            
            for line in lines:
                parts = line.split('\t')
                if len(parts) >= 3:
                    text = parts[0]
                    emotion_labels = parts[1].split(',')
                    
                    # Duygu vektörü oluştur
                    emotion_vector = [0] * len(emotions)
                    for label in emotion_labels:
                        if label in emotions:
                            idx = emotions.index(label)
                            emotion_vector[idx] = 1
                    
                    # Veri satırı oluştur
                    row = {'text': text}
                    for i, emotion in enumerate(emotions):
                        row[emotion] = emotion_vector[i]
                    
                    all_data.append(row)
            
            print(f"{split_name} veri seti başarıyla indirildi!")
            
        except Exception as e:
            print(f"Hata: {split_name} veri seti indirilemedi: {e}")
    
    if all_data:
        # DataFrame oluştur
        df = pd.DataFrame(all_data)
        
        # CSV olarak kaydet
        df.to_csv('go_emotions_dataset.csv', index=False)
        print(f"Toplam {len(df)} örnek kaydedildi!")
        print("Veri seti 'go_emotions_dataset.csv' olarak kaydedildi.")
        
        # Veri seti istatistikleri
        print("\nVeri Seti İstatistikleri:")
        print(f"Toplam örnek sayısı: {len(df)}")
        print(f"Toplam duygu sayısı: {len(emotions)}")
        
        # Her duygu için örnek sayısı
        print("\nDuygu Dağılımı:")
        for emotion in emotions:
            count = df[emotion].sum()
            print(f"{emotion}: {count}")
        
        return True
    else:
        print("Veri seti indirilemedi!")
        return False

def create_sample_dataset():
    """Örnek veri seti oluştur"""
    print("Örnek veri seti oluşturuluyor...")
    
    emotions = [
        'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
        'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
        'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
        'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
        'relief', 'remorse', 'sadness', 'surprise', 'neutral'
    ]
    
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
    
    import random
    
    all_data = []
    
    # Her duygu için örnekler oluştur
    for emotion in emotions:
        for _ in range(100):  # Her duygu için 100 örnek
            text = random.choice(sample_texts)
            
            # Duygu vektörü oluştur
            emotion_vector = [0] * len(emotions)
            emotion_idx = emotions.index(emotion)
            emotion_vector[emotion_idx] = 1
            
            # Rastgele ek duygular ekle
            if random.random() < 0.3:  # %30 olasılıkla ek duygu
                extra_emotion = random.choice(emotions)
                if extra_emotion != emotion:
                    extra_idx = emotions.index(extra_emotion)
                    emotion_vector[extra_idx] = 1
            
            # Veri satırı oluştur
            row = {'text': text}
            for i, emo in enumerate(emotions):
                row[emo] = emotion_vector[i]
            
            all_data.append(row)
    
    # DataFrame oluştur
    df = pd.DataFrame(all_data)
    
    # CSV olarak kaydet
    df.to_csv('go_emotions_dataset.csv', index=False)
    print(f"Örnek veri seti oluşturuldu: {len(df)} örnek")
    print("Veri seti 'go_emotions_dataset.csv' olarak kaydedildi.")
    
    return True

if __name__ == "__main__":
    print("Go Emotions Veri Seti İndirici")
    print("=" * 40)
    
    # Önce gerçek veri setini indirmeyi dene
    if not download_go_emotions_dataset():
        print("\nGerçek veri seti indirilemedi. Örnek veri seti oluşturuluyor...")
        create_sample_dataset()
    
    print("\n✅ Veri seti hazır!")
