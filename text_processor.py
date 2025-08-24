import re
import random
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np

# NLTK verilerini indir
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

class TextProcessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
    def get_synonyms(self, word):
        """Kelimelerin eş anlamlılarını al"""
        synonyms = []
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                if lemma.name() != word and lemma.name() not in synonyms:
                    synonyms.append(lemma.name())
        return synonyms
    
    def process_text(self, text):
        """Metni temizle ve işle"""
        # Küçük harfe çevir
        text = text.lower()
        
        # Özel karakterleri kaldır
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize et
        tokens = word_tokenize(text)
        
        # Stop words'leri kaldır ve lemmatize et
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in self.stop_words and len(token) > 2]
        
        return ' '.join(tokens)
    
    def synonym_replacement(self, text, n=1):
        """Eş anlamlı kelime değiştirme"""
        words = text.split()
        n = min(n, len(words))
        
        if n == 0:
            return text
            
        new_words = words.copy()
        random_word_list = list(set([word for word in words if word not in self.stop_words]))
        random.shuffle(random_word_list)
        
        num_replaced = 0
        for random_word in random_word_list:
            synonyms = self.get_synonyms(random_word)
            if len(synonyms) >= 1:
                synonym = random.choice(synonyms)
                new_words = [synonym if word == random_word else word for word in new_words]
                num_replaced += 1
            if num_replaced >= n:
                break
                
        return ' '.join(new_words)
    
    def random_insertion(self, text, n=1):
        """Rastgele kelime ekleme"""
        words = text.split()
        n = min(n, len(words))
        
        if n == 0:
            return text
            
        new_words = words.copy()
        for _ in range(n):
            add_word = random.choice(words)
            random_idx = random.randint(0, len(new_words))
            new_words.insert(random_idx, add_word)
            
        return ' '.join(new_words)
    
    def random_swap(self, text, n=1):
        """Rastgele kelime değiştirme"""
        words = text.split()
        n = min(n, len(words) - 1)
        
        if n == 0:
            return text
            
        new_words = words.copy()
        for _ in range(n):
            idx1, idx2 = random.sample(range(len(new_words)), 2)
            new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]
            
        return ' '.join(new_words)
    
    def random_deletion(self, text, p=0.1):
        """Rastgele kelime silme"""
        words = text.split()
        
        if len(words) <= 1:
            return text
            
        remaining_words = []
        for word in words:
            if random.random() > p:
                remaining_words.append(word)
                
        if len(remaining_words) == 0:
            rand_int = random.randint(0, len(words) - 1)
            return words[rand_int]
            
        return ' '.join(remaining_words)
    
    def random_shuffle(self, text):
        """Kelimeleri rastgele karıştır"""
        words = text.split()
        random.shuffle(words)
        return ' '.join(words)
    
    def augment_text(self, text, num_augmentations=3):
        """Metni artır"""
        # Boş metin kontrolü
        if not text or len(text.strip()) == 0:
            return [text]
            
        augmented_texts = [text]
        
        for _ in range(num_augmentations):
            # Rastgele bir teknik seç
            technique = random.choice(['synonym', 'insertion', 'swap', 'deletion', 'shuffle'])
            
            if technique == 'synonym':
                augmented_text = self.synonym_replacement(text, n=1)
            elif technique == 'insertion':
                augmented_text = self.random_insertion(text, n=1)
            elif technique == 'swap':
                augmented_text = self.random_swap(text, n=1)
            elif technique == 'deletion':
                augmented_text = self.random_deletion(text, p=0.1)
            elif technique == 'shuffle':
                augmented_text = self.random_shuffle(text)
                
            augmented_texts.append(augmented_text)
            
        return augmented_texts

# Global text processor instance
text_processor = TextProcessor()

def process_text(text):
    """Global metin işleme fonksiyonu"""
    return text_processor.process_text(text)

def augment_text(text, num_augmentations=3):
    """Global metin artırma fonksiyonu"""
    return text_processor.augment_text(text, num_augmentations)
