import pandas as pd
import numpy as np
from tensorflow.keras.datasets import imdb

def download_movie_reviews(data_dir: str = 'data/raw') -> pd.DataFrame:
    """Download and load IMDB Movie Reviews dataset"""
    print("Loading IMDB Movie Reviews dataset...")
    
    # Load data from keras
    (X_train, y_train), (X_test, y_test) = imdb.load_data()
    word_index = imdb.get_word_index()
    
    # Create reverse mapping to decode reviews
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    
    def decode_review(text):
        return ' '.join([reverse_word_index.get(i - 3, '?') for i in text])
    
    # Combine train and test data
    texts = np.concatenate([X_train, X_test])
    labels = np.concatenate([y_train, y_test])
    
    # Decode reviews to readable text
    decoded_texts = [decode_review(text) for text in texts]
    
    return pd.DataFrame({
        'text': decoded_texts,
        'label': labels
    })

def download_movie_reviews_with_metadata(data_dir: str = 'data/raw') -> pd.DataFrame:
    """Download and load IMDB Movie Reviews dataset with basic metadata"""
    print("Loading IMDB Movie Reviews dataset with metadata...")
    
    # Load base dataset
    df = download_movie_reviews()
    
    # Add basic metadata
    df['length'] = df['text'].str.len()
    df['word_count'] = df['text'].str.split().str.len()
    df['avg_word_length'] = df['text'].apply(lambda x: sum(len(word) for word in x.split()) / len(x.split()))
    
    # Categorize reviews by length
    df['length_category'] = pd.qcut(df['length'], q=5, labels=['Very Short', 'Short', 'Medium', 'Long', 'Very Long'])
    
    return df 