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