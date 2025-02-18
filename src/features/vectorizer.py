from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from typing import List, Union
import numpy as np

class TextVectorizer:
    """Text vectorization utilities for converting text to numerical features."""
    
    def __init__(self, vectorizer_type: str = 'tfidf', **kwargs):
        """
        Initialize text vectorizer
        
        Args:
            vectorizer_type: 'tfidf' or 'count'
            **kwargs: Additional arguments for the vectorizer
        """
        self.vectorizer_type = vectorizer_type.lower()
        if self.vectorizer_type == 'tfidf':
            self.vectorizer = TfidfVectorizer(**kwargs)
        elif self.vectorizer_type == 'count':
            self.vectorizer = CountVectorizer(**kwargs)
        else:
            raise ValueError("vectorizer_type must be 'tfidf' or 'count'")
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """
        Fit the vectorizer and transform the texts
        
        Args:
            texts: List of text documents
            
        Returns:
            Document-term matrix
        """
        return self.vectorizer.fit_transform(texts)
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """
        Transform new texts using the fitted vectorizer
        
        Args:
            texts: List of text documents
            
        Returns:
            Document-term matrix
        """
        return self.vectorizer.transform(texts)
    
    def get_feature_names(self) -> List[str]:
        """Get the feature names (words) used in the vectorizer."""
        if self.vectorizer is None:
            raise ValueError("Vectorizer must be fitted first using fit_transform")
        return self.vectorizer.get_feature_names_out()
