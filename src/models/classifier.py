from sklearn.base import BaseEstimator
from typing import Any, Dict, Union
import numpy as np
import pickle

class TextClassifier:
    def __init__(self, model: BaseEstimator):
        """
        Initialize text classifier
        
        Args:
            model: sklearn classifier instance
        """
        self.model = model
    
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the classifier
        
        Args:
            X: Document-term matrix
            y: Labels
        """
        self.model.fit(X, y)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict labels for new documents
        
        Args:
            X: Document-term matrix
            
        Returns:
            Predicted labels
        """
        return self.model.predict(X)
    
    def save_model(self, filepath: str) -> None:
        """Save model to disk"""
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
    
    @classmethod
    def load_model(cls, filepath: str) -> 'TextClassifier':
        """Load model from disk"""
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        return cls(model)
