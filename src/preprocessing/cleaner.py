import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from typing import List, Union

class TextCleaner:
    def __init__(self):
        # Download required NLTK data
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.patterns = {
            'urls': r'https?://\S+|www\.\S+',
            'html_tags': r'<.*?>',
            'punctuation': r'[^\w\s]',
            'numbers': r'\d+'
        }

    def clean_text(self, text: Union[str, List[str]]) -> Union[str, List[str]]:
        """
        Clean text by removing URLs, HTML tags, punctuation, and numbers
        
        Args:
            text: Input text or list of texts to clean
            
        Returns:
            Cleaned text or list of cleaned texts
        """
        if isinstance(text, list):
            return [self.clean_text(t) for t in text]
        
        text = text.lower()
        for pattern in self.patterns.values():
            text = re.sub(pattern, ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text

    def tokenize(self, text: str) -> list:
        """Tokenize text into words."""
        return word_tokenize(text)

    def remove_stopwords(self, tokens: list) -> list:
        """Remove common stopwords."""
        return [token for token in tokens if token not in self.stop_words]

    def lemmatize(self, tokens: list) -> list:
        """Lemmatize tokens to their root form."""
        return [self.lemmatizer.lemmatize(token) for token in tokens]

    def process_text(self, text: str) -> list:
        """Complete text processing pipeline."""
        cleaned_text = self.clean_text(text)
        tokens = self.tokenize(cleaned_text)
        tokens = self.remove_stopwords(tokens)
        tokens = self.lemmatize(tokens)
        return tokens
