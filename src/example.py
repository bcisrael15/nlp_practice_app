from preprocessing.cleaner import TextCleaner
from features.vectorizer import TextVectorizer
from models.classifier import TextClassifier
from models.model_evaluation import evaluate_model, plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from data.load_data import download_movie_reviews
import os

# Example usage
def main():
    # Load Movie Reviews dataset
    print("Loading dataset...")
    df = download_movie_reviews()
    texts = df['text'].tolist()
    labels = df['label'].tolist()
    
    print(f"Dataset loaded: {len(texts)} samples")
    print(f"Positive reviews: {sum(labels)}")
    print(f"Negative reviews: {len(labels) - sum(labels)}")
    
    # Clean texts
    print("\nCleaning texts...")
    cleaner = TextCleaner()
    cleaned_texts = cleaner.clean_text(texts)
    
    # Vectorize
    print("Vectorizing texts...")
    vectorizer = TextVectorizer(
        vectorizer_type='tfidf',
        max_features=10000,  # Increased features for more complex text
        min_df=2,  # Remove words that appear only once
        max_df=0.95  # Remove words that appear in >95% of documents
    )
    X = vectorizer.fit_transform(cleaned_texts)
    
    # Split data
    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Train classifier
    print("Training classifier...")
    classifier = TextClassifier(LogisticRegression(
        max_iter=1000,
        C=1.0,  # Regularization strength
        class_weight='balanced'  # Handle any class imbalance
    ))
    classifier.train(X_train, y_train)
    
    # Evaluate
    print("\nEvaluating model...")
    y_pred = classifier.predict(X_test)
    metrics = evaluate_model(y_test, y_pred)
    print("\nClassification Report:")
    print(metrics['classification_report'])
    
    # Plot confusion matrix
    plot_confusion_matrix(
        metrics['confusion_matrix'], 
        labels=['Negative', 'Positive']
    )
    
    # Save model
    print("\nSaving model...")
    os.makedirs('data/models', exist_ok=True)
    classifier.save_model('data/models/movie_review_classifier.pkl')
    
    # Print some example predictions
    print("\nExample predictions:")
    example_reviews = [
        "This movie was absolutely fantastic! Great acting and storyline.",
        "Terrible waste of time. Poor acting and boring plot.",
        "It was okay, not great but not terrible either."
    ]
    
    # Clean and vectorize example reviews
    cleaned_examples = cleaner.clean_text(example_reviews)
    X_examples = vectorizer.transform(cleaned_examples)
    predictions = classifier.predict(X_examples)
    
    for review, pred in zip(example_reviews, predictions):
        sentiment = "Positive" if pred == 1 else "Negative"
        print(f"\nReview: {review}")
        print(f"Predicted sentiment: {sentiment}")

if __name__ == "__main__":
    main() 