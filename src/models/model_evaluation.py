from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any

def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """
    Evaluate model performance
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary containing evaluation metrics
    """
    report = classification_report(y_true, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    return {
        'classification_report': report,
        'confusion_matrix': conf_matrix
    }

def plot_confusion_matrix(conf_matrix: np.ndarray, labels: list = None) -> None:
    """
    Plot confusion matrix
    
    Args:
        conf_matrix: Confusion matrix
        labels: Class labels
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show() 