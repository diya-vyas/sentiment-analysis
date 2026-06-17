"""
evaluate.py
Evaluation metrics and visualizations for sentiment analysis models.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
    f1_score,
)


def print_report(y_true, y_pred, labels=None):
    """Print a full classification report."""
    print(classification_report(y_true, y_pred, target_names=labels))


def plot_confusion_matrix(y_true, y_pred, labels, save_path=None):
    """Plot and optionally save a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 5))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=labels, yticklabels=labels
    )
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved confusion matrix to {save_path}")
    plt.show()


def plot_training_history(history, save_path=None):
    """Plot training vs validation accuracy and loss from a Keras history object."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history['accuracy'], label='Train Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[0].set_title('Accuracy over Epochs')
    axes[0].legend()

    axes[1].plot(history.history['loss'], label='Train Loss')
    axes[1].plot(history.history['val_loss'], label='Val Loss')
    axes[1].set_title('Loss over Epochs')
    axes[1].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def save_metrics(y_true, y_pred, model_name='model', save_path='results/metrics.json'):
    """Save key metrics to a JSON file."""
    metrics = {
        'model': model_name,
        'accuracy': round(accuracy_score(y_true, y_pred), 4),
        'f1_weighted': round(f1_score(y_true, y_pred, average='weighted'), 4),
    }
    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {save_path}: {metrics}")
    return metrics
