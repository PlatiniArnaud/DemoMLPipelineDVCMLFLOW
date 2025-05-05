import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix

## Function to create confusion matrix
def make_confusion_matrix(model, X_test, y_true, labels=[0, 1], color='Blues', threshold=None):
    """
    Generate and return a confusion matrix heatmap as a Matplotlib figure.
    
    Parameters:
        model     : Trained classifier (must support .predict or .predict_proba)
        X_test    : Feature matrix
        y_true    : Ground truth labels
        labels    : List of class labels (default = [0, 1])
        color     : Seaborn colormap for the heatmap
        threshold : Custom threshold (for binary classification only)
    
    Returns:
        fig       : Matplotlib Figure object
    """
    # Predict using threshold if provided
    if threshold is not None and hasattr(model, "predict_proba"):
        y_scores = model.predict_proba(X_test)[:, 1]
        y_pred = (y_scores > threshold).astype(int)
    else:
        y_pred = model.predict(X_test)

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    df_cm = pd.DataFrame(cm,
                         index=[f"Actual - {l}" for l in labels],
                         columns=[f"Predicted - {l}" for l in labels])

    # Create labels with counts and percentages
    total = np.sum(cm)
    labels_text = [f"{v}\n{v/total:.2%}" for v in cm.flatten()]
    labels_text = np.asarray(labels_text).reshape(len(labels), len(labels))

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(df_cm, annot=labels_text, fmt='', cmap=color, cbar=False, ax=ax)

    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    ax.set_title('Confusion Matrix')
    
    return fig

def plot_roc_curve(model, X_test, y_test):
    """
    Plots the ROC curve for a given model and test set.

    Returns:
        fig: Matplotlib figure object
    """
    y_score = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_score)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label="ROC Curve")
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    ax.grid(True)
    
    return fig


def plot_precision_recall_curve(model, X_test, y_test):
    """
    Plots the Precision-Recall curve for a given model and test set.

    Returns:
        fig: Matplotlib figure object
    """
    y_score = model.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_score)

    fig, ax = plt.subplots()
    ax.plot(recall, precision, color='orange', label="Precision-Recall Curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="lower left")
    ax.grid(True)
    
    return fig
