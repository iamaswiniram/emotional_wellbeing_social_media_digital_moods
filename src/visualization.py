import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np

def plot_confusion_matrix(y_true, y_pred, labels=None, title='Confusion Matrix'):
    """
    The 'Mirror of Truth'. This heatmap reveals where the model is confident 
    and where it gets confused between similar emotions.
    """
    # Permanent Fix: Decouple calculation from labels to avoid str-int comparison errors.
    # We calculate the matrix using unique values present in the data.
    unique_labels = np.unique(np.concatenate([y_true, y_pred]))
    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
    
    plt.figure(figsize=(10, 7))
    # We use the provided human-readable labels only for the axis ticks.
    # If the number of labels doesn't match the unique data values, we fall back to defaults.
    tick_labels = labels if labels is not None and len(labels) == len(unique_labels) else unique_labels
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=tick_labels, yticklabels=tick_labels)
    plt.xlabel('What the Model Predicted')
    plt.ylabel('The Actual Reality')
    plt.title(title)
    plt.show()

def plot_feature_importance(model, feature_names, title='Feature Importance'):
    """
    The 'Main Characters'. This chart shows which behavioral signals 
    (like usage time) were most influential in the model's decision-making.
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(12, 6))
        plt.title(title)
        # We sort them to see the 'Top Influencers' first.
        plt.bar(range(len(importances)), importances[indices], align='center')
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.show()

def plot_usage_vs_emotion(df, usage_col, emotion_col, title='Usage vs Emotion'):
    """
    The 'Behavioral Spread'. We use Boxplots to see the variation in digital 
    habits across different emotional groups.
    """
    plt.figure(figsize=(12, 6))
    # This helps us identify if 'Long Usage' consistently correlates with 'Higher Stress'.
    sns.boxplot(x=emotion_col, y=usage_col, data=df, palette="Set3")
    plt.title(title)
    plt.xticks(rotation=45)
    plt.show()
    
def plot_correlation_heatmap(df, title='Correlation Heatmap'):
    """
    The 'Web of Relationships'. A birds-eye view of how all behavioral 
    metrics (Posts, Likes, Time) interact with each other.
    """
    plt.figure(figsize=(12, 10))
    numeric_df = df.select_dtypes(include=[np.number])
    # Scores near +1 indicate strong positive synergy, near -1 indicate inverse trade-offs.
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title(title)
    plt.show()

def plot_multiclass_roc(model, X_test, y_test, n_classes, title='ROC Curve'):
    """
    The 'Efficiency Curve'. This measures the trade-off between identifying 
    emotions correctly vs making false alarms.
    """
    # A curve that hugs the 'Top Left' corner indicates a 'Master' model.
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize
    
    y_test_bin = label_binarize(y_test, classes=range(n_classes))
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)
    else:
        return

    plt.figure(figsize=(10, 8))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        plt.plot(fpr, tpr, label=f'Class {i}') # Tracking performance for each emotion
        
    plt.plot([0, 1], [0, 1], 'k--') # The 'Random Guess' line
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()
