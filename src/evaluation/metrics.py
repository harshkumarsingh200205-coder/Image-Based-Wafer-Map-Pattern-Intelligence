"""Comprehensive multi-class evaluation metrics and confusion matrices."""

from typing import Dict, List, Optional, Union
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


def compute_classification_metrics(
    y_true: Union[np.ndarray, List],
    y_pred: Union[np.ndarray, List],
    labels: Optional[List[str]] = None,
) -> Dict[str, Union[float, np.ndarray, str]]:
    """Calculates multi-class accuracy, precision, recall, macro-F1, and confusion matrix.

    Args:
        y_true: Ground truth target labels.
        y_pred: Predicted class labels.
        labels: Ordered list of class label names.

    Returns:
        Dictionary of computed performance metrics.
    """
    acc = accuracy_score(y_true, y_pred)
    prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(y_true, y_pred, labels=labels, zero_division=0)

    return {
        "accuracy": float(acc),
        "macro_precision": float(prec_macro),
        "macro_recall": float(rec_macro),
        "macro_f1": float(f1_macro),
        "weighted_f1": float(f1_weighted),
        "confusion_matrix": cm,
        "classification_report": report,
    }
