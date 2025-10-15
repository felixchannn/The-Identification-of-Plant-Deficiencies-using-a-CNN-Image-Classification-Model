# precision.py
"""
Per-class precision utilities.

Use:
    from precision import per_class_precision_from_predictions
    prec = per_class_precision_from_predictions(y_true, y_pred, labels=[0,1,2])

If you don’t pass labels, they’re auto-discovered (sorted). Passing labels is
recommended to lock the order to your class_names list.
"""

from __future__ import annotations
from typing import Iterable, Any, Optional, List, Tuple
import numpy as np

# Optional sklearn for building the confusion matrix
try:
    from sklearn.metrics import confusion_matrix as sk_confusion_matrix
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b != 0 else 0.0

def _cm_numpy(y_true: Iterable[Any], y_pred: Iterable[Any], labels: Iterable[Any]) -> np.ndarray:
    y_true = np.asarray(list(y_true))
    y_pred = np.asarray(list(y_pred))
    labels = list(labels)
    idx = {lab: i for i, lab in enumerate(labels)}
    K = len(labels)
    cm = np.zeros((K, K), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if t in idx and p in idx:
            cm[idx[t], idx[p]] += 1
    return cm

def build_confusion_matrix(y_true: Iterable[Any],
                           y_pred: Iterable[Any],
                           labels: Optional[Iterable[Any]] = None) -> Tuple[np.ndarray, List[Any]]:
    if labels is None:
        labels_used = sorted(set(list(y_true)) | set(list(y_pred)))
    else:
        labels_used = list(labels)
    if SKLEARN_AVAILABLE:
        return sk_confusion_matrix(y_true, y_pred, labels=labels_used), labels_used
    return _cm_numpy(y_true, y_pred, labels_used), labels_used

def per_class_precision_from_cm(cm: np.ndarray) -> np.ndarray:
    """
    Precision per column i = TP_i / (TP_i + FP_i).
    In a CM with rows=true, cols=pred, TP_i = cm[i,i], FP_i = sum(col i) - TP_i.
    """
    tp = np.diag(cm).astype(float)
    predcnt = cm.sum(axis=0).astype(float)  # total predicted as class i
    return np.array([_safe_div(tp[i], predcnt[i]) for i in range(len(tp))], dtype=float)

def per_class_precision_from_predictions(y_true: Iterable[Any],
                                         y_pred: Iterable[Any],
                                         labels: Optional[Iterable[Any]] = None) -> np.ndarray:
    """
    Compute per-class precision vector aligned with 'labels' order.
    If labels is None, labels are auto-discovered (sorted) and precision is returned in that order.
    """
    cm, labels_used = build_confusion_matrix(y_true, y_pred, labels=labels)
    return per_class_precision_from_cm(cm)

# Tiny demo
if __name__ == "__main__":
    y_true = [0,0,1,1,2,2,2]
    y_pred = [0,1,1,0,2,0,2]
    prec = per_class_precision_from_predictions(y_true, y_pred, labels=[0,1,2])
    print("Per-class precision:", np.round(prec, 4))
