# f1_score.py
"""
F1 Score Utilities (Per-class, Macro, Micro, Weighted)

Use:
    from f1_score import f1_from_predictions, classification_report_df
    summary = f1_from_predictions(y_true, y_pred, labels=[0,1,2], label_names=["A","B","C"])
    df = classification_report_df(summary); print(df)

This module has no hard dependency on scikit-learn or pandas.
If they’re installed, they’ll be used; otherwise we fall back to NumPy.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Optional, Any, List, Tuple
import numpy as np

# Optional pandas (for pretty tables)
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except Exception:
    PANDAS_AVAILABLE = False
    pd = None  # type: ignore

# Optional sklearn (for confusion_matrix convenience)
try:
    from sklearn.metrics import confusion_matrix as sk_confusion_matrix
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


# --------------------- helpers ---------------------
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
    if labels is not None:
        labels_used = list(labels)
        if SKLEARN_AVAILABLE:
            return sk_confusion_matrix(y_true, y_pred, labels=labels_used), labels_used
        return _cm_numpy(y_true, y_pred, labels_used), labels_used
    # auto-discover label set (sorted)
    uniq = sorted(set(list(y_true)) | set(list(y_pred)))
    if SKLEARN_AVAILABLE:
        return sk_confusion_matrix(y_true, y_pred, labels=uniq), uniq
    return _cm_numpy(y_true, y_pred, uniq), uniq

def precision_recall_f1_per_class_from_cm(cm: np.ndarray):
    tp = np.diag(cm).astype(float)
    support = cm.sum(axis=1).astype(float)   # true counts per class
    predcnt = cm.sum(axis=0).astype(float)   # predicted counts per class

    precision = np.array([_safe_div(tp[i], predcnt[i]) for i in range(len(tp))], dtype=float)
    recall    = np.array([_safe_div(tp[i], support[i]) for i in range(len(tp))], dtype=float)
    f1 = np.array([
        _safe_div(2 * precision[i] * recall[i], precision[i] + recall[i]) if (precision[i] + recall[i]) > 0 else 0.0
        for i in range(len(tp))
    ], dtype=float)
    return precision, recall, f1, support

def f1_micro_from_cm(cm: np.ndarray) -> float:
    correct = float(np.trace(cm))
    total = float(cm.sum())
    return _safe_div(correct, total)  # = accuracy for single-label multiclass

def f1_macro_from_vec(f1_vec: np.ndarray, support: np.ndarray) -> float:
    mask = support > 0
    return float(f1_vec[mask].mean()) if np.any(mask) else 0.0

def f1_weighted_from_vec(f1_vec: np.ndarray, support: np.ndarray) -> float:
    total = float(support.sum())
    return float((f1_vec * support).sum() / total) if total > 0 else 0.0


# --------------------- public API ---------------------
@dataclass
class F1Summary:
    per_class_precision: np.ndarray
    per_class_recall: np.ndarray
    per_class_f1: np.ndarray
    support: np.ndarray
    labels: List[Any]
    label_names: List[str]
    macro_f1: float
    micro_f1: float
    weighted_f1: float

def f1_from_confusion_matrix(cm: np.ndarray, label_names: Optional[Iterable[str]] = None) -> F1Summary:
    K = cm.shape[0]
    labels = list(range(K))
    names = list(label_names) if label_names is not None else [str(i) for i in labels]
    prec, rec, f1, sup = precision_recall_f1_per_class_from_cm(cm)
    return F1Summary(
        per_class_precision=prec,
        per_class_recall=rec,
        per_class_f1=f1,
        support=sup,
        labels=labels,
        label_names=names,
        macro_f1=f1_macro_from_vec(f1, sup),
        micro_f1=f1_micro_from_cm(cm),
        weighted_f1=f1_weighted_from_vec(f1, sup),
    )

def f1_from_predictions(y_true: Iterable[Any],
                        y_pred: Iterable[Any],
                        labels: Optional[Iterable[Any]] = None,
                        label_names: Optional[Iterable[str]] = None) -> F1Summary:
    cm, used_labels = build_confusion_matrix(y_true, y_pred, labels=labels)
    if label_names is None:
        names = [str(x) for x in used_labels]
    else:
        names = list(label_names)
        if len(names) != len(used_labels):
            raise ValueError("label_names length must equal number of labels")
    return f1_from_confusion_matrix(cm, label_names=names)

def classification_report_df(summary: F1Summary):
    rows = []
    for i, name in enumerate(summary.label_names):
        rows.append({
            "class": name,
            "precision": summary.per_class_precision[i],
            "recall": summary.per_class_recall[i],
            "f1": summary.per_class_f1[i],
            "support": int(summary.support[i]),
        })
    rows.append({"class": "macro_avg",    "precision": np.nan, "recall": np.nan, "f1": summary.macro_f1,    "support": int(summary.support.sum())})
    rows.append({"class": "weighted_avg", "precision": np.nan, "recall": np.nan, "f1": summary.weighted_f1, "support": int(summary.support.sum())})
    rows.append({"class": "micro_avg",    "precision": np.nan, "recall": np.nan, "f1": summary.micro_f1,    "support": int(summary.support.sum())})
    if not PANDAS_AVAILABLE:
        return rows
    return pd.DataFrame(rows, columns=["class", "precision", "recall", "f1", "support"])

# Tiny demo
if __name__ == "__main__":
    y_true = [0,0,0,1,1,1,2,2,2]
    y_pred = [0,0,2,1,1,0,2,0,2]
    names = ["Nitrogen","Potassium","Healthy"]
    s = f1_from_predictions(y_true, y_pred, labels=[0,1,2], label_names=names)
    print("Macro F1:", round(s.macro_f1,4), "Micro F1:", round(s.micro_f1,4), "Weighted F1:", round(s.weighted_f1,4))
    if PANDAS_AVAILABLE:
        print(classification_report_df(s).to_string(index=False))
