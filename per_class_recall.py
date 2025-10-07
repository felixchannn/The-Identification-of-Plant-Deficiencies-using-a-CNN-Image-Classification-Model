# per_class_recall_leaf.py
from typing import List, Any, Dict
import numpy as np

# 1) Canonical class order
CLASS_NAMES = [
    "Nitrogen",
    "Potassium",
    "Nitrogen and Potassium",
    "Healthy",
    "Leaf miner",
]

# 2) Light normaliser for label variants / typos
ALIASES = {
    "n": "Nitrogen",
    "nitrogen": "Nitrogen",
    "k": "Potassium",
    "potassium": "Potassium",
    "n and k": "Nitrogen and Potassium",
    "nitrogen and potassium": "Nitrogen and Potassium",
    "healthy": "Healthy",
    "leafminer": "Leaf miner",
    "leaf miner": "Leaf miner",
    "lead miner": "Leaf miner",   # common typo
}

def canon(x: Any) -> str:
    key = str(x).strip().lower()
    return ALIASES.get(key, x if x in CLASS_NAMES else key.title())

# 3) Per-class recall via sklearn (fallback to pure Python if not installed)
def per_class_recall(y_true: List[Any], y_pred: List[Any], labels: List[str] = CLASS_NAMES) -> Dict[str, float]:
    y_true = [canon(x) for x in y_true]
    y_pred = [canon(x) for x in y_pred]
    try:
        from sklearn.metrics import precision_recall_fscore_support
        _, rec, _, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
        return {lbl: float(r) for lbl, r in zip(labels, rec)}
    except Exception:
        # pure Python fallback using a confusion matrix
        idx = {c:i for i,c in enumerate(labels)}
        cm = np.zeros((len(labels), len(labels)), dtype=int)  # rows=actual, cols=pred
        for t,p in zip(y_true, y_pred):
            if t in idx and p in idx:
                cm[idx[t], idx[p]] += 1
        row_sums = cm.sum(axis=1)
        diag = np.diag(cm)
        rec = np.divide(diag, row_sums, out=np.zeros_like(diag, dtype=float), where=row_sums>0)
        return {lbl: float(r) for lbl, r in zip(labels, rec)}

# 4) (Optional) Confusion matrix plot with the same labels/order
def plot_confusion(y_true, y_pred, labels: List[str] = CLASS_NAMES, save="confusion_norm_true.png"):
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    import matplotlib.pyplot as plt
    y_true = [canon(x) for x in y_true]
    y_pred = [canon(x) for x in y_pred]
    cm_norm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=labels)
    disp.plot(xticks_rotation=45)
    plt.tight_layout(); plt.savefig(save, dpi=200); plt.close()

# --- Example usage ---
if __name__ == "__main__":
    # toy demo
    y_true = ["Healthy","Leaf miner","N and K","Nitrogen","Potassium","Lead miner","Nitrogen"]
    y_pred = ["Healthy","Leaf miner","Nitrogen and Potassium","Potassium","Potassium","Leaf miner","Nitrogen"]
    recalls = per_class_recall(y_true, y_pred)
    print("Per-class recall:")
    for k in CLASS_NAMES:
        print(f"{k:24s} {recalls[k]:.3f}")
    # plot_confusion(y_true, y_pred)  # uncomment to save the row-normalised CM
