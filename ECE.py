# ece_leaf.py
# Usage:
#   from ece_leaf import CLASS_NAMES, ece_overall, ece_per_class_ovr, reliability_plot
#   ece = ece_overall(y_true, y_proba, n_bins=15)                 # overall ECE
#   per_cls = ece_per_class_ovr(y_true, y_proba, n_bins=15)       # dict {class -> ECE}
#   reliability_plot(y_true, y_proba, save="reliability.png")     # optional plot
#
# Requirements: numpy (matplotlib only if you call reliability_plot)
# y_true can be strings or ints; y_proba is (N, C) softmax probabilities.

from typing import List, Any, Dict, Tuple
import numpy as np

# 1) Canonical class order for your project
CLASS_NAMES = [
    "Nitrogen",
    "Potassium",
    "Nitrogen and Potassium",
    "Healthy",
    "Leaf miner",
]

# 2) Aliases / typo normalisation
ALIASES = {
    "n": "Nitrogen", "nitrogen": "Nitrogen",
    "k": "Potassium", "potassium": "Potassium",
    "n and k": "Nitrogen and Potassium",
    "nitrogen and potassium": "Nitrogen and Potassium",
    "healthy": "Healthy",
    "leafminer": "Leaf miner", "leaf miner": "Leaf miner", "lead miner": "Leaf miner",
}

def canon(x: Any) -> str:
    key = str(x).strip().lower()
    return ALIASES.get(key, x if x in CLASS_NAMES else key.title())

def _to_indices(y_true: List[Any], class_names: List[str]) -> np.ndarray:
    """Map possibly-string labels to class indices 0..C-1 using canon()."""
    name_to_idx = {c: i for i, c in enumerate(class_names)}
    idxs = []
    for y in y_true:
        cy = canon(y)
        if cy not in name_to_idx:
            raise ValueError(f"Unknown label after normalisation: {y!r} → {cy!r}")
        idxs.append(name_to_idx[cy])
    return np.asarray(idxs, dtype=int)

def _bin_stats(conf: np.ndarray, correct: np.ndarray, n_bins: int = 15, strategy: str = "uniform") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return per-bin (prop, acc, conf_mean).
    strategy: 'uniform' -> equal-width bins in [0,1], 'quantile' -> equal-mass bins.
    """
    mask = np.isfinite(conf)
    conf = conf[mask]; correct = correct[mask].astype(float)
    if conf.size == 0:
        return np.zeros(0), np.zeros(0), np.zeros(0)

    if strategy == "quantile":
        # Build quantile edges; ensure unique & within [0,1]
        qs = np.linspace(0, 1, n_bins + 1)
        edges = np.unique(np.quantile(conf, qs))
        if edges.size < 2:
            edges = np.array([0.0, 1.0])
    else:  # uniform
        edges = np.linspace(0.0, 1.0, n_bins + 1)

    props, accs, confs = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        # Include right edge only for the final bin
        if hi == edges[-1]:
            m = (conf >= lo) & (conf <= hi)
        else:
            m = (conf >= lo) & (conf < hi)
        if not m.any():
            continue
        props.append(m.mean())
        accs.append(correct[m].mean())
        confs.append(conf[m].mean())
    return np.asarray(props), np.asarray(accs), np.asarray(confs)

def ece_overall(
    y_true: List[Any],
    y_proba: np.ndarray,
    class_names: List[str] = CLASS_NAMES,
    n_bins: int = 15,
    strategy: str = "uniform",
) -> float:
    """
    Multiclass ECE (max-confidence definition).
    - y_true: list/array of labels (strings or ints)
    - y_proba: (N, C) array of probabilities (rows ~ sum to 1)
    """
    y_true_idx = _to_indices(y_true, class_names)
    y_pred_idx = np.argmax(y_proba, axis=1)
    conf = np.max(y_proba, axis=1)
    correct = (y_pred_idx == y_true_idx).astype(float)

    props, accs, confs = _bin_stats(conf, correct, n_bins=n_bins, strategy=strategy)
    if props.size == 0:
        return 0.0
    ece = np.sum(props * np.abs(accs - confs))
    return float(ece)

def ece_per_class_ovr(
    y_true: List[Any],
    y_proba: np.ndarray,
    class_names: List[str] = CLASS_NAMES,
    n_bins: int = 15,
    strategy: str = "uniform",
) -> Dict[str, float]:
    """
    One-vs-Rest ECE for each class k:
      conf_k = p(model says class k)
      correct_k = 1 if true class == k else 0
    """
    y_true_idx = _to_indices(y_true, class_names)
    N, C = y_proba.shape
    if C != len(class_names):
        raise ValueError(f"y_proba has {C} columns but {len(class_names)} class_names provided.")
    out = {}
    for k, name in enumerate(class_names):
        conf_k = y_proba[:, k]
        correct_k = (y_true_idx == k).astype(float)
        props, accs, confs = _bin_stats(conf_k, correct_k, n_bins=n_bins, strategy=strategy)
        ece_k = float(np.sum(props * np.abs(accs - confs))) if props.size else 0.0
        out[name] = ece_k
    return out

# Optional: quick visual to pair with ECE (overall)
def reliability_plot(
    y_true: List[Any],
    y_proba: np.ndarray,
    class_names: List[str] = CLASS_NAMES,
    n_bins: int = 15,
    strategy: str = "uniform",
    save: str = "reliability.png",
):
    import matplotlib.pyplot as plt
    y_true_idx = _to_indices(y_true, class_names)
    conf = np.max(y_proba, axis=1)
    y_pred_idx = np.argmax(y_proba, axis=1)
    correct = (y_pred_idx == y_true_idx).astype(float)

    props, accs, confs = _bin_stats(conf, correct, n_bins=n_bins, strategy=strategy)
    if props.size == 0:
        print("No data for reliability plot.")
        return

    ece = np.sum(props * np.abs(accs - confs))

    plt.figure(figsize=(5.5, 5.5))
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect")
    plt.scatter(confs, accs, s=(props * 800 + 10), alpha=0.8, label="Bins")  # bubble size ~ bin mass
    plt.xlabel("Mean confidence (bin)")
    plt.ylabel("Accuracy (bin)")
    plt.title(f"Reliability (overall) — ECE={ece:.3f}")
    plt.xlim(0, 1); plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save, dpi=200)
    plt.close()

# --- Demo ---
if __name__ == "__main__":
    # Toy example with your 5 classes
    rng = np.random.default_rng(0)
    y_true = ["Healthy","Leaf miner","Nitrogen","Potassium","Nitrogen and Potassium"] * 40
    # fake probs (not calibrated)
    C = len(CLASS_NAMES)
    logits = rng.normal(size=(len(y_true), C))
    y_proba = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)

    print("Overall ECE:", round(ece_overall(y_true, y_proba), 3))
    print("Per-class ECE:", {k: round(v,3) for k,v in ece_per_class_ovr(y_true, y_proba).items()})
    # reliability_plot(y_true, y_proba)  # uncomment to save figure
