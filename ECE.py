# ECE.py
# Computes Expected Calibration Error (overall and per-class OVR) and
# optionally saves a reliability plot — fed by a PyTorch model + ImageFolder.
#
# Usage (standalone):
#   python ECE.py --data ./my_test_folder --weights ./model.pth --out eval_ece
#
# Outputs:
#   eval_ece/ece_overall.txt
#   eval_ece/ece_per_class.csv
#   eval_ece/reliability.png

import argparse
from pathlib import Path
from typing import List, Any, Dict, Tuple
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# ----------------------------
# ECE helpers (self-contained)
# ----------------------------

# You can swap in your own canonical order if needed
CLASS_NAMES_DEFAULT: List[str] = None  # we’ll fill from ImageFolder at runtime

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
        qs = np.linspace(0, 1, n_bins + 1)
        edges = np.unique(np.quantile(conf, qs))
        if edges.size < 2:
            edges = np.array([0.0, 1.0])
    else:
        edges = np.linspace(0.0, 1.0, n_bins + 1)

    props, accs, confs = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        # Include right edge only for the final bin
        m = (conf >= lo) & (conf <= hi) if hi == edges[-1] else (conf >= lo) & (conf < hi)
        if not m.any():
            continue
        props.append(m.mean())
        accs.append(correct[m].mean())
        confs.append(conf[m].mean())
    return np.asarray(props), np.asarray(accs), np.asarray(confs)

def ece_overall(y_true_idx: np.ndarray, y_proba: np.ndarray, n_bins: int = 15, strategy: str = "uniform") -> float:
    """
    Multiclass ECE (max-confidence definition).
    - y_true_idx: (N,) int ground-truth indices
    - y_proba: (N, C) probabilities
    """
    y_pred_idx = np.argmax(y_proba, axis=1)
    conf = np.max(y_proba, axis=1)
    correct = (y_pred_idx == y_true_idx).astype(float)
    props, accs, confs = _bin_stats(conf, correct, n_bins=n_bins, strategy=strategy)
    if props.size == 0:
        return 0.0
    return float(np.sum(props * np.abs(accs - confs)))

def ece_per_class_ovr(y_true_idx: np.ndarray, y_proba: np.ndarray, class_names: List[str], n_bins: int = 15, strategy: str = "uniform") -> Dict[str, float]:
    """
    One-vs-Rest ECE for each class k:
      conf_k = p(model says class k)
      correct_k = 1 if true class == k else 0
    """
    N, C = y_proba.shape
    if C != len(class_names):
        raise ValueError(f"y_proba has {C} columns but {len(class_names)} class_names provided.")
    out = {}
    for k, name in enumerate(class_names):
        conf_k = y_proba[:, k]
        correct_k = (y_true_idx == k).astype(float)
        props, accs, confs = _bin_stats(conf_k, correct_k, n_bins=n_bins, strategy=strategy)
        out[name] = float(np.sum(props * np.abs(accs - confs))) if props.size else 0.0
    return out

def reliability_plot(y_true_idx: np.ndarray, y_proba: np.ndarray, save: Path, n_bins: int = 15, strategy: str = "uniform"):
    import matplotlib.pyplot as plt
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
    sizes = (props * 800 + 10)  # bubble size ~ bin mass
    plt.scatter(confs, accs, s=sizes, alpha=0.8, label="Bins")
    plt.xlabel("Mean confidence (bin)")
    plt.ylabel("Accuracy (bin)")
    plt.title(f"Reliability — ECE={ece:.3f}")
    plt.xlim(0, 1); plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save, dpi=200)
    plt.close()

# ----------------------------
# Torch model + data
# ----------------------------

def load_model(weights_path: Path, n_classes: int, device: str):
    """
    Adjust this to your backbone if not ResNet18.
    Tries to load a full model; if that fails, builds ResNet18 head and loads state_dict.
    """
    try:
        m = torch.load(weights_path, map_location=device)
        if hasattr(m, "eval"):
            return m.to(device).eval()
    except Exception:
        pass
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, n_classes)
    state = torch.load(weights_path, map_location=device)
    sd = state.get("state_dict", state)
    sd = {k.replace("module.","").replace("model.",""): v for k, v in sd.items()}
    m.load_state_dict(sd, strict=True)
    return m.eval().to(device)

@torch.no_grad()
def infer_probas(model, loader, device: str):
    """Return y_true_idx (N,) and y_proba (N, C) softmax probabilities."""
    y_true, probs = [], []
    for x, y in loader:
        logits = model(x.to(device))
        p = torch.softmax(logits, dim=1).cpu().numpy()
        probs.append(p)
        y_true.append(y.cpu().numpy())
    y_true_idx = np.concatenate(y_true, axis=0)
    y_proba = np.concatenate(probs, axis=0)
    return y_true_idx, y_proba

def run_eval(
    data_dir: str,
    weights_path: str,
    out_dir: str = "eval_ece",
    img_size: int = 224,
    batch: int = 32,
    num_workers: int = 2,
    n_bins: int = 15,
    strategy: str = "uniform",  # or "quantile"
    save_plot: bool = True,
):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)

    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    ds = datasets.ImageFolder(data_dir, transform=tfm)
    class_names = ds.classes
    loader = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=num_workers)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_model(weights_path, n_classes=len(class_names), device=device)
    y_true_idx, y_proba = infer_probas(model, loader, device)

    # ECEs
    ece = ece_overall(y_true_idx, y_proba, n_bins=n_bins, strategy=strategy)
    per_cls = ece_per_class_ovr(y_true_idx, y_proba, class_names, n_bins=n_bins, strategy=strategy)

    # Save numbers
    (out / "ece_overall.txt").write_text(f"{ece:.6f}\n")
    np.savetxt(out / "ece_per_class.csv",
               np.c_[np.arange(len(class_names)), [per_cls[name] for name in class_names]],
               fmt=['%d','%.6f'], delimiter=',', header='class_index,ECE', comments='')
    (out / "classes.txt").write_text("\n".join(f"{i},{name}" for i, name in enumerate(class_names)))

    # Reliability plot
    if save_plot:
        reliability_plot(y_true_idx, y_proba, save=out / "reliability.png", n_bins=n_bins, strategy=strategy)

    # Console summary
    print("ECE (overall):", round(ece, 6))
    print("ECE per class:")
    for name in class_names:
        print(f"  {name:>20s}: {per_cls[name]:.6f}")
    print(f"Saved to {out.resolve()}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to ImageFolder (eval set).")
    ap.add_argument("--weights", required=True, help=".pt/.pth checkpoint")
    ap.add_argument("--out", default="eval_ece")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--bins", type=int, default=15, help="ECE bins")
    ap.add_argument("--strategy", choices=["uniform","quantile"], default="uniform", help="binning strategy")
    ap.add_argument("--no-plot", action="store_true", help="skip reliability plot")
    args = ap.parse_args()

    run_eval(
        data_dir=args.data,
        weights_path=args.weights,
        out_dir=args.out,
        img_size=args.img_size,
        batch=args.batch,
        num_workers=args.num_workers,
        n_bins=args.bins,
        strategy=args.strategy,
        save_plot=not args.no_plot
    )

if __name__ == "__main__":
    main()
