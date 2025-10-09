# per_class_recall.py
# Computes per-class recall from a PyTorch model + ImageFolder dataset.
# Also saves a row-normalised confusion matrix plot (optional).
#
# Usage (standalone):
#   python per_class_recall.py --data ./my_test_folder --weights ./model.pth --out eval_recall
#
# Outputs:
#   eval_recall/per_class_recall.csv
#   eval_recall/summary.txt
#   eval_recall/confusion_norm_true.png  (optional)
#   eval_recall/classes.txt
#
# You can also import and use `per_class_recall_from_arrays(y_true, y_pred, labels)` directly.

import argparse
from pathlib import Path
from typing import List, Any, Dict
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# ----------------------------
# Utilities
# ----------------------------

CLASS_NAMES_DEFAULT: List[str] = None  # we fill from ImageFolder at runtime

# Light normaliser (you can extend/remove if not needed)
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
    "lead miner": "Leaf miner",
}
def canon(x: Any) -> str:
    key = str(x).strip().lower()
    return ALIASES.get(key, key.title())

def per_class_recall_from_arrays(y_true: List[Any], y_pred: List[Any], labels: List[str]) -> Dict[str, float]:
    """Returns {label: recall} using sklearn if available, else a pure-NumPy fallback."""
    y_true = [canon(x) for x in y_true]
    y_pred = [canon(x) for x in y_pred]
    try:
        from sklearn.metrics import precision_recall_fscore_support
        _, rec, _, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
        return {lbl: float(r) for lbl, r in zip(labels, rec)}
    except Exception:
        # confusion-matrix fallback
        idx = {c:i for i,c in enumerate(labels)}
        cm = np.zeros((len(labels), len(labels)), dtype=int)  # rows=actual, cols=pred
        for t,p in zip(y_true, y_pred):
            if t in idx and p in idx:
                cm[idx[t], idx[p]] += 1
        row_sums = cm.sum(axis=1)
        diag = np.diag(cm)
        rec = np.divide(diag, row_sums, out=np.zeros_like(diag, dtype=float), where=row_sums>0)
        return {lbl: float(r) for lbl, r in zip(labels, rec)}

def plot_confusion_norm_true(y_true: List[Any], y_pred: List[Any], labels: List[str], save_png: Path):
    try:
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
        import matplotlib.pyplot as plt
    except Exception:
        print("Skipping confusion plot (sklearn/matplotlib not available).")
        return
    y_true = [canon(x) for x in y_true]
    y_pred = [canon(x) for x in y_pred]
    cm_norm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=labels)
    disp.plot(xticks_rotation=45)
    plt.tight_layout(); plt.savefig(save_png, dpi=200); plt.close()

# ----------------------------
# Torch model + data
# ----------------------------

def load_model(weights_path: str, n_classes: int, device: str):
    """
    Adjust if your backbone isn't ResNet18.
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
def infer_preds(model, loader, device: str):
    """Return y_true_idx (N,), y_pred_idx (N,), and class index -> name mapping from the dataset."""
    y_true, y_pred = [], []
    for x, y in loader:
        logits = model(x.to(device))
        p = torch.softmax(logits, dim=1)
        pred = p.argmax(1)
        y_true.extend(y.cpu().numpy().tolist())
        y_pred.extend(pred.cpu().numpy().tolist())
    return np.array(y_true), np.array(y_pred)

def run_eval(
    data_dir: str,
    weights_path: str,
    out_dir: str = "eval_recall",
    img_size: int = 224,
    batch: int = 32,
    num_workers: int = 2,
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
    y_true_idx, y_pred_idx = infer_preds(model, loader, device)

    # Convert to string labels using dataset class order
    idx_to_name = {i: name for i, name in enumerate(class_names)}
    y_true = [idx_to_name[i] for i in y_true_idx]
    y_pred = [idx_to_name[i] for i in y_pred_idx]

    recalls = per_class_recall_from_arrays(y_true, y_pred, labels=class_names)

    # Save numbers
    with open(out / "per_class_recall.csv", "w") as f:
        f.write("class,recall\n")
        for name in class_names:
            f.write(f"{name},{recalls[name]:.6f}\n")
    with open(out / "summary.txt", "w") as f:
        for name in class_names:
            f.write(f"{name:>24s}: {recalls[name]:.6f}\n")
        f.write(f"\nMacro recall: {np.mean(list(recalls.values())):.6f}\n")

    (out / "classes.txt").write_text("\n".join(f"{i},{name}" for i, name in enumerate(class_names)))

    # Optional plot
    if save_plot:
        plot_confusion_norm_true(y_true, y_pred, labels=class_names, save_png=out / "confusion_norm_true.png")

    # Console summary
    print("Per-class recall:")
    for name in class_names:
        print(f"  {name:>24s}: {recalls[name]:.6f}")
    print(f"Macro recall: {np.mean(list(recalls.values())):.6f}")
    print(f"Saved to {out.resolve()}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to ImageFolder (eval set).")
    ap.add_argument("--weights", required=True, help=".pt/.pth checkpoint")
    ap.add_argument("--out", default="eval_recall")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--no-plot", action="store_true", help="Skip confusion matrix plot")
    args = ap.parse_args()

    run_eval(
        data_dir=args.data,
        weights_path=args.weights,
        out_dir=args.out,
        img_size=args.img_size,
        batch=args.batch,
        num_workers=args.num_workers,
        save_plot=not args.no_plot
    )

if __name__ == "__main__":
    main()
