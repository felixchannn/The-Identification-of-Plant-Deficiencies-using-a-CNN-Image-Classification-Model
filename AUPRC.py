# AUPRC.py
import argparse
from pathlib import Path
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

def load_model(weights_path, n_classes, device):
    """
    Adjust this if your backbone isn't ResNet18.
    Tries to load a full model; if that fails, builds a ResNet18 head and loads state_dict.
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
def infer_probas(model, loader, device):
    """Return y_true (N,) and y_proba (N, C) softmax probabilities."""
    y_true = []
    probs = []
    for x, y in loader:
        logits = model(x.to(device))
        p = torch.softmax(logits, dim=1).cpu().numpy()
        probs.append(p)
        y_true.append(y.cpu().numpy())
    y_true = np.concatenate(y_true, axis=0)
    y_proba = np.concatenate(probs, axis=0)
    return y_true, y_proba

def _save_pr_curve(recall, precision, ap, class_name, out_dir):
    # Plot PR curve for a single class
    plt.figure()
    plt.plot(recall, precision, label=f'AP={ap:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'PR Curve — {class_name}')
    plt.legend(loc='lower left')
    plt.tight_layout()
    fn = Path(out_dir) / f'pr_{class_name}.png'
    plt.savefig(fn, dpi=200)
    plt.close()

def run_eval(data_dir, weights_path, out_dir="eval_auprc", img_size=224, batch=32, num_workers=2):
    """
    Computes per-class Average Precision (AP ≈ AUPRC), macro AP, and micro AP.
    Saves per-class PR plots and CSVs.
    """
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)

    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    ds = datasets.ImageFolder(data_dir, transform=tfm)
    class_names = ds.classes
    n_classes = len(class_names)

    loader = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=num_workers)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_model(weights_path, n_classes=n_classes, device=device)
    y_true, y_proba = infer_probas(model, loader, device)

    # One-vs-rest AP per class
    aps = []
    for c in range(n_classes):
        y_true_bin = (y_true == c).astype(np.int32)   # (N,)
        y_score = y_proba[:, c]                       # (N,)
        precision, recall, thresholds = precision_recall_curve(y_true_bin, y_score)
        ap = average_precision_score(y_true_bin, y_score)
        aps.append(ap)

        # Save per-class CSV (align thresholds to precision/recall length)
        # thresholds has length len(precision)-1; pad with NaN to same length
        th = np.full_like(precision, np.nan, dtype=np.float64)
        if thresholds.size > 0:
            th[1:] = thresholds
        pr_mat = np.stack([recall, precision, th], axis=1)
        np.savetxt(out / f'pr_{class_names[c]}.csv', pr_mat,
                   fmt=['%.6f', '%.6f', '%.6f'], delimiter=',',
                   header='recall,precision,threshold', comments='')

        _save_pr_curve(recall, precision, ap, class_names[c], out)

    # Macro AP (unweighted mean)
    macro_ap = float(np.mean(aps)) if aps else float("nan")

    # Micro AP (flatten one-vs-rest)
    y_true_ovr = np.zeros((y_true.size, n_classes), dtype=np.int32)
    y_true_ovr[np.arange(y_true.size), y_true] = 1
    micro_ap = average_precision_score(y_true_ovr.ravel(), y_proba.ravel())

    # Save summary
    with open(out / "ap_summary.txt", "w") as f:
        for name, ap in zip(class_names, aps):
            f.write(f"{name}\t{ap:.6f}\n")
        f.write(f"\nMACRO_AP\t{macro_ap:.6f}\n")
        f.write(f"MICRO_AP\t{micro_ap:.6f}\n")

    np.savetxt(out / "ap_per_class.csv",
               np.c_[np.arange(n_classes), aps],
               fmt=['%d', '%.6f'], delimiter=',',
               header='class_index,AP', comments='')

    print("Per-class AP:")
    for name, ap in zip(class_names, aps):
        print(f"  {name:>20s}: {ap:.4f}")
    print(f"\nMacro AP: {macro_ap:.4f}")
    print(f"Micro AP: {micro_ap:.4f}")
    print(f"Outputs saved to {out.resolve()}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to set (ImageFolder).")
    ap.add_argument("--weights", required=True, help=".pt/.pth checkpoint")
    ap.add_argument("--out", default="eval_auprc")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=2)
    args = ap.parse_args()

    run_eval(
        data_dir=args.data,
        weights_path=args.weights,
        out_dir=args.out,
        img_size=args.img_size,
        batch=args.batch,
        num_workers=args.num_workers
    )

if __name__ == "__main__":
    main()
