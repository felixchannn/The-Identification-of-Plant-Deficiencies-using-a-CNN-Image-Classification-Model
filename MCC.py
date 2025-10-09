# MCC.py
# Computes Matthews Correlation Coefficient (multiclass-capable in sklearn).
# Also saves a predictions CSV for auditing.
#
# Usage (standalone):
#   python MCC.py --data ./my_test_folder --weights ./model.pth --out eval_mcc
#
# Outputs:
#   eval_mcc/mcc.txt
#   eval_mcc/preds.csv
#   eval_mcc/classes.txt

import argparse
from pathlib import Path
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import matthews_corrcoef

def load_model(weights_path, n_classes, device):
    """
    Adjust this to your backbone if not ResNet18.
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
def infer(model, loader, device):
    y_true, y_pred, y_conf, y_paths = [], [], [], []
    for x, y in loader:
        logits = model(x.to(device))
        prob = torch.softmax(logits, dim=1)
        pred = prob.argmax(1)
        conf = prob.max(1).values
        y_true.extend(y.cpu().numpy().tolist())
        y_pred.extend(pred.cpu().numpy().tolist())
        y_conf.extend(conf.cpu().numpy().tolist())
        # Try to capture file paths if the dataset exposes them
        if hasattr(loader.dataset, "samples"):
            # samples = list[(path, class_idx), ...]; DataLoader gives batches in order for shuffle=False
            # We need the right slice for this batch; easier: fallback to None and fill later in a pass
            pass
    return np.array(y_true), np.array(y_pred), np.array(y_conf)

def run_eval(
    data_dir: str,
    weights_path: str,
    out_dir: str = "eval_mcc",
    img_size: int = 224,
    batch: int = 32,
    num_workers: int = 2,
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
    y_true, y_pred, y_conf = infer(model, loader, device)

    # MCC (multiclass handled by sklearn)
    mcc = matthews_corrcoef(y_true, y_pred)

    # Save metric
    (out / "mcc.txt").write_text(f"{mcc:.6f}\n")
    (out / "classes.txt").write_text("\n".join(f"{i},{name}" for i, name in enumerate(class_names)))

    # Save predictions CSV (path, true_idx, pred_idx, confidence_of_pred)
    # We can reconstruct paths from the underlying dataset order (since shuffle=False)
    paths = [p for p, _ in ds.samples]
    import csv
    with open(out / "preds.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "true_idx", "pred_idx", "pred_confidence"])
        for p, t, pr, c in zip(paths, y_true, y_pred, y_conf):
            w.writerow([p, int(t), int(pr), float(c)])

    print(f"MCC: {mcc:.6f}")
    print(f"Saved to {out.resolve()}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to ImageFolder (eval set).")
    ap.add_argument("--weights", required=True, help=".pt/.pth checkpoint")
    ap.add_argument("--out", default="eval_mcc")
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
