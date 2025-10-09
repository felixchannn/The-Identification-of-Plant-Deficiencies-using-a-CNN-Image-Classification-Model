import argparse
from pathlib import Path
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def load_model(weights_path, n_classes, device):
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
    sd = {k.replace("module.","").replace("model.",""): v for k,v in sd.items()}
    m.load_state_dict(sd, strict=True)
    return m.eval().to(device)

@torch.no_grad()
def infer(model, loader, device):
    y_true, y_pred = [], []
    for x, y in loader:
        logits = model(x.to(device))
        y_true.extend(y.cpu().numpy().tolist())
        y_pred.extend(logits.argmax(1).cpu().numpy().tolist())
    return np.array(y_true), np.array(y_pred)

def run_eval(data_dir, weights_path, out_dir="eval_out", img_size=224, batch=32, num_workers=2):
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
    y_true, y_pred = infer(model, loader, device)

    labels_range = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels_range)
    cm_norm = confusion_matrix(y_true, y_pred, labels=labels_range, normalize="true")

    np.savetxt(out/"confusion_counts.csv", cm, fmt="%d", delimiter=",")
    np.savetxt(out/"confusion_rownorm.csv", cm_norm, fmt="%.4f", delimiter=",")

    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    disp.plot(xticks_rotation=45)
    plt.tight_layout(); plt.savefig(out/"confusion_raw.png", dpi=200); plt.close()

    disp = ConfusionMatrixDisplay(cm_norm, display_labels=class_names)
    disp.plot(xticks_rotation=45)
    plt.tight_layout(); plt.savefig(out/"confusion_norm_true.png", dpi=200); plt.close()

    print(f"Saved to {out.resolve()}")
    print("Tip: read the row-normalised plot first to see which classes you’re missing (low diagonal values).")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to test set (ImageFolder).")
    ap.add_argument("--weights", required=True, help=".pt/.pth checkpoint")
    ap.add_argument("--out", default="eval_out")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=2)
    args = ap.parse_args()
    run_eval(args.data, args.weights, args.out, args.img_size, args.batch, args.num_workers)

if __name__ == "__main__":
    main()
