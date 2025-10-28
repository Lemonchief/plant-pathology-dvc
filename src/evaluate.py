import argparse, json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from src.models import build_model
from src.utils import get_device, get_loaders

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--stats", type=str, required=True)
    ap.add_argument("--model-path", type=str, required=True)
    ap.add_argument("--out", type=str, default="metrics/eval.json")
    ap.add_argument("--cm", type=str, default="plots/confusion_matrix.csv")
    args = ap.parse_args()

    stats = json.load(open(args.stats))
    mean, std = stats["mean"], stats["std"]

    train_meta = json.load(open("metrics/train.json"))
    model_type = train_meta["model"]

    _, _, test_loader = get_loaders(args.data, batch_size=256, mean=mean, std=std, model_type=model_type)

    device = get_device()
    model = build_model(model_type).to(device)
    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    ys, ps = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            logits = model(x)
            ys.extend(y.numpy().tolist())
            ps.extend(logits.argmax(1).cpu().numpy().tolist())

    acc = accuracy_score(ys, ps)
    f1m = f1_score(ys, ps, average="macro")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"test_acc": round(acc,5), "test_f1_macro": round(f1m,5)}, f, indent=2)

    cm = confusion_matrix(ys, ps)
    Path(args.cm).parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(args.cm, cm, fmt="%d", delimiter=",")

if __name__ == "__main__":
    main()
