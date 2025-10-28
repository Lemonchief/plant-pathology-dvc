import argparse
from pathlib import Path
import json
from PIL import Image
import numpy as np
from src.utils import CLASSES

def compute_mean_std(train_dir: Path):
    s = 0.0; ss = 0.0; n = 0
    for cls in CLASSES:
        for p in (train_dir / cls).glob("*.jpg"):
            arr = np.asarray(Image.open(p).convert("L"), dtype=np.float32) / 255.0
            s  += arr.mean()
            ss += arr.var()
            n  += 1
    mean = s / n
    std  = (ss / n) ** 0.5
    return float(mean), float(std)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--out", type=str, default="artifacts/stats.json")
    args = ap.parse_args()

    train_dir = Path(args.data) / "train"
    mean, std = compute_mean_std(train_dir)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"mean": mean, "std": std}, f, indent=2)

if __name__ == "__main__":
    main()
