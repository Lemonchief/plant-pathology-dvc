import argparse, random
from pathlib import Path
from PIL import Image
from torchvision import transforms
from shutil import copy2
from src.utils import CLASSES, set_seed

AUG = transforms.Compose([
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
    transforms.RandomHorizontalFlip(p=0.5),
])

def augment_dir(src_dir: Path, dst_dir: Path, n_per_image: int, max_per_class: int | None):
    dst_dir.mkdir(parents=True, exist_ok=True)
    for cls in CLASSES:
        (dst_dir / cls).mkdir(parents=True, exist_ok=True)

    for cls in CLASSES:
        images = sorted((src_dir / cls).glob("*.jpg"))
        if max_per_class and len(images) > max_per_class:
            images = random.sample(images, max_per_class)

        for p in images:
            copy2(p, dst_dir / cls / p.name)
            img = Image.open(p).convert("L")
            for i in range(n_per_image):
                aug = AUG(img)
                aug.save(dst_dir / cls / f"{p.stem}_aug{i}.jpg")

def copy_split(src_dir: Path, dst_dir: Path):
    dst_dir.mkdir(parents=True, exist_ok=True)
    for cls in CLASSES:
        (dst_dir / cls).mkdir(parents=True, exist_ok=True)
        for p in (src_dir / cls).glob("*.jpg"):
            copy2(p, dst_dir / cls / p.name)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=str, default="data/raw")
    ap.add_argument("--out", type=str, default="data/augmented")
    ap.add_argument("--n-per-image", type=int, default=2)
    ap.add_argument("--max-per-class", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    src = Path(args.inp)
    dst = Path(args.out)

    augment_dir(src / "train", dst / "train", args.n_per_image, args.max_per_class or None)
    copy_split(src / "val",  dst / "val")
    copy_split(src / "test", dst / "test")

if __name__ == "__main__":
    main()
