from pathlib import Path
import random
import numpy as np
import torch
from torchvision import transforms, datasets

CLASSES = ["healthy", "multiple_diseases", "rust", "scab"]

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device():
    if torch.cuda.is_available():
        print("Using CUDA GPU")
        return torch.device("cuda")
    else:
        print("Using CPU (no GPU acceleration)")
        return torch.device("cpu")


def build_transforms(mean, std, model_type: str, train: bool):
    ops = []
    if model_type == "resnet18":
        ops += [transforms.Grayscale(num_output_channels=3), transforms.Resize((224, 224))]
    else:
        ops += [transforms.Grayscale(num_output_channels=1), transforms.Resize((28, 28))]
    ops += [transforms.ToTensor()]
    if model_type == "resnet18":
        ops += [transforms.Normalize([mean]*3, [std]*3)]
    else:
        ops += [transforms.Normalize([mean], [std])]
    return transforms.Compose(ops)

def get_loaders(data_dir: str, batch_size: int, mean: float, std: float, model_type: str):
    data_dir = Path(data_dir)
    tf_train = build_transforms(mean, std, model_type, train=True)
    tf_eval = build_transforms(mean, std, model_type, train=False)

    ds_train = datasets.ImageFolder(data_dir / "train", transform=tf_train)
    ds_val   = datasets.ImageFolder(data_dir / "val",   transform=tf_eval)
    ds_test  = datasets.ImageFolder(data_dir / "test",  transform=tf_eval)

    train_loader = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader   = torch.utils.data.DataLoader(ds_val,   batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader  = torch.utils.data.DataLoader(ds_test,  batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, val_loader, test_loader
