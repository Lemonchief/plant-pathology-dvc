import argparse, json
from pathlib import Path
import torch, torch.nn as nn
from torch.optim import Adam
from dvclive import Live
from src.models import build_model
from src.utils import set_seed, get_device, get_loaders

def accuracy(logits, y):
    preds = logits.argmax(dim=1)
    return (preds == y).float().mean().item()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--stats", type=str, required=True)
    ap.add_argument("--model", type=str, default="logreg")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device()

    stats = json.load(open(args.stats))
    mean, std = stats["mean"], stats["std"]

    train_loader, val_loader, _ = get_loaders(args.data, args.batch_size, mean, std, args.model)
    model = build_model(args.model).to(device)
    criterion = nn.CrossEntropyLoss()
    optim = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_acc = 0.0
    Path("models").mkdir(exist_ok=True, parents=True)
    Path("metrics").mkdir(exist_ok=True, parents=True)

    # HTML-отчёт и сохранение DVC-эксперимента включены тут
    with Live("dvclive", save_dvc_exp=True, report="html") as live:
        for epoch in range(1, args.epochs + 1):
            model.train()
            running_loss, running_acc, n = 0.0, 0.0, 0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optim.zero_grad()
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optim.step()
                running_loss += loss.item() * x.size(0)
                running_acc  += (logits.argmax(1) == y).sum().item()
                n += x.size(0)
            train_loss = running_loss / n
            train_acc  = running_acc / n

            model.eval()
            va_loss, va_acc, m = 0.0, 0.0, 0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    logits = model(x)
                    loss = criterion(logits, y)
                    va_loss += loss.item() * x.size(0)
                    va_acc  += (logits.argmax(1) == y).sum().item()
                    m += x.size(0)
            val_loss = va_loss / m
            val_acc  = va_acc / m

            live.log_metric("train/loss", train_loss)
            live.log_metric("train/acc",  train_acc)
            live.log_metric("val/loss",   val_loss)
            live.log_metric("val/acc",    val_acc)
            live.next_step()

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), "models/model.pt")

    with open("metrics/train.json", "w") as f:
        json.dump({
            "model": args.model,
            "best_val_acc": round(best_acc, 5),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
        }, f, indent=2)

if __name__ == "__main__":
    main()
