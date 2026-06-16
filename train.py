import argparse
import torch
import torch.nn as nn
import wandb
from sklearn.metrics import f1_score

from config import Config
from data import get_dataloaders
from model import SmallCNN


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / total
    acc = correct / total
    f1 = f1_score(all_labels, all_preds, average="macro")
    return avg_loss, acc, f1


def train_one_run(run_name, model, config, extra_config=None):
    device = config.DEVICE
    model.to(device)

    train_loader, val_loader = get_dataloaders(
        config.DATA_PATH, config.BATCH_SIZE, config.VAL_SPLIT, config.SEED
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)

    wandb_config = {
        "lr": config.LR,
        "batch_size": config.BATCH_SIZE,
        "epochs": config.EPOCHS,
        "architecture": model.__class__.__name__,
    }
    if extra_config:
        wandb_config.update(extra_config)

    wandb.init(
        project=config.WANDB_PROJECT,
        name=run_name,
        config=wandb_config,
    )
    wandb.watch(model, log="all", log_freq=50)

    for epoch in range(config.EPOCHS):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, device)

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_f1_macro": val_f1,
        })

        print(f"Epoch {epoch+1}/{config.EPOCHS} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}")

    wandb.finish()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default="small_cnn_v1")
    args = parser.parse_args()

    config = Config()
    model = SmallCNN(num_classes=config.NUM_CLASSES)
    train_one_run(args.run_name, model, config)