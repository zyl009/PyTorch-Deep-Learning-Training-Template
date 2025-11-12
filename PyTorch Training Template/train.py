import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from config import CONFIG
from dataset_loader import get_dataloaders
from model_utils import create_resnet18
from train_utils import train_one_epoch, validate


class EarlyStopping:
    def __init__(self, patience=5, delta=1e-4):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, current_score):
        if self.best_score is None:
            self.best_score = current_score
            return False
        if current_score < self.best_score + self.delta:
            self.counter += 1
            print(f" EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.should_stop = True
                print(" Early stopping triggered.")
                return True
        else:
            self.best_score = current_score
            self.counter = 0
        return False


def main():
    cfg = CONFIG
    device = cfg["device"]

    train_loader, val_loader = get_dataloaders(cfg["data_dir"], cfg["batch_size"])
    model = create_resnet18(cfg["num_classes"]).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"])

    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5,
                                  patience=2, threshold=1e-4, verbose=True)

    early_stopper = EarlyStopping(patience=5, delta=1e-4)
    best_acc = 0.0

    for epoch in range(cfg["epochs"]):
        print(f"\nEpoch {epoch+1}/{cfg['epochs']}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.2f}")
        print(f"Val   Loss: {val_loss:.4f} | Acc: {val_acc:.2f}")

        scheduler.step(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), cfg["save_path"])
            print(f" Saved new best model ({best_acc:.2f}%)")

        if early_stopper(val_acc):
            break

        print(f"Current LR: {optimizer.param_groups[0]['lr']:.6f}")

    print(f"\nTraining complete. Best val acc: {best_acc:.2f}%")


if __name__ == "__main__":
    main()
