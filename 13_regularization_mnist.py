# regularization_cifar10.py
# Compare L2, Dropout, Data Augmentation, Early Stopping (and Residual blocks) on CIFAR-10.
import os, math, copy, time, random
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms

def set_seed(seed=42):
    random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed); torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(42)

def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------
# Small ConvNet(s)
# ---------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, dropout_p=0.0):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.dropout = nn.Dropout2d(dropout_p)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x, inplace=True)
        x = self.dropout(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, c, dropout_p=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c)
        self.dropout = nn.Dropout2d(dropout_p)
    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)), inplace=True)
        y = self.dropout(y)
        y = self.bn2(self.conv2(y))
        return F.relu(x + y, inplace=True)

class TinyCNN(nn.Module):
    def __init__(self, base=32, dropout_p=0.0, use_residual=False):
        super().__init__()
        self.use_residual = use_residual
        self.b1 = ConvBlock(3, base, dropout_p)
        self.b2 = ConvBlock(base, base, dropout_p)
        self.pool1 = nn.MaxPool2d(2)  # 32->16
        self.b3 = ConvBlock(base, 2*base, dropout_p)
        self.b4 = ConvBlock(2*base, 2*base, dropout_p)
        self.pool2 = nn.MaxPool2d(2)  # 16->8
        if use_residual:
            self.res = ResidualBlock(2*base, dropout_p)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(2*base, 10)
        )
    def forward(self, x):
        x = self.b1(x); x = self.b2(x); x = self.pool1(x)
        x = self.b3(x); x = self.b4(x); x = self.pool2(x)
        if self.use_residual:
            x = self.res(x)
        return self.head(x)

# ---------------------------
# Config + Data
# ---------------------------
@dataclass
class Config:
    name: str
    batch_size: int = 256
    epochs: int = 20
    lr: float = 3e-4
    weight_decay: float = 0.0      # L2
    dropout_p: float = 0.0         # Dropout prob in conv blocks
    use_aug: bool = False          # Data augmentation
    early_stop_patience: int = 6
    use_residual: bool = False
    train_subset: int = 20000      # None for full 50k
    device: str = pick_device()

def get_transforms(cfg: Config):
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)
    if cfg.use_aug:
        train_tfms = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        train_tfms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    test_tfms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_tfms, test_tfms

def get_loaders(cfg: Config):
    train_tfms, test_tfms = get_transforms(cfg)
    train_full = datasets.CIFAR10(root="./data", train=True, download=True, transform=train_tfms)
    test_ds    = datasets.CIFAR10(root="./data", train=False, download=True, transform=test_tfms)

    # Optional: downsample training set for speed
    if cfg.train_subset is not None and cfg.train_subset < len(train_full):
        idx = torch.randperm(len(train_full))[:cfg.train_subset]
        train_full = Subset(train_full, idx.tolist())

    # Validation split
    val_ratio = 0.1
    n_full = len(train_full)
    n_val = int(n_full * val_ratio)
    n_train = n_full - n_val
    train_ds, val_ds = random_split(train_full, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=512, shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=512, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, val_loader, test_loader

# ---------------------------
# Train / Eval
# ---------------------------
def evaluate(model, loader, device):
    model.eval()
    loss_sum, correct, n = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss_sum += loss.item() * y.size(0)
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            n += y.size(0)
    return loss_sum / n, correct / n

def train_one(cfg: Config):
    device = cfg.device
    train_loader, val_loader, test_loader = get_loaders(cfg)
    model = TinyCNN(base=32, dropout_p=cfg.dropout_p, use_residual=cfg.use_residual).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)  # AdamW: decoupled wd

    best = {"state": copy.deepcopy(model.state_dict()), "val": math.inf}
    no_improve = 0

    print(f"\n=== {cfg.name} ===")
    print(f"Device={device} | nTrain={len(train_loader.dataset)} | L2={cfg.weight_decay} | Dropout={cfg.dropout_p} | Aug={cfg.use_aug} | Residual={cfg.use_residual}")

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running = 0.0; seen = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            opt.step()
            running += loss.item() * y.size(0)
            seen += y.size(0)
        train_loss = running / seen

        val_loss, val_acc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch:02d} | train {train_loss:.4f} | val {val_loss:.4f} | val acc {val_acc:.4f}")

        if val_loss < best["val"] - 1e-4:
            best["val"] = val_loss
            best["state"] = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= cfg.early_stop_patience:
                print("Early stopping.")
                break

    model.load_state_dict(best["state"])
    val_loss, val_acc = evaluate(model, val_loader, device)
    test_loss, test_acc = evaluate(model, test_loader, device)
    print(f"[{cfg.name}] Val loss {val_loss:.4f} | Val acc {val_acc:.4f} | Test loss {test_loss:.4f} | Test acc {test_acc:.4f}")
    return {"name": cfg.name, "val_acc": val_acc, "test_acc": test_acc}

# ---------------------------
# Experiment suite
# ---------------------------
if __name__ == "__main__":
    suites = [
        Config(name="Baseline",          weight_decay=0.0,   dropout_p=0.0, use_aug=False, use_residual=False, epochs=20, train_subset=20000),
        Config(name="L2 (wd=5e-4)",     weight_decay=5e-4,   dropout_p=0.0, use_aug=False, use_residual=False, epochs=20, train_subset=20000),
        Config(name="Dropout (p=0.3)",  weight_decay=0.0,    dropout_p=0.3, use_aug=False, use_residual=False, epochs=20, train_subset=20000),
        Config(name="Augmentation",     weight_decay=0.0,    dropout_p=0.0, use_aug=True,  use_residual=False, epochs=20, train_subset=20000),
        Config(name="L2+Dropout+Aug",   weight_decay=5e-4,   dropout_p=0.2, use_aug=True,  use_residual=False, epochs=20, train_subset=20000),
        Config(name="Residual+L2",      weight_decay=5e-4,   dropout_p=0.1, use_aug=False, use_residual=True,  epochs=20, train_subset=20000),
    ]

    results = []
    t0 = time.time()
    for cfg in suites:
        results.append(train_one(cfg))
    print("\n=== Summary (Val/Test Acc) ===")
    for r in results:
        print(f"{r['name']:>18} | {r['val_acc']:.4f} / {r['test_acc']:.4f}")
    print(f"Total time: {time.time() - t0:.1f}s")
