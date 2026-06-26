import argparse
import datetime
import shutil

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchxrayvision as xrv
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Subset, random_split

from model import build_model


def masked_bce_loss(logits, labels):
    mask = ~torch.isnan(labels)
    labels_safe = torch.nan_to_num(labels, nan=0.0)
    per_elem = F.binary_cross_entropy_with_logits(logits, labels_safe, reduction="none")
    return (per_elem * mask).sum() / mask.sum().clamp(min=1)


@torch.no_grad()
def evaluate(model, loader, device, pathologies):
    model.eval()
    all_probs, all_labels = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        probs = torch.sigmoid(model(imgs)).cpu()
        all_probs.append(probs)
        all_labels.append(labels)
    probs = torch.cat(all_probs).numpy()
    targets = torch.cat(all_labels).numpy()

    aucs = {}
    for i, name in enumerate(pathologies):
        col = targets[:, i]
        valid = ~np.isnan(col)
        if valid.sum() == 0:
            continue
        pos_count = col[valid].sum()
        if pos_count == 0 or pos_count == valid.sum():
            aucs[name] = float("nan")
        else:
            aucs[name] = roc_auc_score(col[valid], probs[valid, i])

    valid_aucs = [v for v in aucs.values() if not np.isnan(v)]
    mean_auc = float(np.mean(valid_aucs)) if valid_aucs else float("nan")
    return aucs, mean_auc


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, n = 0.0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = masked_bce_loss(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        n += imgs.size(0)
    return total_loss / n


def collate(samples):
    imgs = torch.stack([torch.as_tensor(s["img"]).float() for s in samples])
    labels = torch.stack([torch.as_tensor(s["lab"]).float() for s in samples])
    return imgs, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", required=True, help="Folder containing all NIH .png images, flat.")
    parser.add_argument("--train_list", help="Path to NIH's train_val_list.txt")
    parser.add_argument("--test_list", help="Path to NIH's test_list.txt")
    parser.add_argument("--model", choices=["densenet", "simple"], default="densenet")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=None, help="Limit train/test sizes for quick testing.")
    parser.add_argument("--test_frac", type=float, default=0.15)
    parser.add_argument("--out", default="best.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([
        xrv.datasets.XRayCenterCrop(),
        xrv.datasets.XRayResizer(224),
    ])
    full_ds = xrv.datasets.NIH_Dataset(
        imgpath=args.image_dir,
        transform=transform,
        unique_patients=False,
    )
    xrv.datasets.relabel_dataset(xrv.datasets.default_pathologies, full_ds)
    pathologies = xrv.datasets.default_pathologies
    print(f"Dataset: {len(full_ds)} images, {len(pathologies)} pathology slots")
    print(f"Pathologies: {pathologies}")

    n_test = int(len(full_ds) * args.test_frac)
    n_train = len(full_ds) - n_test
    train_ds, test_ds = random_split(
        full_ds, [n_train, n_test],
        generator=torch.Generator().manual_seed(42),
    )

    if args.max_samples:
        train_ds = Subset(train_ds, range(min(args.max_samples, len(train_ds))))
        test_ds = Subset(test_ds, range(min(args.max_samples // 4, len(test_ds))))

    print(f"Train: {len(train_ds)}  |  Test: {len(test_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, collate_fn=collate,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, collate_fn=collate,
    )

    model = build_model(args.model).to(device)
    lr = args.lr if args.model == "densenet" else args.lr * 10
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model}  ({n_params:,} params)  lr={lr}")
    print(f"Start time: {datetime.datetime.now().strftime('%H:%M:%S')}\n")

    print('=' * shutil.get_terminal_size().columns)

    best_auc = 0.0
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, device)
        aucs, mean_auc = evaluate(model, test_loader, device, pathologies)
        scheduler.step(mean_auc)

        print(f"\nEpoch {epoch}/{args.epochs}  loss={loss:.4f}  mean AUC={mean_auc:.4f}")
        epoch_time = datetime.datetime.now().strftime('%H:%M:%S')
        print(f"Time: {epoch_time}\n")

        for name, v in aucs.items():
            print(f"  {name:<22s} {v:.4f}")

        torch.save(
            {"model": model.state_dict(), "aucs": aucs, "epoch": epoch, "time": epoch_time},
            "last.pt",
        )

        if mean_auc > best_auc:
            best_auc = mean_auc
            torch.save({"model": model.state_dict(), "aucs": aucs, "epoch": epoch}, args.out)
            print(f" -> saved new best ({best_auc:.4f}) to {args.out}")

    print(f"\nDone. Best mean AUC: {best_auc:.4f}")


if __name__ == "__main__":
    main()
