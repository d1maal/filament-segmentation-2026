import csv

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.metrics import aggregate, panoptic_quality

from .config import config
from .dataset import FilamentMaskRCNNDataset
from .infer import predict_masks
from .model import build_model
from .utils import collate_fn, get_device, save_json


@torch.no_grad()
def evaluate(model, dataset, device, limit=0):
    model.eval()

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    rows = []

    for i, (images, targets) in enumerate(tqdm(loader, desc="val")):
        if limit and i >= limit:
            break

        pred = predict_masks(model, images[0], device)
        gt = [
            mask.numpy().astype("uint8")
            for mask in targets[0]["masks"]
        ]

        rows.append(panoptic_quality(gt, pred))

    return aggregate(rows)


def train(
    epochs=None,
    image_size=None,
    lr=None,
    eval_limit=0,
):
    epochs = int(epochs or config.epochs)
    image_size = int(image_size or config.image_size)
    lr = float(lr or config.lr)

    device = get_device()
    print(f"device: {device}")

    train_ds = FilamentMaskRCNNDataset(train=True, augment=True)
    val_ds = FilamentMaskRCNNDataset(train=False, augment=False)

    print(f"train images: {len(train_ds)}")
    print(f"val images: {len(val_ds)}")

    if len(train_ds) == 0:
        raise RuntimeError("Mask R-CNN training split is empty")
    if len(val_ds) == 0:
        raise RuntimeError("Mask R-CNN validation split is empty")

    loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )

    model = build_model(image_size=image_size).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=config.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, epochs),
        eta_min=lr * 0.05,
    )

    out = config.output_dir
    weights_dir = out / "weights"
    out.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(parents=True, exist_ok=True)

    best_pq = -1.0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        steps = 0

        for images, targets in tqdm(
            loader, desc=f"epoch {epoch}/{epochs}"
        ):
            images = [image.to(device) for image in images]
            targets = [
                {
                    key: value.to(device)
                    for key, value in target.items()
                }
                for target in targets
            ]

            losses = model(images, targets)
            loss = sum(losses.values())

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.detach().cpu())
            steps += 1

        scheduler.step()

        avg_loss = running_loss / max(steps, 1)
        metrics = evaluate(
            model,
            val_ds,
            device,
            limit=eval_limit,
        )

        row = {
            "epoch": epoch,
            "train_loss": avg_loss,
            "lr": optimizer.param_groups[0]["lr"],
            **metrics,
        }
        history.append(row)

        checkpoint = {
            "model": model.state_dict(),
            "epoch": epoch,
            "image_size": image_size,
        }

        torch.save(checkpoint, out / "last.pt")
        torch.save(
            checkpoint,
            weights_dir / f"epoch_{epoch:02d}.pt",
        )

        pq = float(metrics.get("pq", 0.0))
        if pq > best_pq:
            best_pq = pq
            torch.save(checkpoint, out / "best.pt")

        save_json(out / "train_log.json", history)

        print(
            f"epoch {epoch}: "
            f"loss={avg_loss:.4f} "
            f"PQ={pq:.4f} "
            f"TP={metrics.get('tp', 0)} "
            f"FP={metrics.get('fp', 0)} "
            f"FN={metrics.get('fn', 0)}"
        )

    if history:
        with (out / "train_log.csv").open(
            "w", newline="", encoding="utf-8"
        ) as f:
            writer = csv.DictWriter(
                f, fieldnames=list(history[0].keys())
            )
            writer.writeheader()
            writer.writerows(history)

    return out / "best.pt"
