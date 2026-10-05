import csv
from pathlib import Path

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
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=config.num_workers > 0,
        collate_fn=collate_fn,
    )
    rows = []
    for i, (images, targets) in enumerate(tqdm(loader, desc="val")):
        if limit and i >= limit:
            break
        pred = predict_masks(model, images[0], device)
        gt = [mask.numpy().astype("uint8") for mask in targets[0]["masks"]]
        rows.append(panoptic_quality(gt, pred))
    return aggregate(rows)


def train(epochs=None, image_size=None, lr=None, eval_limit=0, resume=None):
    epochs = int(epochs or config.epochs)
    image_size = int(image_size or config.image_size)
    lr = float(lr or config.lr)
    device = get_device()
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    train_ds = FilamentMaskRCNNDataset(train=True, augment=True)
    val_ds = FilamentMaskRCNNDataset(train=False, augment=False)
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError("Empty train or validation split")
    loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=config.num_workers > 0,
        collate_fn=collate_fn,
    )
    architecture = config.architecture
    start_epoch = 1
    checkpoint = None
    if resume:
        try:
            checkpoint = torch.load(resume, map_location=device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(resume, map_location=device)
        architecture = checkpoint.get("architecture", architecture)
        image_size = int(checkpoint.get("image_size", image_size))
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
    model = build_model(image_size=image_size, architecture=architecture).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        epochs=epochs,
        steps_per_epoch=max(1, len(loader)),
        pct_start=0.10,
        anneal_strategy="cos",
        div_factor=10.0,
        final_div_factor=100.0,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=config.amp and device.type == "cuda")
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        if "scaler" in checkpoint and checkpoint["scaler"] is not None:
            scaler.load_state_dict(checkpoint["scaler"])
    out = config.output_dir
    weights_dir = out / "weights"
    out.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(parents=True, exist_ok=True)
    best_pq = -1.0
    history = []
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        loss_sums = {}
        steps = 0
        for images, targets in tqdm(loader, desc=f"epoch {epoch}/{epochs}"):
            images = [image.to(device, non_blocking=True) for image in images]
            targets = [{key: value.to(device, non_blocking=True) for key, value in target.items()} for target in targets]
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=config.amp and device.type == "cuda"):
                losses = model(images, targets)
                loss = sum(losses.values())
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            for key, value in losses.items():
                loss_sums[key] = loss_sums.get(key, 0.0) + float(value.detach().cpu())
            loss_sums["train_loss"] = loss_sums.get("train_loss", 0.0) + float(loss.detach().cpu())
            steps += 1
        metrics = evaluate(model, val_ds, device, limit=eval_limit)
        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            **{key: value / max(steps, 1) for key, value in loss_sums.items()},
            **metrics,
        }
        history.append(row)
        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict() if scaler.is_enabled() else None,
            "epoch": epoch,
            "image_size": image_size,
            "architecture": architecture,
        }
        torch.save(state, out / "last.pt")
        torch.save(state, weights_dir / f"epoch_{epoch:02d}.pt")
        pq = float(metrics.get("pq", 0.0))
        if pq > best_pq:
            best_pq = pq
            torch.save(state, out / "best.pt")
        save_json(out / "train_log.json", history)
        with (out / "train_log.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)
        print(f"epoch {epoch}: loss={row['train_loss']:.4f} PQ={pq:.4f} TP={metrics.get('tp', 0)} FP={metrics.get('fp', 0)} FN={metrics.get('fn', 0)}")
    return out / "best.pt"
