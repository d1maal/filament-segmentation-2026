import csv
import json
import math
import shutil
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
def evaluate(model, dataset, device, limit=0, tta=False, **predict_kwargs):
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
        pred = predict_masks(model, images[0], device, tta=tta, **predict_kwargs)
        gt = [mask.numpy().astype("uint8") for mask in targets[0]["masks"]]
        rows.append(panoptic_quality(gt, pred))
    return aggregate(rows)


def _make_scheduler(optimizer, lr, epochs, steps_per_epoch):
    total_steps = max(1, int(epochs) * int(steps_per_epoch))
    warmup_steps = max(1, int(config.warmup_epochs) * int(steps_per_epoch))
    min_factor = min(1.0, float(config.min_lr) / float(lr))

    def schedule(step):
        if step < warmup_steps:
            progress = float(step + 1) / float(warmup_steps)
            return 0.10 + 0.90 * progress
        remaining = max(1, total_steps - warmup_steps)
        progress = min(1.0, float(step - warmup_steps) / float(remaining))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_factor + (1.0 - min_factor) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)


def _full_checkpoint(model, optimizer, scheduler, scaler, epoch, image_size, architecture):
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict() if scaler.is_enabled() else None,
        "epoch": int(epoch),
        "image_size": int(image_size),
        "architecture": architecture,
    }


def _model_checkpoint(model, epoch, image_size, architecture, metrics=None):
    return {
        "model": model.state_dict(),
        "epoch": int(epoch),
        "image_size": int(image_size),
        "architecture": architecture,
        "metrics": dict(metrics or {}),
    }


def _load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _load_compatible_weights(model, path, device):
    checkpoint = _load_checkpoint(path, device)
    source = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    current = model.state_dict()
    compatible = {
        key: value
        for key, value in source.items()
        if key in current and current[key].shape == value.shape
    }
    model.load_state_dict(compatible, strict=False)
    return len(compatible), len(current) - len(compatible)


def _average_model_states(paths, device="cpu"):
    checkpoints = [_load_checkpoint(path, device) for path in paths]
    states = [ckpt["model"] for ckpt in checkpoints]
    averaged = {}
    for key in states[0]:
        values = [state[key] for state in states]
        if torch.is_floating_point(values[0]):
            stacked = torch.stack([value.float() for value in values], dim=0)
            averaged[key] = stacked.mean(dim=0).to(values[0].dtype)
        else:
            averaged[key] = values[-1]
    state = dict(checkpoints[-1])
    state["model"] = averaged
    state["averaged_from"] = [str(path) for path in paths]
    return state


def _prune_epoch_checkpoints(weights_dir: Path, keep_last: int):
    paths = sorted(weights_dir.glob("epoch_*.pt"))
    keep_last = max(1, int(keep_last))
    for path in paths[:-keep_last]:
        path.unlink(missing_ok=True)
    return sorted(weights_dir.glob("epoch_*.pt"))


def _load_history(path: Path):
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def train(
    epochs=None,
    image_size=None,
    lr=None,
    eval_limit=50,
    resume=None,
    init_from=None,
):
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
        checkpoint = _load_checkpoint(resume, device)
        architecture = checkpoint.get("architecture", architecture)
        image_size = int(checkpoint.get("image_size", image_size))
        start_epoch = int(checkpoint.get("epoch", 0)) + 1

    model = build_model(image_size=image_size, architecture=architecture).to(device)

    if init_from and not resume:
        loaded, skipped = _load_compatible_weights(model, init_from, device)
        print(f"initialized from {init_from}: loaded={loaded}, skipped={skipped}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=config.weight_decay,
    )
    scheduler = _make_scheduler(
        optimizer,
        lr=lr,
        epochs=epochs,
        steps_per_epoch=max(1, len(loader)),
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
    _prune_epoch_checkpoints(weights_dir, config.checkpoint_keep_last)

    history = _load_history(out / "train_log.json") if resume else []
    if resume:
        history = [row for row in history if int(row.get("epoch", 0)) < start_epoch]

    best_quick_pq = max((float(row.get("pq", -1.0)) for row in history), default=-1.0)
    full_values = [row.get("full_pq") for row in history if row.get("full_pq") is not None]
    best_full_pq = max((float(x) for x in full_values), default=-1.0)

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        loss_sums = {}
        steps = 0

        for images, targets in tqdm(loader, desc=f"epoch {epoch}/{epochs}"):
            images = [image.to(device, non_blocking=True) for image in images]
            targets = [
                {key: value.to(device, non_blocking=True) for key, value in target.items()}
                for target in targets
            ]
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

        quick_metrics = evaluate(model, val_ds, device, limit=eval_limit)
        do_full = eval_limit == 0 or (
            epoch >= int(config.full_eval_start)
            and (
                epoch == epochs
                or (epoch - int(config.full_eval_start)) % int(config.full_eval_every) == 0
            )
        )
        full_metrics = quick_metrics if eval_limit == 0 else None
        if do_full and eval_limit != 0:
            full_metrics = evaluate(model, val_ds, device, limit=0)

        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            **{key: value / max(steps, 1) for key, value in loss_sums.items()},
            **quick_metrics,
            "full_pq": None if full_metrics is None else full_metrics.get("pq", 0.0),
            "full_tp": None if full_metrics is None else full_metrics.get("tp", 0),
            "full_fp": None if full_metrics is None else full_metrics.get("fp", 0),
            "full_fn": None if full_metrics is None else full_metrics.get("fn", 0),
            "full_splits": None if full_metrics is None else full_metrics.get("splits", 0),
            "full_merges": None if full_metrics is None else full_metrics.get("merges", 0),
            "full_images": None if full_metrics is None else full_metrics.get("images", 0),
        }
        history.append(row)

        full_state = _full_checkpoint(
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            image_size,
            architecture,
        )
        torch.save(full_state, out / "last.pt")

        model_state = _model_checkpoint(
            model,
            epoch,
            image_size,
            architecture,
            metrics=full_metrics or quick_metrics,
        )
        epoch_path = weights_dir / f"epoch_{epoch:02d}.pt"
        torch.save(model_state, epoch_path)
        _prune_epoch_checkpoints(weights_dir, config.checkpoint_keep_last)

        quick_pq = float(quick_metrics.get("pq", 0.0))
        if quick_pq > best_quick_pq:
            best_quick_pq = quick_pq
            torch.save(model_state, out / "best_quick.pt")
            if best_full_pq < 0:
                torch.save(model_state, out / "best.pt")

        if full_metrics is not None:
            full_pq = float(full_metrics.get("pq", 0.0))
            if full_pq > best_full_pq:
                best_full_pq = full_pq
                torch.save(model_state, out / "best_full.pt")
                torch.save(model_state, out / "best.pt")

        save_json(out / "train_log.json", history)
        with (out / "train_log.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)

        message = (
            f"epoch {epoch}: loss={row['train_loss']:.4f} "
            f"quick_PQ={quick_pq:.4f} TP={quick_metrics.get('tp', 0)} "
            f"FP={quick_metrics.get('fp', 0)} FN={quick_metrics.get('fn', 0)}"
        )
        if full_metrics is not None:
            message += f" full_PQ={float(full_metrics.get('pq', 0.0)):.4f}"
        print(message)

    retained = _prune_epoch_checkpoints(weights_dir, config.checkpoint_keep_last)
    n = min(int(config.swa_last_n), len(retained))
    if n >= 2:
        swa_paths = retained[-n:]
        swa_state = _average_model_states(swa_paths, device="cpu")
        swa_path = out / f"average_last{n}.pt"
        torch.save(swa_state, swa_path)
        model.load_state_dict(swa_state["model"])
        swa_metrics = evaluate(model, val_ds, device, limit=0)
        save_json(out / f"average_last{n}_metrics.json", swa_metrics)
        swa_pq = float(swa_metrics.get("pq", 0.0))
        print(
            f"average_last{n}: PQ={swa_pq:.4f} "
            f"TP={swa_metrics.get('tp', 0)} FP={swa_metrics.get('fp', 0)} "
            f"FN={swa_metrics.get('fn', 0)}"
        )
        if swa_pq > best_full_pq:
            shutil.copy2(swa_path, out / "best.pt")
            shutil.copy2(swa_path, out / "best_full.pt")

    return out / "best.pt"
