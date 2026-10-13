"""Обучение Swin-модели с валидацией Home PQ и MLflow-трекингом."""
from __future__ import annotations

import csv
import math
import random
from pathlib import Path

import cv2
import mlflow
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import config
from src.data import (
    ensure_val_split,
    image_stem,
    instances_for_image,
    load_coco,
    resolve_image,
)
from src.eval.val_items import _val_items
from src.metrics import aggregate, panoptic_quality
from src.models.m2f.dataset import DATA_SEED, Magfilo
from src.models.mask_dino_swin.build import CHECKPOINTS, OUT_DIR, build
from src.models.mask_dino_swin.dataset import collate
from src.models.mask_dino_swin.inference import predict_masks
from src.models.mask_dino_swin.model import match_and_loss
from src.tracking.dagshub import _log_training_artifacts, init_dagshub

MAX_STEPS = 0


@torch.no_grad()
def _eval_val(model, device, forward_size: int, eval_samples: int = 0) -> float:
    """Считает Home PQ модели на валидационных изображениях.
    params:
        model: обученная модель SwinMaskDINO
        device: устройство выполнения вычислений
        forward_size: сторона входа сети при инференсе
        eval_samples: число случайных изображений для оценки (0 — все)
    returns:
        агрегированный PQ по валидации (0, если нет изображений)
    """
    items = _val_items()
    if eval_samples and len(items) > eval_samples:
        items = random.Random(DATA_SEED).sample(items, eval_samples)
    rows = []
    for coco, info, path in tqdm(items, desc="eval val PQ"):
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        pred = predict_masks(model, device, img, forward_size=forward_size)
        rows.append(panoptic_quality(instances_for_image(coco, info["id"]), pred))
    if not rows:
        return 0.0
    return float(aggregate(rows)["pq"])


@torch.no_grad()
def _eval_train(
    model, device, forward_size: int, eval_samples: int = 0
) -> float:
    """Считает Home PQ модели на тренировочных изображениях.
    params:
        model: обученная модель SwinMaskDINO
        device: устройство выполнения вычислений
        forward_size: сторона входа сети при инференсе
        eval_samples: число случайных изображений для оценки (0 — не считать)
    returns:
        агрегированный PQ по трейну (0, если нет изображений)
    """
    coco = load_coco()
    val = ensure_val_split(coco)
    items = []
    for info in coco.imgs.values():
        if image_stem(info["file_name"]) in val:
            continue
        path = resolve_image(config.train_images, info["file_name"])
        if path is not None:
            items.append((coco, info, path))
    if eval_samples and len(items) > eval_samples:
        items = random.Random(DATA_SEED).sample(items, eval_samples)
    rows = []
    for coco, info, path in tqdm(items, desc="eval train PQ"):
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        pred = predict_masks(model, device, img, forward_size=forward_size)
        rows.append(panoptic_quality(instances_for_image(coco, info["id"]), pred))
    if not rows:
        return 0.0
    return float(aggregate(rows)["pq"])


def _make_scheduler(optimizer, updates_per_epoch: int, epochs: int, warmup_epochs: int):
    """Warmup и cosine decay по шагам оптимизатора.
    params:
        optimizer: AdamW
        updates_per_epoch: число step() за эпоху
        epochs: число эпох
        warmup_epochs: число эпох линейного разогрева
    returns:
        LambdaLR
    """
    total_updates = max(1, updates_per_epoch * epochs)
    warmup_updates = max(0, updates_per_epoch * warmup_epochs)

    def lr_lambda(step: int) -> float:
        if warmup_updates and step < warmup_updates:
            return max(1e-3, float(step + 1) / warmup_updates)
        if total_updates <= warmup_updates:
            return 1.0
        progress = min(1.0, (step - warmup_updates) / (total_updates - warmup_updates))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def log_train(
    size: int = 512,
    epochs: int = 8,
    max_steps: int = 0,
    accum: int = 8,
    lr: float = 1e-4,
    workers: int = 0,
    eval_val_samples: int = 0,
    eval_train_samples: int = 0,
    run_name: str = "mask_dino_swin",
    augment: bool = True,
    backbone: str = "tiny",
    num_queries: int = 100,
    hidden_dim: int = 256,
    dec_layers: int = 3,
    object_crop_prob: float = 0.7,
    negative_crop_prob: float = 0.1,
    warmup_epochs: int = 2,
    backbone_lr_mult: float = 0.1,
) -> dict:
    """Обучает SwinMaskDINO на MAGFiLO и логирует эксперимент в DagsHub.
    params:
        size: сторона кропа при обучении (кратна 32)
        epochs: число эпох
        max_steps: ограничение по числу шагов (0 — без ограничения)
        accum: число шагов накопления градиента
        lr: скорость обучения
        workers: число воркеров DataLoader
        eval_val_samples: число случайных изображений для val PQ (0 — вся валидация)
        eval_train_samples: число случайных изображений для train PQ (0 — не считать)
        run_name: имя MLflow run на DagsHub
        augment: геометрические аугментации
        backbone: tiny или small
        num_queries: число queries
        hidden_dim: ширина трансформера
        dec_layers: число слоёв декодера
        object_crop_prob: доля кропов вокруг филамента
        negative_crop_prob: доля пустых кропов
        warmup_epochs: эпохи линейного разогрева lr
        backbone_lr_mult: множитель lr для энкодера
    returns:
        словарь с историей train_loss, train_pq и val_pq по эпохам
    """
    if not config.train_annotations.exists():
        raise SystemExit(
            f"No data at {config.train_annotations}. On Kaggle: Add Data → filament-segmentation-2026"
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build(
        size=size,
        backbone=backbone,
        num_queries=num_queries,
        hidden_dim=hidden_dim,
        dec_layers=dec_layers,
    )
    model.to(device).train()
    data = Magfilo(
        size=size,
        object_crop_prob=object_crop_prob,
        negative_crop_prob=negative_crop_prob,
        augment=augment,
    )
    if not data:
        raise SystemExit("Empty train split")
    loader = DataLoader(
        data,
        batch_size=1,
        shuffle=True,
        num_workers=workers,
        collate_fn=lambda b: collate(size, b),
    )
    backbone_params = list(model.backbone.parameters())
    backbone_ids = {id(p) for p in backbone_params}
    other_params = [p for p in model.parameters() if id(p) not in backbone_ids]
    opt = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": lr * backbone_lr_mult},
            {"params": other_params, "lr": lr},
        ],
        weight_decay=0.05,
    )
    scheduler = _make_scheduler(
        opt, max(1, math.ceil(len(loader) / max(1, accum))), epochs, warmup_epochs
    )
    try:
        scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
        autocast = lambda: torch.amp.autocast("cuda", enabled=device.type == "cuda")
    except (TypeError, AttributeError):
        scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
        autocast = lambda: torch.cuda.amp.autocast(enabled=device.type == "cuda")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    run_dir = OUT_DIR / "experiments" / run_name
    weights_dir = run_dir / "weights"
    run_dir.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(parents=True, exist_ok=True)
    extra = {
        "backbone": backbone,
        "num_queries": num_queries,
        "hidden_dim": hidden_dim,
        "dec_layers": dec_layers,
        "object_crop_prob": object_crop_prob,
        "negative_crop_prob": negative_crop_prob,
        "warmup_epochs": warmup_epochs,
        "backbone_lr_mult": backbone_lr_mult,
        "lr": lr,
        "augment": augment,
    }
    metrics_rows: list[dict] = []
    best_pq = -1.0
    history: dict = {"train_loss": [], "val_pq": [], "train_pq": []}
    run = None
    if init_dagshub():
        try:
            run = mlflow.start_run(run_name=run_name)
            mlflow.log_params(
                {
                    "checkpoint": CHECKPOINTS[backbone],
                    "size": size,
                    "epochs": epochs,
                    "accum": accum,
                    "weight_decay": 0.05,
                    "workers": workers,
                    **extra,
                }
            )
        except Exception as exc:
            print(f"WARNING: MLflow run init failed, continuing without logging: {exc}")
            run = None
    step_i = 0
    ckpt = OUT_DIR / "last.pt"
    for epoch in range(1, epochs + 1):
        running, seen, step = 0.0, 0, 0
        opt.zero_grad()
        for step, batch in enumerate(tqdm(loader, desc=f"epoch {epoch}"), start=1):
            with autocast():
                outputs = model(batch["pixel_values"].to(device))
                gt = batch["mask_labels"][0].to(device)
                loss, _ = match_and_loss(outputs["pred_logits"][0], outputs["pred_masks"][0], gt)
                loss = loss / accum
            scaler.scale(loss).backward()
            if step % accum == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
                scheduler.step()
            running += float(loss.detach().cpu()) * accum
            seen += 1
            step_i += 1
            if max_steps and step_i >= max_steps:
                break
        if seen and step % accum:
            scaler.step(opt)
            scaler.update()
            opt.zero_grad()
            scheduler.step()
        bundle = {"model": model.state_dict(), "size": size, "epoch": epoch, **extra}
        torch.save(bundle, ckpt)
        torch.save(bundle, weights_dir / f"epoch_{epoch}.pt")
        avg_loss = running / max(seen, 1)
        history["train_loss"].append(avg_loss)
        model.eval()
        val_pq = _eval_val(model, device, size, eval_val_samples)
        train_pq = (
            _eval_train(model, device, size, eval_train_samples) if eval_train_samples else None
        )
        model.train()
        history["val_pq"].append(val_pq)
        history["train_pq"].append(train_pq)
        if run is not None:
            metrics = {"train_loss": avg_loss, "val_pq": val_pq}
            if train_pq is not None:
                metrics["train_pq"] = train_pq
            mlflow.log_metrics(metrics, step=epoch)
        metrics_rows.append(
            {"epoch": epoch, "train_loss": avg_loss, "train_pq": train_pq, "val_pq": val_pq}
        )
        if val_pq > best_pq:
            best_pq = val_pq
            torch.save(bundle, run_dir / "best.pt")
        print(
            f"epoch {epoch} loss={avg_loss:.4f} val_pq={val_pq:.4f} "
            f"lr={opt.param_groups[1]['lr']:.2e} -> {ckpt}"
        )
        if max_steps and step_i >= max_steps:
            break
    with (run_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "train_pq", "val_pq"])
        w.writeheader()
        w.writerows(metrics_rows)
    if run is not None:
        _log_training_artifacts(model, ckpt, history, model_name="mask_dino_swin")
        mlflow.end_run()
    return history


def train(
    size: int = 512,
    epochs: int = 8,
    max_steps: int = 0,
    accum: int = 8,
    lr: float = 1e-4,
    workers: int = 0,
    eval_val_samples: int = 0,
    eval_train_samples: int = 0,
    run_name: str = "mask_dino_swin",
    augment: bool = True,
    backbone: str = "tiny",
    num_queries: int = 100,
    hidden_dim: int = 256,
    dec_layers: int = 3,
    object_crop_prob: float = 0.7,
    negative_crop_prob: float = 0.1,
    warmup_epochs: int = 2,
    backbone_lr_mult: float = 0.1,
) -> Path:
    """Обучает SwinMaskDINO на MAGFiLO через log_train и возвращает путь к last.pt.
    params:
        те же, что у log_train
    returns:
        путь к сохранённому чекпоинту last.pt
    """
    log_train(
        size=size,
        epochs=epochs,
        max_steps=max_steps,
        accum=accum,
        lr=lr,
        workers=workers,
        eval_val_samples=eval_val_samples,
        eval_train_samples=eval_train_samples,
        run_name=run_name,
        augment=augment,
        backbone=backbone,
        num_queries=num_queries,
        hidden_dim=hidden_dim,
        dec_layers=dec_layers,
        object_crop_prob=object_crop_prob,
        negative_crop_prob=negative_crop_prob,
        warmup_epochs=warmup_epochs,
        backbone_lr_mult=backbone_lr_mult,
    )
    return OUT_DIR / "last.pt"


def cmd_train(
    smoke: bool,
    size: int,
    epochs: int,
    eval_val_samples: int,
    eval_train_samples: int,
    run_name: str,
    backbone: str,
    num_queries: int,
    lr: float,
    warmup_epochs: int,
    backbone_lr_mult: float,
) -> None:
    """Запускает обучение SwinMaskDINO; в smoke-режиме — 1 эпоха и до 30 шагов.
    params:
        smoke: короткий прогон
        size: сторона кропа (кратна 32)
        epochs: число эпох
        eval_val_samples: сколько val-картинок для PQ после эпохи (0 — все)
        eval_train_samples: сколько train-картинок для PQ (0 — не считать)
        run_name: имя run на DagsHub
        backbone: tiny или small
        num_queries: число queries
        lr: скорость обучения
        warmup_epochs: эпохи разогрева lr
        backbone_lr_mult: множитель lr энкодера
    """
    max_steps = MAX_STEPS
    if smoke:
        epochs, max_steps = 1, max_steps or 30
    train(
        size=size,
        epochs=epochs,
        max_steps=max_steps,
        accum=config.train.accum,
        eval_val_samples=eval_val_samples,
        eval_train_samples=eval_train_samples,
        run_name=run_name,
        augment=config.train.augment,
        backbone=backbone,
        num_queries=num_queries,
        object_crop_prob=config.train.object_crop_prob,
        negative_crop_prob=config.train.negative_crop_prob,
        warmup_epochs=warmup_epochs,
        lr=lr,
        backbone_lr_mult=backbone_lr_mult,
    )
