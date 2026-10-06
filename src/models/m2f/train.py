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
from src.io.visualization import _save_best_worst
from src.metrics import PQ_IOU_THR, aggregate, panoptic_quality
from src.models.m2f.build import CHECKPOINTS, build
from src.models.m2f.dataset import DATA_SEED, Magfilo, collate
from src.models.m2f.ema import ModelEMA
from src.models.m2f.inference import predict_masks
from src.tracking.dagshub import _log_training_artifacts, init_dagshub

MAX_STEPS = 0
ACCUM = config.train.accum
LR = 3e-4
GRAD_CLIP_NORM = 0.03
EMA_DECAY = 0.996
BACKBONE_LR_MULT = 0.3
MIN_LR_RATIO = 0.0


@torch.no_grad()
def _eval_val(model, processor, device, eval_samples: int = 0) -> dict:
    """Считает Home PQ модели на валидационных изображениях.
    params:
        model: обученная модель Mask2Former
        processor: процессор Mask2Former
        device: устройство выполнения вычислений
        eval_samples: число случайных изображений для оценки (0 — все)
    returns:
        агрегированные метрики по валидации (pq, sq, rq, tp, fp, fn, ...)
    """
    items = _val_items()
    if eval_samples and len(items) > eval_samples:
        items = random.Random(DATA_SEED).sample(items, eval_samples)
    rows = []
    for coco, info, path in tqdm(items, desc="eval val PQ"):
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        pred = predict_masks(model, processor, img, device)
        rows.append(panoptic_quality(instances_for_image(coco, info["id"]), pred))
    return aggregate(rows)


@torch.no_grad()
def _eval_train(model, processor, device, eval_samples: int = 0) -> dict:
    """Считает Home PQ модели на тренировочных изображениях.
    params:
        model: обученная модель Mask2Former
        processor: процессор Mask2Former
        device: устройство выполнения вычислений
        eval_samples: число случайных изображений для оценки (0 — не считать)
    returns:
        агрегированные метрики по трейну (pq, sq, rq, tp, fp, fn, ...)
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
        pred = predict_masks(model, processor, img, device)
        rows.append(panoptic_quality(instances_for_image(coco, info["id"]), pred))
    return aggregate(rows)


def _make_scheduler(
    optimizer,
    updates_per_epoch: int,
    epochs: int,
    warmup_epochs: int,
    min_lr_ratio: float = MIN_LR_RATIO,
):
    """Warmup и cosine decay по шагам оптимизатора.
    params:
        optimizer: AdamW
        updates_per_epoch: число step() за эпоху
        epochs: число эпох
        warmup_epochs: число эпох линейного разогрева
        min_lr_ratio: нижняя граница lr как доля от пика (0 — затухание до нуля)
    returns:
        LambdaLR
    """
    total_updates = max(1, updates_per_epoch * epochs)
    warmup_updates = max(0, updates_per_epoch * warmup_epochs)
    floor = min(1.0, max(0.0, min_lr_ratio))

    def lr_lambda(step: int) -> float:
        if warmup_updates and step < warmup_updates:
            return max(1e-3, float(step + 1) / warmup_updates)
        if total_updates <= warmup_updates:
            return 1.0
        progress = min(1.0, (step - warmup_updates) / (total_updates - warmup_updates))
        return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def log_train(
    size: int = 512,
    epochs: int = 8,
    max_steps: int = 0,
    accum: int = ACCUM,
    lr: float = LR,
    workers: int = 0,
    eval_val_samples: int = 0,
    eval_train_samples: int = 0,
    run_name: str = "m2f",
    augment: bool = True,
    backbone: str = "tiny",
    object_crop_prob: float = 0.7,
    negative_crop_prob: float = 0.1,
    full_image_epochs: int = 0,
    no_object_weight: float = 0.08,
    dice_weight: float = 5.0,
    mask_weight: float = 5.0,
    warmup_epochs: int = 2,
    min_lr_ratio: float = MIN_LR_RATIO,
    backbone_lr_mult: float = BACKBONE_LR_MULT,
    grad_clip_norm: float = GRAD_CLIP_NORM,
    ema_decay: float = EMA_DECAY,
    grad_checkpoint: bool = False,
) -> dict:
    """Обучает Mask2Former на MAGFiLO и логирует эксперимент в DagsHub.
    params:
        size: сторона кропа при обучении
        epochs: число эпох
        max_steps: ограничение по числу шагов (0 — без ограничения)
        accum: число шагов накопления градиента
        lr: скорость обучения
        workers: число воркеров DataLoader
        eval_val_samples: число случайных изображений для val PQ (0 — вся валидация)
        eval_train_samples: число случайных изображений для train PQ (0 — не считать)
        run_name: имя MLflow run на DagsHub
        augment: геометрические аугментации
        backbone: tiny, small, base, base-in21k или large
        object_crop_prob: доля кропов вокруг филамента
        negative_crop_prob: доля пустых кропов
        full_image_epochs: сколько последних эпох учить на полном диске
        no_object_weight: вес класса «нет объекта»
        dice_weight: вес Dice-лосса
        mask_weight: вес mask-лосса
        warmup_epochs: эпохи линейного разогрева lr
        min_lr_ratio: нижняя граница lr как доля от пика (0 — затухание до нуля)
        backbone_lr_mult: множитель lr для энкодера
        grad_clip_norm: предел нормы градиента перед шагом оптимизатора
        ema_decay: коэффициент затухания EMA
        grad_checkpoint: gradient checkpointing для экономии памяти
    returns:
        словарь с историей train_loss, train_pq и val_pq по эпохам
    """
    if not config.train_annotations.exists():
        raise SystemExit(
            f"No data at {config.train_annotations}. On Kaggle: Add Data → filament-segmentation-2026"
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor, model = build(
        size=size,
        backbone=backbone,
        no_object_weight=no_object_weight,
        dice_weight=dice_weight,
        mask_weight=mask_weight,
        grad_checkpoint=grad_checkpoint,
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
        collate_fn=lambda b: collate(processor, b),
    )
    backbone_params = list(model.model.pixel_level_module.encoder.parameters())
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
        opt,
        max(1, math.ceil(len(loader) / max(1, accum))),
        epochs,
        warmup_epochs,
        min_lr_ratio,
    )
    ema = ModelEMA(model, decay=ema_decay)
    try:
        scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
        autocast = lambda: torch.amp.autocast("cuda", enabled=device.type == "cuda")
    except (TypeError, AttributeError):
        scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
        autocast = lambda: torch.cuda.amp.autocast(enabled=device.type == "cuda")
    out = config.models_path
    out.mkdir(parents=True, exist_ok=True)
    run_dir = out / "experiments" / run_name
    weights_dir = run_dir / "weights"
    run_dir.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(parents=True, exist_ok=True)
    extra = {
        "backbone": backbone,
        "no_object_weight": no_object_weight,
        "dice_weight": dice_weight,
        "mask_weight": mask_weight,
        "object_crop_prob": object_crop_prob,
        "negative_crop_prob": negative_crop_prob,
        "full_image_epochs": full_image_epochs,
        "warmup_epochs": warmup_epochs,
        "min_lr_ratio": min_lr_ratio,
        "backbone_lr_mult": backbone_lr_mult,
        "lr": lr,
        "augment": augment,
        "ema_decay": ema_decay,
        "grad_clip_norm": grad_clip_norm,
        "grad_checkpoint": grad_checkpoint,
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
                    "pq_iou_thr": PQ_IOU_THR,
                    **extra,
                }
            )
        except Exception as exc:
            print(f"WARNING: MLflow run init failed, continuing without logging: {exc}")
            run = None
    step_i = 0
    ckpt = out / "last.pt"
    for epoch in range(1, epochs + 1):
        use_full = full_image_epochs > 0 and epoch > epochs - full_image_epochs
        data.set_full_image(use_full)
        mode = "full" if use_full else "crop"
        running, seen, step = 0.0, 0, 0
        opt.zero_grad()
        for step, batch in enumerate(tqdm(loader, desc=f"epoch {epoch} [{mode}]"), start=1):
            with autocast():
                loss = (
                    model(
                        pixel_values=batch["pixel_values"].to(device),
                        mask_labels=[m.to(device) for m in batch["mask_labels"]],
                        class_labels=[c.to(device) for c in batch["class_labels"]],
                    ).loss
                    / accum
                )
            scaler.scale(loss).backward()
            if step % accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
                scheduler.step()
                ema.update(model)
            running += float(loss.detach().cpu()) * accum
            seen += 1
            step_i += 1
            if max_steps and step_i >= max_steps:
                break
        if seen and step % accum:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad()
            scheduler.step()
            ema.update(model)
        bundle = {"model": model.state_dict(), "size": size, "epoch": epoch, **extra}
        torch.save(bundle, ckpt)
        torch.save(bundle, weights_dir / f"epoch_{epoch}.pt")
        avg_loss = running / max(seen, 1)
        history["train_loss"].append(avg_loss)
        model.eval()
        ema.apply_to(model)
        val = _eval_val(model, processor, device, eval_val_samples)
        train = (
            _eval_train(model, processor, device, eval_train_samples) if eval_train_samples else None
        )
        ema.restore(model)
        model.train()
        val_pq = float(val["pq"])
        train_pq = float(train["pq"]) if train is not None else None
        history["val_pq"].append(val_pq)
        history["train_pq"].append(train_pq)
        if run is not None:
            metrics = {
                "train_loss": avg_loss,
                "val_pq_ema": val_pq,
                "val_tp_ema": val["tp"],
                "val_fp_ema": val["fp"],
                "val_fn_ema": val["fn"],
                "val_sq_ema": val["sq"],
                "val_rq_ema": val["rq"],
                "val_precision_ema": val["precision"],
                "val_recall_ema": val["recall"],
                "lr": opt.param_groups[1]["lr"],
                "lr_backbone": opt.param_groups[0]["lr"],
            }
            if train is not None:
                metrics.update(
                    {
                        "train_pq_ema": train_pq,
                        "train_tp_ema": train["tp"],
                        "train_fp_ema": train["fp"],
                        "train_fn_ema": train["fn"],
                        "train_sq_ema": train["sq"],
                        "train_rq_ema": train["rq"],
                        "train_precision_ema": train["precision"],
                        "train_recall_ema": train["recall"],
                    }
                )
            mlflow.log_metrics(metrics, step=epoch)
        metrics_rows.append(
            {
                "epoch": epoch,
                "train_loss": avg_loss,
                "train_pq_ema": train_pq,
                "val_pq_ema": val_pq,
                "val_tp_ema": val["tp"],
                "val_fp_ema": val["fp"],
                "val_fn_ema": val["fn"],
                "val_sq_ema": val["sq"],
                "val_rq_ema": val["rq"],
                "train_tp_ema": train["tp"] if train is not None else None,
                "train_fp_ema": train["fp"] if train is not None else None,
                "train_fn_ema": train["fn"] if train is not None else None,
                "train_sq_ema": train["sq"] if train is not None else None,
                "train_rq_ema": train["rq"] if train is not None else None,
            }
        )
        if val_pq > best_pq:
            best_pq = val_pq
            torch.save({**bundle, "model": ema.shadow}, run_dir / "best.pt")
        print(
            f"epoch {epoch} mode={mode} loss={avg_loss:.4f} "
            f"lr={opt.param_groups[1]['lr']:.2e} -> {ckpt}"
        )
        if max_steps and step_i >= max_steps:
            break
    with (run_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "train_loss",
                "train_pq_ema",
                "val_pq_ema",
                "val_tp_ema",
                "val_fp_ema",
                "val_fn_ema",
                "val_sq_ema",
                "val_rq_ema",
                "train_tp_ema",
                "train_fp_ema",
                "train_fn_ema",
                "train_sq_ema",
                "train_rq_ema",
            ],
        )
        w.writeheader()
        w.writerows(metrics_rows)
    processor.save_pretrained(out / "processor")
    best_ckpt = run_dir / "best.pt"
    if best_ckpt.exists():
        model.load_state_dict(
            torch.load(best_ckpt, map_location=device, weights_only=False)["model"]
        )
        model.eval()
        _save_best_worst(model, processor, device, run_dir)
    if run is not None:
        _log_training_artifacts(model, ckpt, history)
        mlflow.end_run()
    return history


def train(
    size: int = 512,
    epochs: int = 8,
    max_steps: int = 0,
    accum: int = ACCUM,
    lr: float = LR,
    workers: int = 0,
    eval_val_samples: int = 0,
    eval_train_samples: int = 0,
    run_name: str = "m2f",
    augment: bool = True,
    backbone: str = "small",
    object_crop_prob: float = 0.7,
    negative_crop_prob: float = 0.1,
    full_image_epochs: int = 0,
    no_object_weight: float = 0.08,
    dice_weight: float = 5.0,
    mask_weight: float = 5.0,
    warmup_epochs: int = 2,
    min_lr_ratio: float = MIN_LR_RATIO,
    backbone_lr_mult: float = BACKBONE_LR_MULT,
    grad_clip_norm: float = GRAD_CLIP_NORM,
    ema_decay: float = EMA_DECAY,
    grad_checkpoint: bool = False,
) -> Path:
    """Обучает Mask2Former на MAGFiLO через log_train и возвращает путь к last.pt.
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
        object_crop_prob=object_crop_prob,
        negative_crop_prob=negative_crop_prob,
        full_image_epochs=full_image_epochs,
        no_object_weight=no_object_weight,
        dice_weight=dice_weight,
        mask_weight=mask_weight,
        warmup_epochs=warmup_epochs,
        min_lr_ratio=min_lr_ratio,
        backbone_lr_mult=backbone_lr_mult,
        grad_clip_norm=grad_clip_norm,
        ema_decay=ema_decay,
        grad_checkpoint=grad_checkpoint,
    )
    return config.models_path / "last.pt"


def cmd_train(
    smoke: bool,
    size: int,
    epochs: int,
    eval_val_samples: int,
    eval_train_samples: int,
    run_name: str,
    backbone: str,
    full_image_epochs: int,
    warmup_epochs: int,
    lr: float,
    min_lr_ratio: float = MIN_LR_RATIO,
    backbone_lr_mult: float = BACKBONE_LR_MULT,
    grad_clip_norm: float = GRAD_CLIP_NORM,
    ema_decay: float = EMA_DECAY,
    grad_checkpoint: bool = False,
) -> None:
    """Запускает обучение Mask2Former; в smoke-режиме — 1 эпоха и до 30 шагов.
    params:
        smoke: короткий прогон
        size: сторона кропа
        epochs: число эпох
        eval_val_samples: сколько val-картинок для PQ после эпохи (0 — все)
        eval_train_samples: сколько train-картинок для PQ (0 — не считать)
        run_name: имя run на DagsHub
        backbone: tiny, small, base, base-in21k или large
        full_image_epochs: сколько последних эпох на полном диске
        warmup_epochs: эпохи разогрева lr
        lr: скорость обучения
        min_lr_ratio: нижняя граница lr как доля от пика (0 — затухание до нуля)
        backbone_lr_mult: множитель lr энкодера
        grad_clip_norm: предел нормы градиента
        ema_decay: коэффициент затухания EMA
        grad_checkpoint: gradient checkpointing для экономии памяти
    """
    max_steps = MAX_STEPS
    if smoke:
        epochs, max_steps = 1, max_steps or 30
    train(
        size=size,
        epochs=epochs,
        max_steps=max_steps,
        accum=ACCUM,
        eval_val_samples=eval_val_samples,
        eval_train_samples=eval_train_samples,
        run_name=run_name,
        augment=config.train.augment,
        backbone=backbone,
        object_crop_prob=config.train.object_crop_prob,
        negative_crop_prob=config.train.negative_crop_prob,
        full_image_epochs=full_image_epochs,
        no_object_weight=config.train.no_object_weight,
        dice_weight=config.train.dice_weight,
        mask_weight=config.train.mask_weight,
        warmup_epochs=warmup_epochs,
        lr=lr,
        min_lr_ratio=min_lr_ratio,
        backbone_lr_mult=backbone_lr_mult,
        grad_clip_norm=grad_clip_norm,
        ema_decay=ema_decay,
        grad_checkpoint=grad_checkpoint,
    )