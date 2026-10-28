from __future__ import annotations

import csv
import logging
import math
import random
import time
from pathlib import Path

import cv2
import mlflow
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import config
from src.data import image_stem, instances_for_image
from src.io.overlay import _draw_overlay
from src.models.edgeattn.build import build_arch, count_params
from src.models.edgeattn.dataset import (
    DATA_SEED,
    SemanticMagfilo,
    train_items,
    unique_val_items,
)
from src.models.edgeattn.inference import (
    _eval_rows,
    _summary,
    mask_to_labels,
    predict_probability,
)
from src.models.edgeattn.losses import bce_dice_loss
from src.models.edgeattn.pq_fast import (
    instances_from_labels,
    panoptic_quality_from_labels,
)
from src.tracking.dagshub import init_dagshub

PQ_SCORE_THR = 0.5
PQ_MIN_AREA = 80
VIZ_TOP_K = 5


def _mute_loggers() -> None:
    """Приглушает сторонние логгеры до WARNING на время обучения."""
    for name in ("transformers", "PIL", "matplotlib"):
        logging.getLogger(name).setLevel(logging.WARNING)


def _sample(items: list[tuple], limit: int, seed: int) -> list[tuple]:
    """Случайно ограничивает набор элементов.
    params:
        items: исходный список
        limit: сколько оставить (0 — все)
        seed: сид выбора
    returns:
        выбранные элементы
    """
    if limit and len(items) > limit:
        return random.Random(seed).sample(items, limit)
    return items


def _eval_items(
    model: torch.nn.Module,
    checkpoint: dict,
    device: torch.device,
    items: list[tuple],
    tiled: bool,
    tile_size: int,
    tile_overlap: int,
    desc: str,
) -> tuple[float, list[dict]]:
    """Считает Home PQ на наборе изображений.
    params:
        model: модель
        checkpoint: чекпоинт с size/in_ch
        device: устройство
        items: список (coco, info, path)
        tiled: тайловый инференс
        tile_size: сторона тайла
        tile_overlap: перекрытие
        desc: подпись прогресс-бара
    returns:
        PQ и покартинные метрики
    """
    rows = _eval_rows(
        model,
        checkpoint,
        device,
        items,
        PQ_SCORE_THR,
        PQ_MIN_AREA,
        tiled,
        tile_size,
        tile_overlap,
        desc=desc,
        leave=False,
    )
    return float(_summary(rows)["pq"]), rows


def _plot_history(history: dict, path: Path) -> None:
    """Строит и сохраняет графики loss/PQ по эпохам.
    params:
        history: словари train_loss, val_pq, train_pq
        path: куда сохранить PNG
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    ax.plot(
        epochs, history["train_loss"], label="train_loss", marker="o", color="tab:blue"
    )
    ax.set_xlabel("epoch")
    ax.set_ylabel("train_loss")
    ax2 = ax.twinx()
    val = [(e, v) for e, v in zip(epochs, history["val_pq"]) if v is not None]
    train = [(e, v) for e, v in zip(epochs, history["train_pq"]) if v is not None]
    if val:
        ax2.plot(*zip(*val), label="val_pq", marker="s", color="tab:orange")
    if train:
        ax2.plot(*zip(*train), label="train_pq", marker="^", color="tab:green")
    ax2.set_ylabel("pq")
    fig.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _save_viz(
    model: torch.nn.Module,
    checkpoint: dict,
    device: torch.device,
    items: list[tuple],
    rows: list[dict],
    run_dir: Path,
    tiled: bool,
    tile_size: int,
    tile_overlap: int,
) -> None:
    """Сохраняет лучшие и худшие кадры с overlay GT/pred.
    params:
        model: модель
        checkpoint: чекпоинт
        device: устройство
        items: val-элементы
        rows: покартинные метрики с полем stem
        run_dir: папка эксперимента
        tiled: тайловый инференс
        tile_size: сторона тайла
        tile_overlap: перекрытие
    """
    if not rows:
        return
    by_stem = {
        image_stem(info["file_name"]): (coco, info, path) for coco, info, path in items
    }
    worst = sorted(rows, key=lambda row: (row["pq"], -row["fn"], -row["fp"]))[
        :VIZ_TOP_K
    ]
    best = sorted(
        rows, key=lambda row: (row["pq"], row["fn"], row["fp"]), reverse=True
    )[:VIZ_TOP_K]
    saved = 0
    for name, subset in (("bad", worst), ("good", best)):
        folder = run_dir / "viz" / name
        folder.mkdir(parents=True, exist_ok=True)
        for row in subset:
            coco, info, path = by_stem[row["stem"]]
            gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                continue
            prob = predict_probability(
                model,
                gray,
                device,
                size=int(checkpoint.get("size", 512)),
                in_ch=int(checkpoint.get("in_ch", 3)),
                tiled=tiled,
                tile_size=tile_size,
                tile_overlap=tile_overlap,
            )
            labels = mask_to_labels(prob, PQ_SCORE_THR, PQ_MIN_AREA)
            gt = instances_for_image(coco, info["id"])
            fresh = panoptic_quality_from_labels(gt, labels)
            vis = _draw_overlay(gray, gt, instances_from_labels(labels), fresh)
            cv2.imwrite(str(folder / f"pq_{fresh['pq']:.3f}_{row['stem']}.jpg"), vis)
            saved += 1
    if saved:
        print(f"Saved {saved} visualizations to {run_dir / 'viz'}")


def train(
    arch: str = "edgeattn",
    in_ch: int = 3,
    size: int = 512,
    epochs: int = 8,
    batch: int = 8,
    accum: int = 1,
    ema: bool = False,
    ema_decay: float = 0.99,
    lr: float = 1e-4,
    run_name: str = "edgeattn",
    eval_samples_val: int = 0,
    eval_samples_train: int = 0,
    eval_tiled: bool = False,
    cosine: bool = False,
    workers: int = 0,
    max_steps: int = 0,
    seed: int = DATA_SEED,
    base: int = 64,
    blocks: int = 2,
    heads: int = 4,
    dropout: float = 0.1,
    augment: bool = True,
) -> Path:
    """Обучает EdgeAttNet на MAGFiLO и логирует run в DagsHub.
    params:
        arch: вариант архитектуры (unet или edgeattn)
        in_ch: число каналов входного изображения
        size: сторона обучающего кропа
        epochs: число эпох
        batch: размер батча
        accum: число батчей на один шаг оптимизатора
        ema: вести экспоненциальное среднее весов
        ema_decay: коэффициент затухания EMA
        lr: скорость обучения
        run_name: имя MLflow-run
        eval_samples_val: число val-кадров для PQ (0 — все)
        eval_samples_train: число train-кадров для PQ (0 — не считать)
        eval_tiled: считать PQ тайловым инференсом
        cosine: косинус-декай learning rate по шагам оптимизатора
        workers: число воркеров DataLoader
        max_steps: ограничение числа шагов (0 — без ограничения)
        seed: сид обучения
        base: число каналов первой стадии энкодера
        blocks: число EG-MHSA блоков
        heads: число голов attention
        dropout: dropout в EG-MHSA
        augment: геометрические аугментации
    returns:
        путь к last.pt
    """
    _mute_loggers()
    if accum < 1:
        raise ValueError(f"accum must be >= 1, got {accum}")
    if ema and not 0.0 < ema_decay < 1.0:
        raise ValueError(f"ema_decay must be in (0, 1), got {ema_decay}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = SemanticMagfilo(in_ch=in_ch, size=size, augment=augment)
    loader = DataLoader(
        dataset,
        batch_size=batch,
        shuffle=True,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        generator=torch.Generator().manual_seed(seed),
    )
    model = build_arch(
        arch=arch, in_ch=in_ch, base=base, blocks=blocks, heads=heads, dropout=dropout
    ).to(device)
    n_params = count_params(model)
    ema_state: dict[str, torch.Tensor] | None = None
    ema_model: torch.nn.Module | None = None
    if ema:
        ema_state = {
            name: tensor.detach().clone()
            for name, tensor in model.state_dict().items()
            if tensor.is_floating_point()
        }
        ema_model = build_arch(
            arch=arch,
            in_ch=in_ch,
            base=base,
            blocks=blocks,
            heads=heads,
            dropout=dropout,
        ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    def _update_ema() -> None:
        """Обновляет EMA весов после шага оптимизатора."""
        if ema_state is None:
            return
        with torch.no_grad():
            for name, tensor in model.state_dict().items():
                average = ema_state.get(name)
                if average is not None:
                    average.mul_(ema_decay).add_(tensor.detach(), alpha=1.0 - ema_decay)

    steps_per_epoch = max(1, math.ceil(len(dataset) / batch))
    optimizer_steps = max(1, math.ceil(steps_per_epoch * epochs / accum))
    if max_steps:
        optimizer_steps = min(optimizer_steps, max(1, math.ceil(max_steps / accum)))
    scheduler = None
    if cosine:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, optimizer_steps), eta_min=0.0
        )
    run_dir = config.edgeattn_experiments_path / run_name
    weights_dir = run_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    val_items = unique_val_items()
    eval_val = _sample(val_items, eval_samples_val, seed)
    train_pool = train_items()
    eval_train = (
        _sample(train_pool, eval_samples_train, seed) if eval_samples_train else []
    )
    params = {
        "model": type(model).__name__,
        "arch": arch,
        "in_ch": in_ch,
        "size": size,
        "base": base,
        "blocks": blocks,
        "heads": heads,
        "dropout": dropout,
        "n_params": n_params,
        "epochs": epochs,
        "batch": batch,
        "accum": accum,
        "effective_batch": batch * accum,
        "ema": ema,
        "ema_decay": ema_decay,
        "optimizer": "Adam",
        "lr": lr,
        "dataset": "MAGFiLO 1.0 (Kaggle 2026)",
        "n_train": len(dataset),
        "n_train_images": len(train_pool),
        "n_val": len(val_items),
        "val_split": "splits/val_ids.txt",
        "object_crop_prob": 0.7,
        "negative_crop_prob": 0.1,
        "augment": augment,
        "seed": seed,
        "max_steps": max_steps,
        "scheduler": "cosine" if cosine else "const",
    }
    run = None
    if init_dagshub():
        try:
            run = mlflow.start_run(run_name=run_name)
            mlflow.log_params(params)
        except Exception as exc:  # noqa: BLE001 - обучение не должно падать из-за сети
            print(f"WARNING: MLflow run init failed, continuing without logging: {exc}")
            run = None
    history: dict = {"train_loss": [], "val_pq": [], "train_pq": []}
    rows_csv: list[dict] = []
    best_pq = -1.0
    best_ema_pq = -1.0
    total_steps = 0
    stop = False
    ckpt_path = run_dir / "last.pt"
    try:
        for epoch in range(1, epochs + 1):
            started = time.time()
            model.train()
            epoch_lr = float(optimizer.param_groups[0]["lr"])
            current_lr = epoch_lr
            running, seen = 0.0, 0
            step_in_group = 0
            pbar = tqdm(loader, desc=f"epoch {epoch} [crop]")
            for sample in pbar:
                if step_in_group == 0:
                    optimizer.zero_grad(set_to_none=True)
                images = sample["image"].to(device, non_blocking=True)
                targets = sample["target"].to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits = model(images)
                loss = bce_dice_loss(logits.float(), targets) / accum
                scaler.scale(loss).backward()
                running += float(loss.detach()) * accum
                seen += 1
                total_steps += 1
                step_in_group += 1
                if step_in_group == accum:
                    scaler.step(optimizer)
                    scaler.update()
                    if scheduler is not None:
                        scheduler.step()
                    _update_ema()
                    step_in_group = 0
                current_lr = scheduler.get_last_lr()[0] if scheduler is not None else lr
                pbar.set_postfix(loss=f"{running / seen:.4f}", lr=f"{current_lr:.1e}")
                if max_steps and total_steps >= max_steps:
                    stop = True
                    break
            if step_in_group:
                scaler.step(optimizer)
                scaler.update()
                if scheduler is not None:
                    scheduler.step()
                _update_ema()
            avg_loss = running / max(seen, 1)
            bundle = {
                "model": model.state_dict(),
                "arch": arch,
                "in_ch": in_ch,
                "size": size,
                "base": base,
                "blocks": blocks,
                "heads": heads,
                "dropout": dropout,
                "epoch": epoch,
            }
            torch.save(bundle, ckpt_path)
            torch.save(bundle, weights_dir / f"epoch_{epoch}.pt")
            model.eval()
            val_pq, val_rows = _eval_items(
                model,
                bundle,
                device,
                eval_val,
                eval_tiled,
                size,
                size // 4,
                "eval val PQ",
            )
            val_pq_ema = None
            if ema_state is not None and ema_model is not None:
                live = {
                    name: tensor.detach().clone()
                    for name, tensor in model.state_dict().items()
                }
                live.update(ema_state)
                ema_model.load_state_dict(live)
                ema_bundle = {
                    key: value for key, value in bundle.items() if key != "model"
                }
                ema_bundle["model"] = ema_model.state_dict()
                val_pq_ema, _ = _eval_items(
                    ema_model,
                    ema_bundle,
                    device,
                    eval_val,
                    eval_tiled,
                    size,
                    size // 4,
                    "eval val PQ (EMA)",
                )
                if val_pq_ema > best_ema_pq:
                    best_ema_pq = val_pq_ema
                    torch.save(ema_bundle, run_dir / "best_ema.pt")
            train_pq = None
            if eval_train:
                train_pq, _ = _eval_items(
                    model,
                    bundle,
                    device,
                    eval_train,
                    eval_tiled,
                    size,
                    size // 4,
                    "eval train PQ",
                )
            history["train_loss"].append(avg_loss)
            history["val_pq"].append(val_pq)
            history["train_pq"].append(train_pq)
            if val_pq > best_pq:
                best_pq = val_pq
                torch.save(bundle, run_dir / "best.pt")
            if run is not None:
                metrics = {"train_loss": avg_loss, "val_pq": val_pq, "lr": epoch_lr}
                if train_pq is not None:
                    metrics["train_pq"] = train_pq
                if val_pq_ema is not None:
                    metrics["val_pq_ema"] = val_pq_ema
                mlflow.log_metrics(metrics, step=epoch)
            rows_csv.append(
                {
                    "epoch": epoch,
                    "train_loss": avg_loss,
                    "train_pq": train_pq,
                    "val_pq": val_pq,
                    "lr": epoch_lr,
                    "val_pq_ema": val_pq_ema,
                }
            )
            train_pq_text = f"{train_pq:.4f}" if train_pq is not None else "n/a"
            ema_text = f" val_pq_ema={val_pq_ema:.4f}" if val_pq_ema is not None else ""
            print(
                f"epoch {epoch} loss={avg_loss:.4f} train_pq={train_pq_text} "
                f"val_pq={val_pq:.4f}{ema_text} lr={current_lr:.1e} "
                f"time={time.time() - started:.1f}s"
            )
            model.train()
            if stop:
                break
        csv_path = run_dir / "metrics.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "epoch",
                    "train_loss",
                    "train_pq",
                    "val_pq",
                    "lr",
                    "val_pq_ema",
                ],
            )
            writer.writeheader()
            writer.writerows(rows_csv)
        best_path = run_dir / "best.pt"
        if best_path.exists() and val_items:
            best_bundle = torch.load(best_path, map_location=device, weights_only=False)
            model.load_state_dict(best_bundle["model"])
            model.eval()
            _save_viz(
                model,
                best_bundle,
                device,
                val_items,
                val_rows,
                run_dir,
                eval_tiled,
                size,
                size // 4,
            )
        if run is not None:
            try:
                mlflow.log_artifact(str(csv_path))
                for path in (ckpt_path, run_dir / "best.pt"):
                    if path.exists():
                        mlflow.log_artifact(str(path))
                curves = run_dir / "training_curves.png"
                _plot_history(history, curves)
                mlflow.log_artifact(str(curves))
                viz_dir = run_dir / "viz"
                if viz_dir.exists():
                    mlflow.log_artifacts(str(viz_dir), "viz")
            except Exception as exc:  # noqa: BLE001 - артефакты не должны ронять обучение
                print(f"WARNING: MLflow artifacts failed: {exc}")
    finally:
        if run is not None:
            mlflow.end_run()
    return ckpt_path


def cmd_train(
    arch: str = "edgeattn",
    in_ch: int = 3,
    base: int = 64,
    size: int = 512,
    epochs: int = 8,
    batch: int = 8,
    accum: int = 1,
    ema: bool = False,
    ema_decay: float = 0.99,
    lr: float = 1e-4,
    run_name: str = "edgeattn",
    eval_samples_val: int = 0,
    eval_samples_train: int = 0,
    eval_tiled: bool = False,
    cosine: bool = False,
    workers: int = 0,
    max_steps: int = 0,
) -> None:
    """Запускает обучение EdgeAttNet из CLI.
    params:
        arch: вариант архитектуры (unet или edgeattn)
        in_ch: число каналов входного изображения
        base: число каналов первой стадии энкодера
        size: сторона обучающего кропа
        epochs: число эпох
        batch: размер батча
        accum: число батчей на один шаг оптимизатора
        ema: вести экспоненциальное среднее весов
        ema_decay: коэффициент затухания EMA
        lr: скорость обучения
        run_name: имя MLflow-run
        eval_samples_val: число val-кадров для PQ (0 — все)
        eval_samples_train: число train-кадров для PQ (0 — не считать)
        eval_tiled: считать PQ тайловым инференсом
        cosine: косинус-декай learning rate по шагам оптимизатора
        workers: число воркеров DataLoader
        max_steps: ограничение числа шагов (0 — без ограничения)
    """
    path = train(
        arch=arch,
        in_ch=in_ch,
        base=base,
        size=size,
        epochs=epochs,
        batch=batch,
        accum=accum,
        ema=ema,
        ema_decay=ema_decay,
        lr=lr,
        run_name=run_name,
        eval_samples_val=eval_samples_val,
        eval_samples_train=eval_samples_train,
        eval_tiled=eval_tiled,
        cosine=cosine,
        workers=workers,
        max_steps=max_steps,
    )
    print(f"Training finished: {path}")
