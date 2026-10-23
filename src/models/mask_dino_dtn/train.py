"""Обучение MaskDINO (ResNet-50) на MAGFiLO с MLflow/DagsHub-логгированием."""

from __future__ import annotations

import copy
import itertools
import math
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mlflow
import torch
from detectron2.config import CfgNode, get_cfg
from detectron2.data import (
    DatasetCatalog,
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.engine import DefaultTrainer, HookBase
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils import comm
from detectron2.utils.env import seed_all_rng
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import setup_logger

from src.config import config as project_config
from src.models.mask_dino_dtn.dataset_mapper import MagfiloThreeChannelDatasetMapper
from src.models.mask_dino_dtn.evaluation import HomePQEvaluator
from src.models.mask_dino_dtn.maskdino import add_maskdino_config
from src.tracking.dagshub import init_dagshub, log_maskdino_artifacts


class Trainer(DefaultTrainer):
    """DefaultTrainer с 3-канальным маппером MAGFiLO."""

    @classmethod
    def build_train_loader(cls, cfg: CfgNode):
        """Собирает train-загрузчик с 3-канальным маппером (LSJ)."""
        mapper = MagfiloThreeChannelDatasetMapper(cfg, is_train=True)
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_test_loader(cls, cfg: CfgNode, dataset_name: str):
        """Собирает test-загрузчик с 3-канальным маппером (только resize)."""
        mapper = MagfiloThreeChannelDatasetMapper(cfg, is_train=False)
        return build_detection_test_loader(cfg, dataset_name, mapper=mapper)

    @classmethod
    def build_evaluator(cls, cfg: CfgNode, dataset_name: str, output_folder: str | None = None):
        """Возвращает эвалюатор датасета.

        params:
            cfg: конфиг detectron2
            dataset_name: имя датасета
            output_folder: папка для артефактов эвалюатора
        returns:
            эвалюатор detectron2
        """
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type == "magfilo":
            return HomePQEvaluator(dataset_name)
        return super().build_evaluator(cfg, dataset_name, output_folder)

    @classmethod
    def build_optimizer(cls, cfg: CfgNode, model):
        """Собирает оптимизатор как в референсе MaskDINO.

        Учитывает BACKBONE_MULTIPLIER и full_model-клиппинг, которые
        стандартный detectron2-оптимизатор не поддерживает.

        params:
            cfg: конфиг detectron2
            model: модель
        returns:
            оптимизатор
        """
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED
        defaults = {
            "lr": cfg.SOLVER.BASE_LR,
            "weight_decay": cfg.SOLVER.WEIGHT_DECAY,
        }
        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )

        params: list[dict] = []
        memo: set = set()
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad or value in memo:
                    continue
                memo.add(value)
                hyperparams = copy.copy(defaults)
                if "backbone" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
                if (
                    "relative_position_bias_table" in module_param_name
                    or "absolute_pos_embed" in module_param_name
                ):
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hyperparams})

        def maybe_add_full_model_gradient_clipping(optim):
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE != "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer


class MLflowLossHook(HookBase):
    """Логирует средний train_loss за эпоху в активный MLflow-ран.

    Attributes:
        iters_per_epoch: число итераций в одной эпохе
    """

    def __init__(self, iters_per_epoch: int) -> None:
        self.iters_per_epoch = max(1, iters_per_epoch)
        self._losses: list[float] = []

    def after_step(self) -> None:
        latest = get_event_storage().latest()
        if "total_loss" in latest:
            value = latest["total_loss"]
            self._losses.append(float(value[1] if isinstance(value, tuple) else value))
        if (self.trainer.iter + 1) % self.iters_per_epoch == 0 and self._losses:
            epoch = (self.trainer.iter + 1) // self.iters_per_epoch
            mlflow.log_metrics({"train_loss": sum(self._losses) / len(self._losses)}, step=epoch)
            self._losses = []


class MLflowEvalHook(HookBase):
    """Логирует Home PQ в MLflow после каждого eval."""

    def __init__(self) -> None:
        self._logged_iter = -1

    def after_step(self) -> None:
        self._maybe_log()

    def after_train(self) -> None:
        self._maybe_log()

    def _maybe_log(self) -> None:
        """Логирует PQ, если с прошлого раза появились новые результаты eval."""
        results = getattr(self.trainer, "_last_eval_results", None)
        if not results or self.trainer.iter == self._logged_iter:
            return
        if "PQ" in results:
            self._logged_iter = self.trainer.iter
            mlflow.log_metrics({"val_pq": float(results["PQ"])}, step=self.trainer.iter)


def _iters_per_epoch(cfg: CfgNode) -> int:
    """Считает число итераций в одной эпохе по размеру train-сплита.

    params:
        cfg: конфиг detectron2
    returns:
        число итераций на эпоху
    """
    n_train = len(DatasetCatalog.get(cfg.DATASETS.TRAIN[0]))
    return max(1, math.ceil(n_train / cfg.SOLVER.IMS_PER_BATCH))


def _build_cfg(
    config: Path,
    smoke: bool,
    size: int,
    epochs: int,
    ims_per_batch: int | None,
    num_workers: int | None,
    run_name: str,
    opts: list[str],
) -> CfgNode:
    """Собирает замороженный CfgNode из YAML и CLI-аргументов.

    params:
        config: путь к YAML-конфигу
        smoke: короткий прогон без eval
        size: INPUT.IMAGE_SIZE
        epochs: число эпох (пересчитывается в SOLVER.MAX_ITER)
        ims_per_batch: SOLVER.IMS_PER_BATCH (None — из конфига)
        num_workers: DATALOADER.NUM_WORKERS (None — из конфига)
        run_name: имя run для OUTPUT_DIR
        opts: дополнительные опции detectron2 (KEY VALUE ...)
    returns:
        замороженный CfgNode
    """
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskdino_config(cfg)
    cfg.merge_from_file(str(config))

    merge_opts: list[str] = ["INPUT.IMAGE_SIZE", str(size)]
    if ims_per_batch is not None:
        merge_opts += ["SOLVER.IMS_PER_BATCH", str(ims_per_batch)]
    if num_workers is not None:
        merge_opts += ["DATALOADER.NUM_WORKERS", str(num_workers)]
    if smoke:
        merge_opts += [
            "SOLVER.MAX_ITER",
            "5",
            "SOLVER.WARMUP_ITERS",
            "0",
            "SOLVER.IMS_PER_BATCH",
            "1",
            "SOLVER.CHECKPOINT_PERIOD",
            "5",
            "DATALOADER.NUM_WORKERS",
            "0",
            "TEST.EVAL_PERIOD",
            "0",
            # smoke не должен прогонять полный val-eval в after_train
            "DATASETS.TEST",
            "()",
        ]
    cfg.merge_from_list(merge_opts)

    if not smoke:
        cfg.SOLVER.MAX_ITER = _iters_per_epoch(cfg) * epochs

    cfg.merge_from_list(opts)  # --opts имеют приоритет

    cfg.OUTPUT_DIR = str(project_config.outputs_path / "mask_dino_dtn" / "experiments" / run_name)
    cfg.freeze()
    return cfg


def _setup_logging(cfg: CfgNode, config_path: Path) -> None:
    """Создаёт OUTPUT_DIR, копирует config.yaml и настраивает логгер/seed.

    params:
        cfg: замороженный конфиг
        config_path: путь к исходному YAML
    """
    out = Path(cfg.OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config_path, out / "config.yaml")
    setup_logger(output=str(out), distributed_rank=comm.get_rank(), name="maskdino")
    seed_all_rng(cfg.SEED)


def cmd_train(
    smoke: bool,
    size: int,
    epochs: int,
    run_name: str,
    ims_per_batch: int | None,
    num_workers: int | None,
    resume: bool,
    opts: list[str],
    config: Path,
) -> None:
    """Запускает обучение MaskDINO с MLflow/DagsHub-логгированием.

    params:
        smoke: короткий прогон без eval
        size: INPUT.IMAGE_SIZE
        epochs: число эпох
        run_name: имя MLflow-рана
        ims_per_batch: SOLVER.IMS_PER_BATCH (None — из конфига)
        num_workers: DATALOADER.NUM_WORKERS (None — из конфига)
        resume: продолжить с последнего чекпоинта
        opts: дополнительные опции detectron2 (KEY VALUE ...)
        config: путь к YAML-конфигу
    """
    cfg = _build_cfg(
        config=config,
        smoke=smoke,
        size=size,
        epochs=epochs,
        ims_per_batch=ims_per_batch,
        num_workers=num_workers,
        run_name=run_name,
        opts=opts,
    )
    _setup_logging(cfg, config)

    run = None
    if init_dagshub():
        try:
            run = mlflow.start_run(run_name=run_name)
            mlflow.log_params(
                {
                    "size": size,
                    "epochs": epochs,
                    "ims_per_batch": cfg.SOLVER.IMS_PER_BATCH,
                    "base_lr": cfg.SOLVER.BASE_LR,
                    "max_iter": cfg.SOLVER.MAX_ITER,
                    "config": str(config),
                }
            )
        except Exception as exc:  # noqa: BLE001 — best-effort трекинг
            print(f"WARNING: MLflow run init failed, continuing without logging: {exc}")
            run = None

    trainer = Trainer(cfg)
    if run is not None and not smoke:
        trainer.register_hooks([MLflowLossHook(_iters_per_epoch(cfg)), MLflowEvalHook()])
    trainer.resume_or_load(resume=resume)
    trainer.train()

    if run is not None:
        log_maskdino_artifacts(Path(cfg.OUTPUT_DIR))
        mlflow.end_run()
