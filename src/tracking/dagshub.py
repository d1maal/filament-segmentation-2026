from __future__ import annotations

from pathlib import Path

import dagshub
import matplotlib.pyplot as plt
import mlflow
import numpy as np

from src.config import config


def init_dagshub() -> bool:
    """Инициализирует трекинг MLflow на DagsHub.
    returns:
        True, если трекинг настроен, иначе False
    """
    dc = config.dagshub
    if not dc.configured:
        print("DagsHub logging disabled: set DAGSHUB_TOKEN in .env")
        return False
    try:
        dagshub.init(repo_owner=dc.repo_owner, repo_name=dc.repo_name, mlflow=True)
        mlflow.set_experiment(dc.experiment_name)
        return True
    except Exception as exc:
        print(f"WARNING: DagsHub init failed, continuing without logging: {exc}")
        return False


def _log_training_artifacts(model, ckpt: Path, history: dict) -> None:
    """Сохраняет в активный MLflow-ран графики обучения, модель и чекпоинт.
    params:
        model: обученная модель Mask2Former
        ckpt: путь к чекпоинту last.pt
        history: словарь с историей train_loss, val_pq и train_pq по эпохам
    """
    try:
        fig, ax = plt.subplots(figsize=(8, 5))
        epochs = np.arange(1, len(history["train_loss"]) + 1)
        ax.plot(epochs, history["train_loss"], label="train_loss", marker="o")
        valid_val = [(e, v) for e, v in zip(epochs, history["val_pq"]) if v is not None]
        valid_train = [(e, v) for e, v in zip(epochs, history["train_pq"]) if v is not None]
        if valid_val or valid_train:
            ax2 = ax.twinx()
            if valid_val:
                ax2.plot(*zip(*valid_val), label="val_pq", marker="s", color="tab:orange")
            if valid_train:
                ax2.plot(*zip(*valid_train), label="train_pq", marker="^", color="tab:green")
            ax2.set_ylabel("pq")
        ax.set_xlabel("epoch")
        ax.set_ylabel("train_loss")
        fig.legend(loc="upper right")
        fig.tight_layout()
        mlflow.log_figure(fig, "training_curves.png")
        plt.close(fig)
        mlflow.pytorch.log_model(model, "mask2former_model", serialization_format="pt2")
    except Exception as exc:
        print(f"WARNING: MLflow artifacts failed, ckpt still logged: {exc}")
    mlflow.log_artifact(str(ckpt))