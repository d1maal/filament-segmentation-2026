from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


def _discover_dataset_root() -> Path:
    """Определяет корень датасета MAGFiLO.
    returns:
        путь к корню датасета
    """
    env = os.environ.get("MAGFILO_ROOT")
    if env:
        return Path(env)
    candidates = [
        ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026",
        Path("/kaggle/input/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026"),
        Path("/kaggle/input/filament-segmentation-2026"),
    ]
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        candidates.extend(
            p.parent.parent if p.parent.name == "train" else p.parent
            for p in kaggle_input.rglob("train_images")
        )
    for root in candidates:
        if (root / "train" / "train_images").is_dir():
            return root
    return candidates[0]


def _discover_outputs_path() -> Path:
    """Определяет папку для выходных артефактов.
    returns:
        путь к папке выходных артефактов
    """
    if Path("/kaggle/working").is_dir():
        return Path("/kaggle/working/outputs")
    return ROOT / "outputs"


@dataclass
class DagshubConfig:
    """Хранит настройки логирования экспериментов в DagsHub."""

    repo_owner: str = os.environ.get("DAGSHUB_REPO_OWNER", "BraveCactus")
    repo_name: str = os.environ.get("DAGSHUB_REPO_NAME", "solar-filament-segmentation")
    experiment_name: str = os.environ.get("DAGSHUB_EXPERIMENT_NAME", "Test_Experiment_1")
    token: str = os.environ.get("DAGSHUB_TOKEN", "")
    mlflow_run_name: str = "m2f"

    @property
    def configured(self) -> bool:
        """Возвращает признак того, что логирование настроено: владелец, репозиторий и токен заданы."""
        return bool(self.repo_owner and self.repo_name and self.token)


@dataclass
class Config:
    """Хранит все пути проекта и настройки логирования в DagsHub."""

    root: Path = ROOT
    dataset_path: Path = field(default_factory=_discover_dataset_root)
    outputs_path: Path = field(default_factory=_discover_outputs_path)
    models_path: Path = field(init=False)
    val_ids_path: Path = field(init=False)
    train_images: Path = field(init=False)
    train_annotations: Path = field(init=False)
    test_images: Path = field(init=False)
    submission_baseline_path: Path = field(init=False)
    submission_model_path: Path = field(init=False)

    dagshub: DagshubConfig = field(default_factory=DagshubConfig)

    def __post_init__(self) -> None:
        """Вычисляет производные пути из корня датасета и выходной папки."""
        train = self.dataset_path / "train"
        anns = sorted(train.glob("*Annotations*.json"))
        annotations = (
            anns[0] if anns else train / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
        )
        self.train_images = train / "train_images"
        self.train_annotations = annotations
        self.test_images = self.dataset_path / "test" / "test_images"
        self.models_path = self.outputs_path / "m2f"
        self.val_ids_path = self.root / "splits" / "val_ids.txt"
        self.submission_baseline_path = self.outputs_path / "submission_baseline.csv"
        self.submission_model_path = self.outputs_path / "submission_m2f.csv"


config = Config()
