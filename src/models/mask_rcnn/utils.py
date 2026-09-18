import json
from pathlib import Path

import torch


def collate_fn(batch):
    return tuple(zip(*batch))


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
