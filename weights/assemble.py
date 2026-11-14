"""Rebuild weights/mask_rcnn_v5_best.pt from the chunks stored in the repository.

GitHub rejects files above 100 MB, so the 188 MB checkpoint is kept as four
plain-git chunks (mask_rcnn_v5_best.pt.part-00 ... part-03).

    python weights/assemble.py
"""
from __future__ import annotations

import hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
TARGET = HERE / "mask_rcnn_v5_best.pt"
SHA256 = "c7638d6e766d22e4294f3662a929ce22de0519ace96fb84b429714e08ecaf3b6"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def assemble(target: Path = TARGET) -> Path:
    if target.exists() and sha256(target) == SHA256:
        return target
    parts = sorted(HERE.glob(target.name + ".part-*"))
    if not parts:
        raise FileNotFoundError(f"no chunks for {target.name} in {HERE}")
    with open(target, "wb") as out:
        for part in parts:
            out.write(part.read_bytes())
    digest = sha256(target)
    if digest != SHA256:
        target.unlink()
        raise ValueError(f"checksum mismatch: {digest}")
    return target


if __name__ == "__main__":
    print(assemble())
