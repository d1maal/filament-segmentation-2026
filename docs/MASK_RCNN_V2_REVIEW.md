# Mask R-CNN v2 review — `07e109c`

Branch: `feature/mask-rcnn`  
Commit: `07e109c` (Huzmzmzm4444, 2026-10-05) — *upgrade Mask R-CNN architecture and disjoint postprocessing*  
Base: `c949b21` (postprocess sweep)

## What changed

| Area | v1 (trained, LB ~0.28) | v2 (this commit, untrained) |
|------|------------------------|-----------------------------|
| Architecture flag | implicit ResNet50-FPN | `architecture="v2"` (+ keep `v1` builder) |
| Norm | BatchNorm (torchvision default) | **GroupNorm(32)** throughout backbone/heads |
| Heads | stock | deeper box head (4×256 conv + FC) / mask head (4×256) |
| Anchors aspect | `(0.25…4)` | **`(0.125…8)`** for thin filaments |
| Image size | 1024 | **1280** |
| Epochs / lr / wd | 20 / 1e-4 / 0.05 | **30 / 2e-4 / 1e-4** |
| Schedule | (previous) | **OneCycleLR** + AMP + grad clip 5 |
| Augment | geometric only | **photometric** + geometric |
| Score / min_area | 0.30 / 80 | **0.50 / 150** |
| Dedup | optional / drop | default **merge** + NMS + containment |
| Overlaps | external noolap hurt LB | **`enforce_disjoint`** pixel assignment by `score^power` |

Diff size: 8 files, ~422+/471−. Core files: `model.py`, `infer.py`, `train.py`, `config.py`, `dataset.py`, `sweep_postprocess.py`.

## Why this is the right next run

1. Home PQ of v1 (**0.345**) already beats M2F (~0.27–0.31).
2. Public LB drop to **~0.28** was forced-disjoint damage; v2 builds disjoint into infer.
3. Elongated anchors + GN + 1280 target the failure modes of thin Hα filaments.
4. v1 `best.pt` is **not** loadable into v2 — full retrain required.

## How to run

```python
from src.models.mask_rcnn.config import config
from src.models.mask_rcnn.train import train
# defaults: architecture=v2, image_size=1280, epochs=30
train(epochs=30, image_size=1280, eval_limit=50)
```

Kaggle notebook: `notebooks/kaggle_mask_rcnn.ipynb` (v2 defaults). Smoke 1 epoch, then full train.

## Risks

- 30×1280 on T4 may approach the 12h session limit → keep `EVAL_LIMIT=50` during train; full val after.
- GroupNorm + from-scratch may need the full 30 epochs before matching v1 probe PQ (0.301 @50 imgs).
- Do not resume v1 checkpoints into v2.
