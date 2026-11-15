# Experiments log

Home PQ on frozen val (whole disks). Train: random crop; infer: full 2048 → model `size`.

**Final model:** Mask R-CNN v5, see `notebooks/pipeline_mask_rcnn.ipynb`.

**Lead decision (2026-10-07):** mainline = Mask R-CNN family — see [`docs/ARCHITECTURE_DECISION.md`](docs/ARCHITECTURE_DECISION.md).

| id | date | size | epochs | train loss (last) | home PQ | TP | FP | FN | notes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| m2f-v1-512 | 2026-09-04 | 512 | 8 | 18.16 | **0.1145** | 318 | 292 | 2515 | Swin-T, 3ch, accum=8. Weights lost (Kaggle session restart). |
| m2f-v2-1024 | 2026-09-04 | 1024 | 8 | 20.43 | **0.2734** | 957 | 663 | 1876 | Swin-T, 3ch. splits=168 merges=95. Save Kaggle Version + last.pt. |
| mrcnn-v1-1024 | 2026-09-24/10-01 | 1024 | 20 | — | **0.3447** | 1712 | 1799 | 1121 | Mask R-CNN ResNet50-FPN. Public LB ~0.28 after noolap. See `outputs/mask_rcnn_submit/dasha_report/`. |
| mrcnn-v2-1280 | 2026-10-07 | 1280 | 30 | — | **0.3914** | | | | GN + elongated anchors + `enforce_disjoint`. Public LB 0.34. |
| mrcnn-v3-1344 | 2026-10-12 | 1344 | 36 (stopped at 34) | 0.614 | **0.4166** (ep 30) | | | | mask BCE+Dice, 6-conv mask head, SWA. Run died at epoch 34: disk full (full checkpoint every epoch). |
| **mrcnn-v5-1344** | 2026-11-13 | 1344 | 42 | 0.600 | **0.4235** (ep 36) | 1912 | 1278 | 921 | `ed857d3`. Model-only checkpoints. SWA(last 5) 0.4223. Same weights, three inference settings: no TTA home 0.4235 / LB **0.37** (final), 4-view TTA 0.4126 / 0.36, 8-view TTA 0.4047 / 0.35. Weights: `weights/mask_rcnn_v5_best.pt`. |

## Commands

```bash
# v1
python run.py train --size 512 --epochs 8
python run.py infer

# v2
python run.py train --size 1024 --epochs 8
python run.py infer
```

