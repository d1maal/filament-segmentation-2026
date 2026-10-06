# Experiments log

Home PQ on frozen val (whole disks). Train: random crop; infer: full 2048 → model `size`.

**Lead decision (2026-10-07):** mainline = Mask R-CNN family — see [`docs/ARCHITECTURE_DECISION.md`](docs/ARCHITECTURE_DECISION.md).

| id | date | size | epochs | train loss (last) | home PQ | TP | FP | FN | notes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| m2f-v1-512 | 2026-09-04 | 512 | 8 | 18.16 | **0.1145** | 318 | 292 | 2515 | Swin-T, 3ch, accum=8. Weights lost (Kaggle session restart). |
| m2f-v2-1024 | 2026-09-04 | 1024 | 8 | 20.43 | **0.2734** | 957 | 663 | 1876 | Swin-T, 3ch. splits=168 merges=95. Save Kaggle Version + last.pt. |
| mrcnn-v1-1024 | 2026-09-24/10-01 | 1024 | 20 | — | **0.3447** | 1712 | 1799 | 1121 | Mask R-CNN ResNet50-FPN. Public LB ~0.28 after noolap. See `outputs/mask_rcnn_submit/dasha_report/`. |
| mrcnn-v2-1280 | 2026-10-07 | 1280 | 30 | TBD | TBD | | | | GN + elongated anchors + `enforce_disjoint`. Kaggle: `dm1triyaleksandrov/mask-rcnn-v2`. |

## Commands

```bash
# v1
python run.py train --size 512 --epochs 8
python run.py infer

# v2
python run.py train --size 1024 --epochs 8
python run.py infer
```

