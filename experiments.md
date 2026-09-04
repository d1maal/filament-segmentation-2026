# Experiments log

Home PQ on frozen val (whole disks). Train: random crop; infer: full 2048 → model `size`.

| id | date | size | epochs | train loss (last) | home PQ | TP | FP | FN | notes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| v1-512 | 2026-09-04 | 512 | 8 | 18.16 | **0.1145** | 318 | 292 | 2515 | Swin-T, 3ch, accum=8. Weights lost (Kaggle session restart). |
| v2-1024 | 2026-09-04 | 1024 | 8 | 20.43 | **0.2734** | 957 | 663 | 1876 | Swin-T, 3ch. splits=168 merges=95. Save Kaggle Version + last.pt. |

## Commands

```bash
# v1
python run.py train --size 512 --epochs 8
python run.py infer

# v2
python run.py train --size 1024 --epochs 8
python run.py infer
```

