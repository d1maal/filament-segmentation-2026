# Mask R-CNN v2 run status

Updated: 2026-10-07

## Done locally

- [x] Checked out `feature/mask-rcnn` @ `07e109c`
- [x] Review: `docs/MASK_RCNN_V2_REVIEW.md`
- [x] Notebook updated: `notebooks/kaggle_mask_rcnn.ipynb` (1280 / 30 / v2 / disjoint assert)
- [x] Error autopsy: `docs/ERROR_AUTOPSY_V1.md`
- [x] Lead decision: `docs/ARCHITECTURE_DECISION.md`

## Kaggle train

| Step | Status | Notes |
|------|--------|-------|
| Push branch with notebooks/docs | **done** | `12d7ef4` on `origin/feature/mask-rcnn` |
| Kernel push + GPU run | **launched** | https://www.kaggle.com/code/dm1triyaleksandrov/mask-rcnn-v2 (v1, T4, 12h timeout) |
| Smoke 1 epoch | skipped for overnight full run | re-run with `SMOKE=True` if this version errors early |
| Full train 30 ep @1280 | **running / queued** | `SMOKE=False`, `EVAL_LIMIT=50` in notebook |
| Full val PQ | pending | after train cell in same notebook |
| Submit `submission_mask_rcnn_v2.csv` | pending | overlap assert must pass |

Public LB baselines (API 2026-10-07): MRCNN noolap **0.28**, M2F noolap **0.24**.

### Evening metrics checklist

Fill from kernel Output / `train_log.csv` when available:

```
kernel_url: https://www.kaggle.com/code/dm1triyaleksandrov/mask-rcnn-v2
epoch_mid (e.g. 10): loss=____ probe_pq=____
epoch_late (e.g. 20): loss=____ probe_pq=____
best_epoch: ____ best_probe_pq=____
full_val_pq: ____  (target ≥ 0.345 home, improve LB vs 0.28)
overlap_pixels_val: ____ (must be 0)
public_lb: ____
```

Compare to v1 probe best **0.301** @ epoch 20 (50-img) and full home **0.345**.

### If the kernel fails early

1. Open the kernel → check clone / CUDA / OOM.
2. Interactive: set `SMOKE=True`, run 1 epoch.
3. Fix, push notebook again: `kaggle kernels push -p <folder> -t 43200 --accelerator gpu`.
