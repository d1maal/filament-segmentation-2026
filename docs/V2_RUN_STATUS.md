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
| Push branch with notebooks/docs | pending push | |
| Smoke 1 epoch | pending on Kaggle | set `SMOKE=True` first |
| Full train 30 ep @1280 | pending | `SMOKE=False`, `EVAL_LIMIT=50` |
| Full val PQ | pending | after train cell |
| Submit `submission_mask_rcnn_v2.csv` | pending | overlap assert must pass |

### How to launch (Kaggle UI)

1. New GPU notebook, Internet **on**, Add Data = `filament-segmentation-2026`.
2. Secret `GITHUB_TOKEN` attached to the notebook.
3. Upload / copy cells from `notebooks/kaggle_mask_rcnn.ipynb` **or** clone branch and `%run` / open that file.
4. Run clone + GPU cells → set `SMOKE=True` → train 1 epoch.
5. If OK: `SMOKE=False`, Save Version **Save & Run All** with Output.
6. When finished: open `train_log.csv`, run val+submit cell, download `submission_mask_rcnn_v2.csv`.

### Evening metrics checklist

Fill when train progresses:

```
epoch_mid (e.g. 10): loss=____ probe_pq=____
epoch_late (e.g. 20): loss=____ probe_pq=____
best_epoch: ____ best_probe_pq=____
full_val_pq: ____  (target ≥ 0.345 home, improve LB vs 0.28)
overlap_pixels_val: ____ (must be 0)
public_lb: ____
```

Compare to v1 probe best **0.301** @ epoch 20 (50-img) and full home **0.345**.
