# Mask R-CNN v2 run status

Updated: 2026-10-07 (~00:26 MSK)

## Done locally

- [x] Checked out `feature/mask-rcnn` @ `07e109c` (+ follow-up commits)
- [x] Review: `docs/MASK_RCNN_V2_REVIEW.md`
- [x] Notebook updated: `notebooks/kaggle_mask_rcnn.ipynb` (1280 / 30 / v2 / disjoint assert; dataset unpack)
- [x] Error autopsy: `docs/ERROR_AUTOPSY_V1.md`
- [x] Lead decision: `docs/ARCHITECTURE_DECISION.md`
- [x] Source dataset: https://www.kaggle.com/datasets/dm1triyaleksandrov/mask-rcnn-v2-src

## Kaggle train

| Step | Status | Notes |
|------|--------|-------|
| Push branch with notebooks/docs | **done** | `feature/mask-rcnn` includes notebook + docs |
| Kernel push + GPU run | **RUNNING** | https://www.kaggle.com/code/dm1triyaleksandrov/mask-rcnn-v2 **version 3** |
| Early failures | v1 git private / v2 missing zip path — fixed in v3 via unpacked dataset | |
| Full train 30 ep @1280 | **in progress** | `SMOKE=False`, `EVAL_LIMIT=50`, T4, 12h timeout |
| Full val PQ | pending | same notebook after train |
| Submit `submission_mask_rcnn_v2.csv` | pending | overlap assert must pass |

Public LB baselines (API 2026-10-07): MRCNN noolap **0.28**, M2F noolap **0.24**.

### Evening metrics checklist (fill when Output ready)

```
kernel_url: https://www.kaggle.com/code/dm1triyaleksandrov/mask-rcnn-v2
status_at_handoff: RUNNING (v3, ~00:26 MSK 2026-10-07)
epoch_mid (e.g. 10): loss=____ probe_pq=____
epoch_late (e.g. 20): loss=____ probe_pq=____
best_epoch: ____ best_probe_pq=____
full_val_pq: ____  (target ≥ 0.345 home, improve LB vs 0.28)
overlap_pixels_val: ____ (must be 0)
public_lb: ____
```

Compare to v1 probe best **0.301** @ epoch 20 (50-img) and full home **0.345**.

Poll: `env -u HTTP_PROXY -u HTTPS_PROXY kaggle kernels status dm1triyaleksandrov/mask-rcnn-v2`

### If the kernel fails

1. `kaggle kernels output dm1triyaleksandrov/mask-rcnn-v2 -p /tmp/out`
2. Interactive: `SMOKE=True`, 1 epoch.
3. Re-push: folder `/tmp/kaggle-mask-rcnn-v2-push` + `kaggle kernels push -p . -t 43200 --accelerator gpu`.
