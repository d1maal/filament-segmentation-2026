# Error autopsy — Mask R-CNN v1 (home val)

Source: `outputs/mask_rcnn_submit/dasha_report/val_per_image.csv`  
Model: ResNet50-FPN, 20 ep @1024, postprocess score=0.5 / min_area=150 / dedup  
Aggregate: **PQ 0.345** | TP 1712 | FP **1799** | FN **1121** | splits 258 | merges 111 | n=387  
Public LB after noolap: **~0.28**

## Diagnosis (priority order)

1. **Precision problem dominates** — FP/FN ≈ **1.60**. Almost as many false positives as true positives. Junk blobs and chromosphere darkenings are expensive under PQ (each FP costs 0.5 in the denominator).
2. **Recall still material** — 1121 FN; crowded / faint disks (e.g. `20220721185332Bh` fn=13) miss whole filaments.
3. **Fragmentation > over-merge** — splits 258 vs merges 111. Model often breaks one filament into several; merge-mode dedup in v2 should help, but selector/merge logic still needed.
4. **Hard failures** — 19 images with PQ=0 (total miss or pure noise). Top FP images sit at 14–17 FP (`20130719140154Bh`, `20140710083214Th`, `20151119133734Ch`).
5. **Home→LB gap (~0.06)** — attributed to forced disjoint after overlapping masks. v2 `enforce_disjoint` addresses this in-infer.

## Worst stems (for overlay review next)

| Mode | Stems |
|------|-------|
| PQ=0 | `20130225171214Mh`, `20131107043054Uh`, `20170106133714Th`, `20170218173710Mh`, … |
| High FP | `20130719140154Bh`, `20140710083214Th`, `20151119133734Ch`, `20150423120034Ch` |
| High FN | `20220721185332Bh`, `20120610175114Mh`, `20130201084914Th`, `20131011195414Mh` |
| High splits | `20150825082034Lh`, `20140210180154Bh`, `20120427162614Mh` |

## Submit-path check (v2 code)

Verified in `src/models/mask_rcnn/config.py` + `infer.py`:

- `enforce_disjoint: bool = True` (default)
- `duplicate_mode: str = "merge"`
- `predict_masks` → `postprocess_output` → `postprocess_prepared` → `_make_disjoint` when flag set
- Notebook val/test cells assert **0 overlap pixels** before writing `submission_mask_rcnn_v2.csv`

No separate `resolve_overlaps` / noolap step required if defaults are left alone.

## Next engineering (after v2 train), not new architectures

1. Component / score selector to cut FP (largest expected gain).
2. Fragment merge for high-split stems.
3. Keep disjoint assignment score-aware (already `overlap_score_power=0.5`).
4. Only then Cascade/HTC or CondInst if PQ plateaus &lt; ~0.36–0.38.
