# Architecture decision (lead) — 2026-10-07

## Decision

**Mainline = Mask R-CNN family. Freeze open-ended architecture hunting until v2 + one error-driven iteration plateau.**

| Track | Role | Status |
|-------|------|--------|
| Mask R-CNN v2 (Dasha) | **mainline** | code on `feature/mask-rcnn` @ `07e109c+`; retrain now |
| Mask2Former + SAHI/TTA (Matvey) | diversity / ensemble only | `feature/optimization`; do not block mainline |
| Cascade Mask R-CNN / HTC | next upgrade **inside** family | after v2 metrics |
| CondInst | single side-bet if plateau | not this week |
| EdgeAttNet / tracers / random SOTA | **out** | wrong ROI before Nov 15 |

## Why

1. Empirics: MRCNN home PQ **0.345** > M2F **~0.27–0.31** on the same competition data.
2. Biggest known leak is **home→LB (~0.06)** from overlap rejection — a postprocess problem; v2 ships `enforce_disjoint`.
3. Error autopsy: FP-heavy, splits > merges — solved by selector + merge + cascade heads, not a new transformer.
4. Time: deadline **2026-11-15**. Train → autopsy → postprocess → cascade beats literature tourism.
5. Paper quality filter: Ahmadzadeh-style Mask R-CNN on Hα and CondInst/HTC are solid; crack-transformer / EdgeAttNet / Pick-and-Trace are either semantic-only or too heavy for this contest.

## Stop rule

If after **v2 train + disjoint submit + one autopsy-driven postprocess pass** genuine home PQ stays **&lt; ~0.36–0.38**, authorize **one** of: Cascade/HTC **or** CondInst. Not both. Not a third family.

## Team split

- **Dasha + Dmitriy:** Mask R-CNN v2 train, submit, postprocess, cascade path.
- **Matvey:** finish SAHI/TTA numbers on M2F cheaply; keep branch unmerged until v2 is green; optional ensemble later.
- **Lead:** no new architecture PRs without beating the stop rule.
