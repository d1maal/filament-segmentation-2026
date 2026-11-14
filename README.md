# Solar Filament Segmentation Challenge 2026: Mask R-CNN solution

Team code for the Kaggle competition [filament-segmentation-2026](https://www.kaggle.com/competitions/filament-segmentation-2026)
(instance segmentation of solar filaments in GONG H-alpha images, MAGFiLO annotations).

**Final submission:** Mask R-CNN (ResNet-50-FPN with GroupNorm), 1344 px input, 8-view TTA, non-overlapping masks.
Home validation PQ 0.4235 (387 images), public leaderboard PQ 0.35.

## Reproducing the submission

```bash
git clone https://github.com/d1maal/filament-segmentation-2026.git
cd filament-segmentation-2026
pip install -r requirements.txt
export MAGFILO_ROOT=/path/to/MAGFiLO_1.0_Kaggle_2026   # not needed on Kaggle
jupyter notebook notebooks/pipeline_mask_rcnn.ipynb
```

The checkpoint (188 MB, SHA-256 `c7638d6e…f3b6`) is stored as four chunks under `weights/` because GitHub caps files at
100 MB; the notebook joins them with `weights/assemble.py`, evaluates the model on the validation split and writes
`outputs/submission_mask_rcnn_v5.csv`. No training is needed. Same thing from the command line:

```bash
python weights/assemble.py
python -m src.models.mask_rcnn.submission --weights weights/mask_rcnn_v5_best.pt --tta full \
       --output outputs/submission_mask_rcnn_v5.csv
```

On Kaggle: GPU T4, Internet on, Add Data = the competition, clone the repo into `/kaggle/working`, open the notebook.

## Pipeline

| Stage | What happens | Code |
|---|---|---|
| Input | 2048×2048 grayscale JPEG → 3 channels: raw, CLAHE, high-pass (image − Gaussian blur) | `src/data.py::three_channel` |
| Split | 80/20 by image file name, seed 42 (all annotator copies of an image stay together) | `src/data.py::ensure_val_split` |
| Augmentation | random 90° rotation, h/v flips; brightness/contrast, gamma, Gaussian noise | `src/models/mask_rcnn/dataset.py` |
| Model | Mask R-CNN, ResNet-50-FPN + GroupNorm (trained from scratch), anchors 16–256 px with aspect ratios 1/8…8, 4-conv box head, 6-conv mask head on 32×32 RoIs, BCE + 0.5·Dice mask loss | `src/models/mask_rcnn/model.py`, `losses.py` |
| Training | 42 epochs, AdamW 1.5e-4, 2-epoch warm-up + cosine to 5e-6, AMP, batch 1, best epoch by full-val PQ | `src/models/mask_rcnn/train.py` |
| Inference | 8 views (identity, 3 rotations, 2 flips, transpose, anti-transpose), masks mapped back | `src/models/mask_rcnn/infer.py` |
| Selection | score ≥ 0.55, area ≥ 150 px, quality = score × mask confidence × stability, weighted merge of duplicates (mask IoU ≥ 0.6) | `infer.py::postprocess_prepared` |
| No overlaps | each contested pixel goes to the mask with the highest logit(p) + 0.35·logit(quality); masks keeping < 10 % of their area are dropped | `infer.py::_make_disjoint` |
| Output | RLE counts (pycocotools), one row per filament | `src/models/mask_rcnn/submission.py` |

Metric used for model selection (`src/metrics.py`): instances matched at IoU ≥ 0.5,
PQ = ΣIoU / (TP + 0.5·FP + 0.5·FN), accumulated over whole images.

## Repository layout

```
notebooks/pipeline_mask_rcnn.ipynb   end-to-end pipeline (final solution)
weights/                             released checkpoint (epoch 36) in 4 chunks + assemble.py
src/models/mask_rcnn/                final model: config, dataset, model, losses, train, infer,
                                     submission, sweep_postprocess (threshold search), autopsy (error analysis)
src/models/m2f/, src/models/edgeattn/  earlier experiments: Mask2Former, Edge Attention Net
src/baseline.py                      Otsu + connected components baseline
src/data.py, src/metrics.py          data loading, RLE, split, PQ
experiments.md                       all runs with home PQ / leaderboard scores
docs/                                design notes and error analyses
notebooks/archive/                   executed Kaggle notebooks from earlier runs
```

## Results

| Model | Home PQ | Public LB |
|---|---:|---:|
| Mask2Former Swin-T, 1024 px | 0.273 | 0.24 |
| Edge Attention Net | — | 0.22 |
| Mask R-CNN v1, 1024 px | 0.345 | 0.28 |
| Mask R-CNN v2, 1280 px | 0.391 | 0.34 |
| **Mask R-CNN v5, 1344 px** | **0.4235** | **0.35** |

Details per run: [`experiments.md`](experiments.md).

## Team

Daria Repkina, Matvey Baranchugov, Dmitriy Aleksandrov (MIPT).
