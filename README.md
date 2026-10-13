# Filament Segmentation 2026

Kaggle: [filament-segmentation-2026](https://www.kaggle.com/competitions/filament-segmentation-2026).

v1: Otsu+CC ruler + Mask2Former (Swin-T, 3ch raw/CLAHE/high-pass). Same home PQ and RLE CSV. Train on Kaggle GPU. Cursor stays in this repo; git is the bridge.

```
src/data.py      paths, COCO, RLE, val split
src/metrics.py   home PQ
src/baseline.py  Otsu + connected components
src/m2f.py       Mask2Former dataset / train / predict
run.py           check | baseline | train | infer
```

```bash
pip install -r requirements.txt
python run.py check
python run.py baseline --viz 8          # needs competition data
python run.py train --smoke             # 30 steps, Kaggle GPU
python run.py train --size 512 --epochs 8
python run.py train --size 1024 --epochs 8 --run-name test_run --eval-samples-val 100 --eval-samples-train 200
python run.py infer --submit
```

Swin pipeline:

```bash
python run.py mask_dino_swin train --smoke       # 30 steps, Kaggle GPU
python run.py mask_dino_swin train --size 512 --epochs 8 --run-name mask_dino_swin_t
python run.py mask_dino_swin infer --submit
python run.py mask_dino_swin sweep
```

On Kaggle: GPU notebook, Add Data = this competition, clone or zip-upload this repo (no `.venv`), import `notebooks/kaggle_mask2former.ipynb`. Paste errors back here.

Home PQ: match instances at IoU ≥ 0.5, then `sum(IoU) / (TP + 0.5 FP + 0.5 FN)`. Val is whole images.

Training flags:

- `--run-name NAME` — MLflow run name on DagsHub (default `m2f`);
- `--eval-samples-val N` — PQ val on N random val images (0 = all val);
- `--eval-samples-train N` — PQ train on N random train images (0 = train PQ skipped).

PQ runs on whole 2048×2048 images for both train and val each epoch (when train eval is enabled).

Artifacts per run, `outputs/m2f/experiments/<run_name>/`:

```
metrics.csv         epoch, train_loss, train_pq, val_pq per epoch
weights/epoch_N.pt  weights after each epoch
best.pt             best weights by val_pq
```
