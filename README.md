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
python run.py infer --submit
```

On Kaggle: GPU notebook, Add Data = this competition, clone or zip-upload this repo (no `.venv`), import `notebooks/kaggle_mask2former.ipynb`. Paste errors back here.

Home PQ: match instances at IoU ≥ 0.5, then `sum(IoU) / (TP + 0.5 FP + 0.5 FN)`. Val is whole images.
