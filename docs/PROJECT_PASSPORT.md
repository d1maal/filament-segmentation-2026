# Filament 2026 — паспорт проекта

Соревнование: [filament-segmentation-2026](https://www.kaggle.com/competitions/filament-segmentation-2026)  
Дедлайн: **15 ноября 2026**  
Метрика: **PQ** (качество instance-масок; на Kaggle ещё важно, чтобы маски **не пересекались**)

**Схема (картинка, как блок-схема):** [diagrams/filament-pipeline.svg](diagrams/filament-pipeline.svg)

---

## 1. Задача простыми словами

На снимках Солнца в Hα нужно найти **отдельные нити (filaments)** и нарисовать маску каждой.  
Оценивают, насколько маски совпадают с разметкой и не дробят/не склеивают объекты лишний раз.

---

## 2. Команда

| Кто | Трек | Зона ответственности |
|-----|------|----------------------|
| Даша | Mask R-CNN | модель, train, postprocess, ветка `feature/mask-rcnn` |
| Матвей | Mask2Former | `feature/optimization`, SAHI/TTA — **не mainline**, для справки/ensemble |
| Дмитрий | лид | Kaggle прогоны, submit, Notion/статус, merge когда есть цифры |

---

## 3. Решение (зафиксировано)

- **Mainline:** семейство **Mask R-CNN**, не прыгаем на новую архитектуру каждую неделю.
- **Mask2Former:** параллельно, но не блокирует основную линию.
- **Если v2 + submit + один круг postprocess не дают рост:** один апгрейд **внутри** Mask R-CNN (cascade), не «новая SOTA с нуля».

Подробнее: [ARCHITECTURE_DECISION.md](ARCHITECTURE_DECISION.md)

---

## 4. Карта репозитория

| Путь | Что это |
|------|---------|
| `src/models/mask_rcnn/` | Код Mask R-CNN (train, infer, config, postprocess) — **главное** |
| `src/models/m2f/` | Mask2Former — трек Матвея |
| `run.py` | CLI в основном под M2F; Mask R-CNN на Kaggle зовём из ноутбука напрямую |
| `notebooks/kaggle_mask_rcnn.ipynb` | Train v2 на Kaggle |
| `notebooks/kaggle_mask_rcnn_infer.ipynb` | Infer + submit с `best.pt` |
| `docs/` | Решения, autopsy, этот паспорт, SVG-схема |
| `experiments.md` | Короткий лог прогонов |

**Ветки:**

| Ветка | Содержание | Статус |
|-------|------------|--------|
| `feature/mask-rcnn` | Mask R-CNN v1 + v2, ноутбуки, фиксы infer | **рабочая** |
| `feature/optimization` | M2F + SAHI/TTA + EMA | не вмержено в main |
| `main` | старее, без актуального mask_rcnn |

---

## 5. Прогоны и цифры

PQ **home** = наша проверка на 387 val-картинках (если не написано иное).  
PQ **LB** = public leaderboard Kaggle после submit.

| ID | Модель | Условия | PQ home | PQ LB | Комментарий |
|----|--------|---------|---------|-------|-------------|
| m2f-v2 | Mask2Former Swin-T | 1024, 8 ep | ~0.27 | — | experiments.md |
| mrcnn-v1 | Mask R-CNN ResNet50 | 1024, 20 ep | **0.345** | **~0.28** | submit noolap; LB ниже из‑за overlap |
| mrcnn-v2 | Mask R-CNN v2 | 1280, 30 ep | **~0.40*** | — | *probe: 50 val-картинок/эпоху, не full 387 |
| mrcnn-v2 | best epoch (probe) | ep 27 | **0.4025** | — | train_log.csv |
| mrcnn-v2 | last epoch (probe) | ep 30 | 0.3944 | — | TP 230 FP 181 FN 114 на 50 img |

**Артефакты v2 train:**

- Kaggle kernel: `dm1triyaleksandrov/mask-rcnn-v2`
- `best.pt`, `train_log.csv` — в Output / Downloads
- Full val 387 + CSV submit — **ещё не сделаны** (kernel упал на infer; в git поправлен импорт)

---

## 6. Что сломалось и что починили

После train v2 ноутбук упал на ячейке val/submit: при импорте `src.eval` подтягивался Mask2Former и требовал `mlflow`.  
**Fix:** `src/eval/__init__.py` — не грузим M2F при импорте submission helpers (`ad24e05`).

---

## 7. План работ (ближайшее)

1. **Infer + full val 387** на `best.pt` (disjoint маски в infer v2).
2. **Submit** → сравнить LB с 0.28.
3. **Postprocess sweep** под v2, если LB/home не выросли как ждём.
4. **Матвей:** коротко SAHI/TTA цифры — без блокировки mainline.
5. **Если плато:** cascade/HTC на том же пайплайне данных.

---

## 8. Notion и «доска как Miro»

**Notion не умеет** бесконечный холст со стрелками как Miro/FigJam/narezka — только страницы, таблицы, embed.

**Как сделать нормальную схему:**

1. **FigJam** (figma.com/figjam) или **Miro** — рисуем блоки и стрелки как на narezka.
2. В Notion на странице проекта: `/embed` → вставить ссылку на board.
3. **Эталонная SVG-схема** лежит в репо (`docs/diagrams/filament-pipeline.svg`) — можно импортировать в FigJam или вставить картинкой в Notion.

Текстовый источник правды: **этот файл** + обновление после каждого submit.
