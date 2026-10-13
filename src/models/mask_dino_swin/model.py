"""Swin-бэкбон с MaskDINO-подобной query-головой на чистом torch.

Упрощения относительно оригинального MaskDINO: вместо deformable attention —
обычный TransformerDecoder, вместо detectron2 — HF SwinModel, боксовая голова
опущена (только маски), сопоставление — венгерское по focal+dice стоимостям.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from transformers import AutoModel, SwinConfig

STAGE_CHANNELS = (96, 192, 384, 768)
MATCH_RESOLUTION = 128


class PositionEmbeddingSine(nn.Module):
    """Синусоидальное позиционное кодирование для карт признаков."""

    def __init__(self, num_pos_feats: int = 128, temperature: float = 10000.0) -> None:
        """Запоминает число признаков и температуру шкалы.
        params:
            num_pos_feats: признаков на каждую ось
            temperature: база температурной шкалы
        """
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Строит позиционное кодирование того же пространственного размера, что x.
        params:
            x: карта признаков (B, C, H, W)
        returns:
            кодирование (B, 2 * num_pos_feats, H, W)
        """
        _, _, h, w = x.shape
        mask = torch.zeros((x.shape[0], h, w), device=x.device, dtype=torch.bool)
        y_embed = mask.cumsum(1, dtype=torch.float32)
        x_embed = mask.cumsum(2, dtype=torch.float32)
        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * 2 * torch.pi
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * 2 * torch.pi
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        return torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)


class MLP(nn.Module):
    """Многослойный перцептрон для mask-эмбеддингов."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        """Собирает два линейных слоя с ReLU между ними.
        params:
            in_dim: размер входа
            hidden_dim: размер скрытого слоя
            out_dim: размер выхода
        """
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Применяет MLP к последнему измерению.
        params:
            x: входной тензор
        returns:
            преобразованный тензор
        """
        return self.layers(x)


class SwinMaskDINO(nn.Module):
    """Swin-энкодер с FPN и query-декодером в духе MaskDINO."""

    def __init__(
        self,
        backbone_name: str,
        num_queries: int = 100,
        hidden_dim: int = 256,
        mask_dim: int = 128,
        dec_layers: int = 3,
        nheads: int = 8,
        pretrained: bool = True,
    ) -> None:
        """Собирает бэкбон, FPN-декодер и трансформерную голову.
        params:
            backbone_name: HF-имя Swin-бэкбона (tiny/small, embed_dim 96)
            num_queries: число learnable queries
            hidden_dim: ширина трансформера
            mask_dim: ширина mask-фичей FPN
            dec_layers: число слоёв декодера
            nheads: число голов внимания
            pretrained: грузить ImageNet-веса бэкбона
        """
        super().__init__()
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        # Запасной конфиг повторяет swin-tiny (embed_dim 96), чтобы сошлись каналы FPN.
        tiny_config = SwinConfig(embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24))
        self.backbone = (
            AutoModel.from_pretrained(backbone_name, add_pooling_layer=False)
            if pretrained
            else AutoModel.from_config(tiny_config)
        )
        self.lateral = nn.ModuleList(
            [nn.Conv2d(ch, mask_dim, kernel_size=1) for ch in STAGE_CHANNELS]
        )
        self.out_conv = nn.Sequential(
            nn.Conv2d(mask_dim, mask_dim, kernel_size=3, padding=1),
            nn.GroupNorm(32, mask_dim),
            nn.ReLU(inplace=True),
        )
        self.input_proj = nn.Conv2d(STAGE_CHANNELS[-1], hidden_dim, kernel_size=1)
        self.pos_enc = PositionEmbeddingSine(hidden_dim // 2)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.query_pos = nn.Embedding(num_queries, hidden_dim)
        layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=nheads, dim_feedforward=1024, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=dec_layers)
        self.class_embed = nn.Linear(hidden_dim, 1)
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim)

    def _backbone_features(self, pixel_values: torch.Tensor) -> list[torch.Tensor]:
        """Достаёт 4 стадии Swin по пространственному размеру, а не по позиции.
        params:
            pixel_values: батч (B, 3, S, S)
        returns:
            список из 4 карт признаков со страйдами 4/8/16/32
        """
        in_size = pixel_values.shape[-1]
        expected = {in_size // (4 * 2**i): i for i in range(4)}
        out = self.backbone(pixel_values, output_hidden_states=True)
        states = list(getattr(out, "hidden_states", None) or [out.last_hidden_state])
        picked: dict[int, torch.Tensor] = {}
        for h in states:
            if h.ndim == 4:
                h4 = h
            elif h.ndim == 3:
                b, length, ch = h.shape
                s = int(round(length**0.5))
                if s * s != length:
                    continue
                h4 = h.transpose(1, 2).reshape(b, ch, s, s)
            else:
                continue
            idx = expected.get(h4.shape[-1])
            if idx is not None and idx not in picked:
                picked[idx] = h4
        if len(picked) != 4:
            raise RuntimeError(
                f"Expected 4 Swin stages, found {sorted(picked)} for input {in_size}"
            )
        feats = [picked[i] for i in range(4)]
        for feat, ch in zip(feats, STAGE_CHANNELS):
            if feat.shape[1] != ch:
                raise RuntimeError(
                    f"Backbone channel mismatch: got {feat.shape[1]}, want {ch}"
                )
        return feats

    def forward(self, pixel_values: torch.Tensor) -> dict[str, torch.Tensor]:
        """Считает скоры queries и маски инстансов.
        params:
            pixel_values: нормализованный батч (B, 3, S, S), S кратно 32
        returns:
            словарь с pred_logits (B, Q) и pred_masks (B, Q, S/4, S/4)
        """
        feats = self._backbone_features(pixel_values)
        lateral = [conv(f) for conv, f in zip(self.lateral, feats)]
        y = lateral[3]
        for i in (2, 1, 0):
            y = F.interpolate(y, size=lateral[i].shape[-2:], mode="nearest") + lateral[i]
        mask_features = self.out_conv(y)
        mem = self.input_proj(feats[3])
        pos = self.pos_enc(mem).flatten(2).permute(0, 2, 1)
        mem = mem.flatten(2).permute(0, 2, 1)
        b = pixel_values.shape[0]
        queries = self.query_embed.weight.unsqueeze(0).expand(b, -1, -1)
        qpos = self.query_pos.weight.unsqueeze(0).expand(b, -1, -1)
        hs = self.decoder(tgt=queries + qpos, memory=mem + pos)
        pred_logits = self.class_embed(hs).squeeze(-1)
        pred_masks = torch.einsum("bqd,bdhw->bqhw", self.mask_embed(hs), mask_features)
        return {"pred_logits": pred_logits, "pred_masks": pred_masks}


def _focal_elementwise(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Поэлементный sigmoid focal loss без редукции.
    params:
        inputs: логиты
        targets: цели 0/1 той же формы
    returns:
        тензор потерь той же формы
    """
    prob = inputs.sigmoid()
    ce = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    alpha_t = 0.25 * targets + 0.75 * (1 - targets)
    return alpha_t * ce * ((1 - p_t) ** 2.0)


def sigmoid_focal_loss(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Усреднённый sigmoid focal loss.
    params:
        inputs: логиты
        targets: цели 0/1 той же формы
    returns:
        скалярный лосс
    """
    return _focal_elementwise(inputs, targets).mean()


def dice_loss(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Dice loss по queries, усреднённый по первой оси.
    params:
        inputs: логиты масок (N, ...)
        targets: бинарные цели (N, ...)
    returns:
        скалярный лосс
    """
    prob = inputs.sigmoid().flatten(1)
    tgt = targets.flatten(1)
    numerator = 2 * (prob * tgt).sum(1)
    denominator = prob.sum(1) + tgt.sum(1)
    return ((1 - (numerator + 1.0) / (denominator + 1.0))).mean()


@torch.no_grad()
def _pairwise_cost(
    pred_masks: torch.Tensor, gt_masks: torch.Tensor
) -> torch.Tensor:
    """Стоимости focal+dice для каждой пары (query, gt).
    params:
        pred_masks: логиты (Q, r, r)
        gt_masks: бинарные маски (G, r, r)
    returns:
        матрица стоимостей (Q, G)
    """
    q, g = pred_masks.shape[0], gt_masks.shape[0]
    pred = pred_masks.unsqueeze(1).expand(q, g, -1, -1)
    gt = gt_masks.unsqueeze(0).expand(q, g, -1, -1)
    focal = _focal_elementwise(pred, gt).mean(dim=(-2, -1))
    prob = pred.sigmoid().flatten(2)
    tgt = gt.flatten(2)
    dice = 1 - (2 * (prob * tgt).sum(2) + 1.0) / (prob.sum(2) + tgt.sum(2) + 1.0)
    return focal + dice


def match_and_loss(
    pred_logits: torch.Tensor,
    pred_masks: torch.Tensor,
    gt_masks: torch.Tensor,
    no_object_weight: float = 0.1,
) -> tuple[torch.Tensor, int]:
    """Венгерское сопоставление queries с GT и суммарный лосс одного изображения.
    params:
        pred_logits: скоры queries (Q,)
        pred_masks: логиты масок (Q, h, w)
        gt_masks: бинарные маски GT (G, H, W)
        no_object_weight: вес лосса пустых изображений
    returns:
        кортеж (скалярный лосс, число сопоставленных пар)
    """
    q = pred_logits.shape[0]
    g = gt_masks.shape[0]
    cls_target = torch.zeros_like(pred_logits)
    if g == 0:
        return sigmoid_focal_loss(pred_logits, cls_target) * no_object_weight, 0
    h, w = pred_masks.shape[-2:]
    if max(h, w) > MATCH_RESOLUTION:
        pred_small = F.interpolate(
            pred_masks.unsqueeze(0), size=MATCH_RESOLUTION, mode="bilinear",
            align_corners=False,
        ).squeeze(0)
    else:
        pred_small = pred_masks
    gt_small = F.interpolate(
        gt_masks.unsqueeze(0).float(), size=pred_small.shape[-2:], mode="nearest"
    ).squeeze(0)
    cost = _pairwise_cost(pred_small, gt_small)
    rows, cols = linear_sum_assignment(cost.detach().cpu().numpy())
    cls_target[torch.as_tensor(rows, device=pred_logits.device)] = 1.0
    cls_loss = sigmoid_focal_loss(pred_logits, cls_target)
    mask_loss = sigmoid_focal_loss(pred_small[rows], gt_small[cols])
    mask_loss = mask_loss + dice_loss(pred_small[rows], gt_small[cols])
    return cls_loss + mask_loss, len(rows)
