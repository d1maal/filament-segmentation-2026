"""Swin-бэкбон с MaskDINO-головой на чистом torch.

Полноценная (без deformable attention) MaskDINO-голова: бокс-голова с L1/GIoU,
итеративное уточнение reference-боксов (DAB-DETR), two-stage выбор queries,
deep supervision по слоям декодера и denoising-обучение (DN).
"""

from __future__ import annotations

import copy
import math

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel, SwinConfig

STAGE_CHANNELS = (96, 192, 384, 768)


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
    """Многослойный перцептрон с ReLU."""

    def __init__(
        self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int = 2
    ) -> None:
        """Собирает линейные слои с ReLU между ними.

        params:
            in_dim: размер входа
            hidden_dim: размер скрытых слоёв
            out_dim: размер выхода
            num_layers: полное число линейных слоёв
        """
        super().__init__()
        layers: list[nn.Module] = []
        for i in range(num_layers - 1):
            layers.append(nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim if num_layers > 1 else in_dim, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Применяет MLP к последнему измерению.

        params:
            x: входной тензор
        returns:
            преобразованный тензор
        """
        return self.layers(x)


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Обратный сигмоид с ограничением.

    params:
        x: вход в [0, 1]
        eps: минимальное расстояние до границ
    returns:
        logit-значение
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def gen_sineembed_for_position(pos_tensor: torch.Tensor) -> torch.Tensor:
    """Синусоидальное кодирование координат боксов (cx, cy, w, h).

    params:
        pos_tensor: тензор (..., 4) с координатами в [0, 1]
    returns:
        кодирование (..., 512)
    """
    scale = 2 * math.pi
    dim_t = torch.arange(128, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / 128)
    x_embed = pos_tensor[..., 0] * scale
    y_embed = pos_tensor[..., 1] * scale
    pos_x = x_embed[..., None] / dim_t
    pos_y = y_embed[..., None] / dim_t
    pos_x = torch.stack(
        (pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1
    ).flatten(-2)
    pos_y = torch.stack(
        (pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1
    ).flatten(-2)
    w_embed = pos_tensor[..., 2] * scale
    pos_w = w_embed[..., None] / dim_t
    pos_w = torch.stack(
        (pos_w[..., 0::2].sin(), pos_w[..., 1::2].cos()), dim=-1
    ).flatten(-2)
    h_embed = pos_tensor[..., 3] * scale
    pos_h = h_embed[..., None] / dim_t
    pos_h = torch.stack(
        (pos_h[..., 0::2].sin(), pos_h[..., 1::2].cos()), dim=-1
    ).flatten(-2)
    return torch.cat((pos_y, pos_x, pos_w, pos_h), dim=-1)


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
    """Dice loss по первой оси.

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
    return (1 - (numerator + 1.0) / (denominator + 1.0)).mean()


class DecoderLayer(nn.Module):
    """Один слой декодера: self-attn, cross-attn, FFN (обычное внимание)."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.0,
        activation: str = "relu",
    ) -> None:
        """Собирает подслои декодера.

        params:
            d_model: ширина трансформера
            nhead: число голов
            dim_feedforward: ширина FFN
            dropout: dropout
            activation: функция активации FFN
        """
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(
        self,
        tgt: torch.Tensor,
        query_pos: torch.Tensor,
        memory: torch.Tensor,
        memory_key_padding_mask: torch.Tensor | None = None,
        tgt_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Применяет один слой декодера.

        params:
            tgt: queries (B, Q, C)
            query_pos: позиционное кодирование queries (B, Q, C)
            memory: память энкодера (B, N, C)
            memory_key_padding_mask: маска паддинга памяти (B, N)
            tgt_mask: аддитивная маска self-attention (Q, Q)
        returns:
            обновлённые queries (B, Q, C)
        """
        q = k = tgt + query_pos
        tgt2 = self.self_attn(q, k, tgt, attn_mask=tgt_mask)[0]
        tgt = self.norm1(tgt + self.dropout1(tgt2))
        tgt2 = self.cross_attn(
            tgt + query_pos, memory, memory, key_padding_mask=memory_key_padding_mask
        )[0]
        tgt = self.norm2(tgt + self.dropout2(tgt2))
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = self.norm3(tgt + self.dropout3(tgt2))
        return tgt


class TransformerDecoder(nn.Module):
    """Декодер с итеративным уточнением reference-боксов."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        num_layers: int,
        activation: str = "relu",
    ) -> None:
        """Собирает стек слоёв декодера и голову позиционного кодирования боксов.

        params:
            d_model: ширина трансформера
            nhead: число голов
            dim_feedforward: ширина FFN
            dropout: dropout
            num_layers: число слоёв декодера
            activation: функция активации FFN
        """
        super().__init__()
        layer = DecoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.ref_point_head = MLP(512, d_model, d_model, 2)
        self.bbox_embed: nn.Module | None = None

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        refpoints_unsigmoid: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        memory_key_padding_mask: torch.Tensor | None = None,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Прогоняет queries через слои, уточняя боксы.

        params:
            tgt: queries (B, Q, C)
            memory: память энкодера (B, N, C)
            refpoints_unsigmoid: стартовые reference-боксы (B, Q, 4) в logit-пространстве
            tgt_mask: булева маска self-attention (Q, Q)
            memory_key_padding_mask: маска паддинга памяти
        returns:
            кортеж (hs: L выходов (B, Q, C), references: L+1 боксов (B, Q, 4))
        """
        output = tgt
        reference_points = refpoints_unsigmoid.sigmoid()
        ref_points = [reference_points]
        hs: list[torch.Tensor] = []
        if tgt_mask is not None:
            attn_mask: torch.Tensor | None = tgt_mask.float().masked_fill(
                tgt_mask, float("-inf")
            )
        else:
            attn_mask = None
        for layer in self.layers:
            reference_before = reference_points
            query_sine = gen_sineembed_for_position(reference_before)
            query_pos = self.ref_point_head(query_sine)
            output = layer(
                output, query_pos, memory, memory_key_padding_mask, attn_mask
            )
            if self.bbox_embed is not None:
                delta = self.bbox_embed(output)
                new_ref = (inverse_sigmoid(reference_before) + delta).sigmoid()
                reference_points = new_ref.detach()
                ref_points.append(new_ref)
            hs.append(output)
        return hs, ref_points


class SwinMaskDINO(nn.Module):
    """Swin-энкодер с FPN и MaskDINO-декодером."""

    def __init__(
        self,
        backbone_name: str,
        num_queries: int = 100,
        hidden_dim: int = 256,
        mask_dim: int = 128,
        dec_layers: int = 3,
        nheads: int = 8,
        dim_feedforward: int = 1024,
        pretrained: bool = True,
        two_stage: bool = False,
        dn: str = "no",
        dn_num: int = 100,
        noise_scale: float = 0.4,
        num_classes: int = 1,
        deep_supervision: bool = True,
    ) -> None:
        """Собирает бэкбон, FPN и MaskDINO-декодер.

        params:
            backbone_name: HF-имя Swin-бэкбона
            num_queries: число learnable queries
            hidden_dim: ширина трансформера
            mask_dim: ширина mask-фичей FPN
            dec_layers: число слоёв декодера
            nheads: число голов внимания
            dim_feedforward: ширина FFN декодера
            pretrained: грузить ImageNet-веса бэкбона
            two_stage: выбирать queries из энкодера (top-k proposals)
            dn: режим denoising ("no" или "seg")
            dn_num: число DN-запросов
            noise_scale: масштаб шума боксов DN
            num_classes: число классов (1 — филамент)
            deep_supervision: aux-лоссы по слоям декодера
        """
        super().__init__()
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.two_stage = two_stage
        self.dn = dn
        self.dn_num = dn_num
        self.noise_scale = noise_scale
        self.deep_supervision = deep_supervision
        self.mask_dim = mask_dim

        tiny_config = SwinConfig(
            embed_dim=96,
            depths=(2, 2, 6, 2),
            num_heads=(3, 6, 12, 24),
            attn_implementation="eager",
        )
        self.backbone = (
            AutoModel.from_pretrained(
                backbone_name, add_pooling_layer=False, attn_implementation="eager"
            )
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

        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        self.query_embed = nn.Embedding(num_queries, 4)
        nn.init.constant_(self.query_embed.weight[..., :2], 0.5)
        nn.init.constant_(self.query_embed.weight[..., 2:], 1.0)

        if two_stage:
            self.enc_output = nn.Linear(hidden_dim, hidden_dim)
            self.enc_output_norm = nn.LayerNorm(hidden_dim)

        self.decoder = TransformerDecoder(
            hidden_dim, nheads, dim_feedforward, 0.0, dec_layers
        )
        self.decoder_norm = nn.LayerNorm(hidden_dim)

        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.label_enc = nn.Embedding(num_classes, hidden_dim)
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        self.decoder.bbox_embed = self.bbox_embed

    def _backbone_features(self, pixel_values: torch.Tensor) -> list[torch.Tensor]:
        """Достаёт 4 стадии Swin по пространственному размеру.

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
                s = round(length**0.5)
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

    def _pixel_features(
        self, pixel_values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Считает mask-фичи FPN и память энкодера.

        params:
            pixel_values: батч (B, 3, S, S)
        returns:
            кортеж (mask_features, memory, mem_flat, mem)
        """
        feats = self._backbone_features(pixel_values)
        lateral = [conv(f) for conv, f in zip(self.lateral, feats)]
        y = lateral[3]
        for i in (2, 1, 0):
            y = (
                F.interpolate(y, size=lateral[i].shape[-2:], mode="nearest")
                + lateral[i]
            )
        mask_features = self.out_conv(y)
        mem = self.input_proj(feats[3])
        pos = self.pos_enc(mem)
        mem_flat = mem.flatten(2).transpose(1, 2)
        pos_flat = pos.flatten(2).transpose(1, 2)
        memory = mem_flat + pos_flat
        return mask_features, memory, mem_flat, mem

    def _gen_proposals(self, mem: torch.Tensor) -> torch.Tensor:
        """Генерирует нормализованные grid-предложения для two-stage выбора.

        params:
            mem: карта признаков энкодера (B, C, H, W)
        returns:
            предложения (B, N, 4) в logit-пространстве
        """
        b, _, h, w = mem.shape
        gy, gx = torch.meshgrid(
            torch.arange(h, dtype=torch.float32, device=mem.device),
            torch.arange(w, dtype=torch.float32, device=mem.device),
            indexing="ij",
        )
        grid = torch.stack([gx, gy], dim=-1)
        grid = (grid + 0.5) / torch.tensor(
            [w, h], dtype=torch.float32, device=mem.device
        )
        wh = torch.full_like(grid, 0.05)
        proposal = torch.cat([grid, wh], dim=-1).reshape(1, -1, 4)
        return inverse_sigmoid(proposal).expand(b, -1, -1)

    def prepare_for_dn(
        self, targets: list[dict], batch_size: int
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        dict | None,
    ]:
        """Строит зашумлённые DN-запросы и маску внимания.

        params:
            targets: список целей с labels, boxes, masks
            batch_size: размер батча
        returns:
            кортеж (input_query_label, input_query_bbox, attn_mask, mask_dict)
        """
        scalar, noise_scale = self.dn_num, self.noise_scale
        known_num = [t["labels"].numel() for t in targets]
        if max(known_num) == 0:
            return None, None, None, None
        scalar = scalar // max(known_num)
        if scalar == 0:
            return None, None, None, None
        device = targets[0]["labels"].device
        labels = torch.cat([t["labels"] for t in targets])
        boxes = torch.cat([t["boxes"] for t in targets])
        masks = torch.cat([t["masks"] for t in targets])
        batch_idx = torch.cat(
            [
                torch.full((t["labels"].numel(),), i, dtype=torch.long, device=device)
                for i, t in enumerate(targets)
            ]
        )
        known_labels = labels.repeat(scalar)
        known_boxes = boxes.repeat(scalar, 1)
        known_masks = masks.repeat(scalar, 1, 1)
        known_bid = batch_idx.repeat(scalar)

        known_boxes_noisy = known_boxes.clone()
        if noise_scale > 0:
            diff = torch.zeros_like(known_boxes_noisy)
            diff[:, :2] = known_boxes_noisy[:, 2:] / 2
            diff[:, 2:] = known_boxes_noisy[:, 2:]
            known_boxes_noisy = (
                known_boxes_noisy
                + (torch.rand_like(known_boxes_noisy) * 2 - 1.0) * diff * noise_scale
            )
            known_boxes_noisy = known_boxes_noisy.clamp(min=0.0, max=1.0)

        input_label_embed = self.label_enc(known_labels)
        input_bbox_embed = inverse_sigmoid(known_boxes_noisy)

        single_pad = max(known_num)
        pad_size = single_pad * scalar
        padding_label = torch.zeros(pad_size, self.hidden_dim, device=device)
        padding_bbox = torch.zeros(pad_size, 4, device=device)
        input_query_label = padding_label.unsqueeze(0).repeat(batch_size, 1, 1)
        input_query_bbox = padding_bbox.unsqueeze(0).repeat(batch_size, 1, 1)

        map_known_indice = torch.cat(
            [torch.arange(num, device=device) for num in known_num]
        )
        map_known_indice = torch.cat(
            [map_known_indice + single_pad * i for i in range(scalar)]
        )
        input_query_label[(known_bid, map_known_indice)] = input_label_embed
        input_query_bbox[(known_bid, map_known_indice)] = input_bbox_embed

        tgt_size = pad_size + self.num_queries
        attn_mask = torch.zeros(tgt_size, tgt_size, dtype=torch.bool, device=device)
        attn_mask[pad_size:, :pad_size] = True
        for i in range(scalar):
            if i == 0:
                attn_mask[
                    single_pad * i : single_pad * (i + 1),
                    single_pad * (i + 1) : pad_size,
                ] = True
            elif i == scalar - 1:
                attn_mask[single_pad * i : single_pad * (i + 1), : single_pad * i] = (
                    True
                )
            else:
                attn_mask[
                    single_pad * i : single_pad * (i + 1),
                    single_pad * (i + 1) : pad_size,
                ] = True
                attn_mask[single_pad * i : single_pad * (i + 1), : single_pad * i] = (
                    True
                )

        mask_dict = {
            "pad_size": pad_size,
            "scalar": scalar,
            "known_labels": known_labels,
            "known_boxes": known_boxes,
            "known_masks": known_masks,
            "known_bid": known_bid,
            "map_known_indice": map_known_indice,
        }
        return input_query_label, input_query_bbox, attn_mask, mask_dict

    def pred_box(
        self, references: list[torch.Tensor], hs: list[torch.Tensor]
    ) -> torch.Tensor:
        """Итеративно предсказывает боксы по слоям декодера.

        params:
            references: список L+1 reference-боксов (B, Q, 4)
            hs: список L выходов декодера (B, Q, C)
        returns:
            боксы (L, B, Q, 4)
        """
        out = []
        for ref, h in zip(references[:-1], hs):
            delta = self.bbox_embed(h)
            out.append((inverse_sigmoid(ref) + delta).sigmoid())
        return torch.stack(out)

    def dn_post_process(
        self,
        outputs_class: torch.Tensor,
        outputs_coord: torch.Tensor,
        outputs_mask: torch.Tensor,
        mask_dict: dict,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Отделяет DN-часть от обычных предсказаний.

        params:
            outputs_class: (L, B, Q', 1)
            outputs_coord: (L, B, Q', 4)
            outputs_mask: (L, B, Q', H, W)
            mask_dict: метаданные DN
        returns:
            кортеж чистых (class, coord, mask) без DN-квайери
        """
        pad = mask_dict["pad_size"]
        out_known_class = outputs_class[:, :, :pad, :]
        outputs_class = outputs_class[:, :, pad:, :]
        out_known_coord = outputs_coord[:, :, :pad, :]
        outputs_coord = outputs_coord[:, :, pad:, :]
        out_known_mask = outputs_mask[:, :, :pad, :, :]
        outputs_mask = outputs_mask[:, :, pad:, :, :]
        mask_dict["output_known_lbs_bboxes"] = {
            "pred_logits": out_known_class[-1],
            "pred_boxes": out_known_coord[-1],
            "pred_masks": out_known_mask[-1],
        }
        return outputs_class, outputs_coord, outputs_mask

    def _prediction_heads(self, h: torch.Tensor, mask_features: torch.Tensor):
        """Классификация и маски по нормализованным queries.

        params:
            h: выходы декодера (B, Q, C)
            mask_features: пиксельные эмбеддинги (B, mask_dim, H, W)
        returns:
            кортеж (pred_logits (B, Q, 1), pred_masks (B, Q, H, W))
        """
        decoder_output = self.decoder_norm(h)
        out_class = self.class_embed(decoder_output)
        mask_embed = self.mask_embed(decoder_output)
        out_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)
        return out_class, out_mask

    def forward(
        self, pixel_values: torch.Tensor, targets: list[dict] | None = None
    ) -> dict:
        """Считает скоры, маски и боксы queries.

        params:
            pixel_values: нормализованный батч (B, 3, S, S), S кратно 32
            targets: список целей для DN (labels, boxes, masks) либо None
        returns:
            словарь с pred_logits, pred_masks, pred_boxes, aux_outputs и dn_meta
        """
        b = pixel_values.shape[0]
        mask_features, memory, mem_flat, mem = self._pixel_features(pixel_values)

        interm_outputs: dict | None = None
        if self.two_stage:
            output_memory = self.enc_output_norm(self.enc_output(mem_flat))
            enc_class = self.class_embed(output_memory)
            proposals = self._gen_proposals(mem)
            enc_coord = self.bbox_embed(output_memory) + proposals
            topk = self.num_queries
            topk_idx = torch.topk(enc_class.squeeze(-1), topk, dim=1)[1]
            tgt = torch.gather(
                output_memory, 1, topk_idx.unsqueeze(-1).expand(-1, -1, self.hidden_dim)
            )
            refpoint = torch.gather(
                enc_coord, 1, topk_idx.unsqueeze(-1).expand(-1, -1, 4)
            )
            interm_class, interm_mask = self._prediction_heads(tgt, mask_features)
            interm_outputs = {
                "pred_logits": interm_class,
                "pred_masks": interm_mask,
                "pred_boxes": refpoint.sigmoid(),
            }
            tgt = tgt.detach()
        else:
            tgt = self.query_feat.weight.unsqueeze(0).expand(b, -1, -1)
            refpoint = (
                inverse_sigmoid(self.query_embed.weight).unsqueeze(0).expand(b, -1, -1)
            )

        mask_dict: dict | None = None
        tgt_mask: torch.Tensor | None = None
        if self.dn != "no" and self.training and targets is not None:
            input_query_label, input_query_bbox, dn_mask, mask_dict = (
                self.prepare_for_dn(targets, b)
            )
            if mask_dict is not None:
                tgt = torch.cat([input_query_label, tgt], dim=1)
                refpoint = torch.cat([input_query_bbox, refpoint], dim=1)
                tgt_mask = dn_mask
            else:
                tgt_mask = None

        hs, references = self.decoder(
            tgt=tgt,
            memory=memory,
            refpoints_unsigmoid=refpoint,
            tgt_mask=tgt_mask,
        )

        normed = [self.decoder_norm(h) for h in hs]
        predictions_class = [self.class_embed(h) for h in normed]
        predictions_mask = [
            torch.einsum("bqc,bchw->bqhw", self.mask_embed(h), mask_features)
            for h in normed
        ]
        out_boxes = self.pred_box(references, hs)

        if mask_dict is not None:
            predictions_class = torch.stack(predictions_class)
            predictions_mask = torch.stack(predictions_mask)
            predictions_class, out_boxes, predictions_mask = self.dn_post_process(
                predictions_class, out_boxes, predictions_mask, mask_dict
            )
            predictions_class = list(predictions_class)
            predictions_mask = list(predictions_mask)

        out: dict = {
            "pred_logits": predictions_class[-1],
            "pred_masks": predictions_mask[-1],
            "pred_boxes": out_boxes[-1],
        }
        if self.deep_supervision:
            out["aux_outputs"] = [
                {
                    "pred_logits": predictions_class[i],
                    "pred_masks": predictions_mask[i],
                    "pred_boxes": out_boxes[i],
                }
                for i in range(len(predictions_class) - 1)
            ]
        else:
            out["aux_outputs"] = []
        if interm_outputs is not None:
            out["interm_outputs"] = interm_outputs
        if mask_dict is not None:
            out["dn_meta"] = {
                "output_known": mask_dict["output_known_lbs_bboxes"],
                "known_labels": mask_dict["known_labels"],
                "known_boxes": mask_dict["known_boxes"],
                "known_masks": mask_dict["known_masks"],
                "known_bid": mask_dict["known_bid"],
                "map_known_indice": mask_dict["map_known_indice"],
            }
        else:
            out["dn_meta"] = None
        return out
