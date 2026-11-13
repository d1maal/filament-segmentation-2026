from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class DoubleConv(nn.Module):
    """Двойной свёрточный блок 3×3: Conv-BN-ReLU, Conv-BN-ReLU."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        """Создание блока двойной свёртки.
        params:
            in_ch: число входных каналов
            out_ch: число выходных каналов
        """
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Прямой проход через двойную свёртку.
        params:
            x: тензор (B, in_ch, H, W)
        returns:
            тензор (B, out_ch, H, W)
        """
        return self.block(x)


class Down(nn.Module):
    """Стадия энкодера: max-pool 2×2 и двойная свёртка."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        """Создание стадии downsampling.
        params:
            in_ch: число входных каналов
            out_ch: число выходных каналов
        """
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Сжатие входа в 2 раза и применяет двойную свёртку.
        params:
            x: тензор (B, in_ch, H, W)
        returns:
            тензор (B, out_ch, H/2, W/2)
        """
        return self.conv(self.pool(x))


class Up(nn.Module):
    """Стадия декодера: transposed conv, конкатенация skip и двойная свёртка."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        """Создание стадии upsampling.
        params:
            in_ch: число каналов входа (выход предыдущего блока декодера)
            out_ch: число каналов после upsampling и на выходе блока
        """
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_ch * 2, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Поднятие разрешения, добавление skip-соединений.
        params:
            x: тензор с выхода предыдущего блока декодера
            skip: тензор из соответствующей стадии энкодера
        returns:
            тензор
        """
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(
                x, size=skip.shape[-2:], mode="bilinear", align_corners=False
            )
        return self.conv(torch.cat([skip, x], dim=1))


class EdgePrior(nn.Module):
    """Выучиваемая карта краёв: Conv3×3 + sigmoid, затем проекция в каналы bottleneck."""

    def __init__(self, in_ch: int, channels: int) -> None:
        """Создание ветки edge-приора.
        params:
            in_ch: число каналов входного изображения
            channels: число каналов bottleneck, в которые проецируется карта
        """
        super().__init__()
        self.edge = nn.Conv2d(in_ch, 1, kernel_size=3, padding=1)
        self.proj = nn.Conv2d(1, channels, kernel_size=1)

    def forward(self, x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        """Подсчет карты краёв и приведение её к разрешению и каналам bottleneck.
        params:
            x: входное изображение
            size: целевое разрешение bottleneck
        returns:
            тензор
        """
        edge = torch.sigmoid(self.edge(x))
        edge = F.interpolate(edge, size=size, mode="bilinear", align_corners=False)
        return self.proj(edge)


class EGAttention(nn.Module):
    """Edge-guided multi-head self-attention (EG-MHSA)."""    

    def __init__(self, channels: int, heads: int = 4, dropout: float = 0.1) -> None:
        """Создание блока EG-MHSA.
        params:
            channels: число каналов входа
            heads: число голов attention
            dropout: вероятность dropout после residual-сложения
        """
        super().__init__()
        if channels % heads:
            raise ValueError(f"channels={channels} must be divisible by heads={heads}")
        self.channels = channels
        self.heads = heads
        self.dk = channels // heads
        self.scale = self.dk**-0.5
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.norm = nn.LayerNorm(channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_bias: torch.Tensor) -> torch.Tensor:
        """Подсчет self-attention с edge-смещением в Q и K.
        params:
            x: карта признаков
            edge_bias: edge-приор
        returns:
            карта признаков
        """
        if edge_bias.shape != x.shape:
            raise ValueError(
                f"edge_bias {tuple(edge_bias.shape)} != x {tuple(x.shape)}"
            )
        b, c, h, w = x.shape
        n = h * w
        q, k, v = self.qkv(x).chunk(3, dim=1)
        q = q + edge_bias
        k = k + edge_bias
        q = q.reshape(b, self.heads, self.dk, n).permute(0, 1, 3, 2)
        k = k.reshape(b, self.heads, self.dk, n).permute(0, 1, 3, 2)
        v = v.reshape(b, self.heads, self.dk, n).permute(0, 1, 3, 2)
        attn = torch.softmax((q @ k.transpose(-2, -1)) * self.scale, dim=-1)
        out = (attn @ v).permute(0, 1, 3, 2).reshape(b, c, h, w)
        out = self.drop(self.proj(out))
        out = (x + out).flatten(2).transpose(1, 2)
        return self.norm(out).transpose(1, 2).reshape(b, c, h, w)


class EdgeBottleneck(nn.Module):
    """Боттлнек EdgeAttNet: двойная свёртка и EG-MHSA блоки."""

    def __init__(
        self, channels: int, blocks: int = 2, heads: int = 4, dropout: float = 0.1
    ) -> None:
        """Создание боттлнек с EG-MHSA.
        params:
            channels: число каналов признаков на bottleneck
            blocks: число последовательных EG-MHSA блоков
            heads: число голов attention
            dropout: вероятность dropout в EG-MHSA
        """
        super().__init__()
        self.conv = DoubleConv(channels, channels)
        self.attn = nn.ModuleList(
            EGAttention(channels=channels, heads=heads, dropout=dropout)
            for _ in range(blocks)
        )

    def forward(self, x: torch.Tensor, edge_bias: torch.Tensor) -> torch.Tensor:
        """Применеие двойной свёртки и EG-MHSA блоки.
        params:
            x: карта признаков 
            edge_bias: edge-приор
        returns:
            карта признаков
        """
        x = self.conv(x)
        for block in self.attn:
            x = block(x, edge_bias)
        return x
