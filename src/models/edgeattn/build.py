from __future__ import annotations

import torch
from torch import nn

from src.models.edgeattn.modules import DoubleConv, EdgeBottleneck, EdgePrior, Up

ARCHS = ("unet", "edgeattn")
DEPTH = 16


class EdgeAttnUNet(nn.Module):
    def __init__(
        self,
        in_ch: int = 3,
        base: int = 64,
        arch: str = "edgeattn",
        blocks: int = 2,
        heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        """Создаёт сеть EdgeAttNet или U-Net.
        params:
            in_ch: число каналов входного изображения
            base: число каналов первой стадии энкодера
            arch: вариант архитектуры (unet или edgeattn)
            blocks: число EG-MHSA блоков на bottleneck
            heads: число голов attention
            dropout: вероятность dropout в EG-MHSA
        """
        super().__init__()
        if arch not in ARCHS:
            raise ValueError(f"Unknown arch: {arch}. Use one of: {', '.join(ARCHS)}")
        self.arch = arch
        c1, c2, c3, c4 = base, base * 2, base * 4, base * 8
        self.enc1 = DoubleConv(in_ch, c1)
        self.enc2 = DoubleConv(c1, c2)
        self.enc3 = DoubleConv(c2, c3)
        self.enc4 = DoubleConv(c3, c4)
        self.pool = nn.MaxPool2d(2)
        self.edge: EdgePrior | None = (
            EdgePrior(in_ch, c4) if arch == "edgeattn" else None
        )
        self.bottleneck: nn.Module = (
            EdgeBottleneck(c4, blocks=blocks, heads=heads, dropout=dropout)
            if arch == "edgeattn"
            else DoubleConv(c4, c4)
        )
        self.up1 = Up(c4, c4)
        self.up2 = Up(c4, c3)
        self.up3 = Up(c3, c2)
        self.up4 = Up(c2, c1)
        self.out = nn.Conv2d(c1, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Подсчет логитов сегментации.
        params:
            x: изображение (B, in_ch, H, W), H и W кратны 16
        returns:
            логиты (B, 1, H, W)
        """
        h, w = x.shape[-2:]
        if h % DEPTH or w % DEPTH:
            raise ValueError(f"Input size {h}x{w} must be divisible by {DEPTH}")
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        s4 = self.enc4(self.pool(s3))
        z = self.pool(s4)
        if self.edge is not None:
            z = self.bottleneck(z, self.edge(x, z.shape[-2:]))
        else:
            z = self.bottleneck(z)
        z = self.up1(z, s4)
        z = self.up2(z, s3)
        z = self.up3(z, s2)
        z = self.up4(z, s1)
        return self.out(z)


def build_arch(
    arch: str = "edgeattn",
    in_ch: int = 3,
    base: int = 64,
    blocks: int = 2,
    heads: int = 4,
    dropout: float = 0.1,
) -> EdgeAttnUNet:
    """Сборка EdgeAttNet или U-Net.
    params:
        arch: вариант архитектуры (unet или edgeattn)
        in_ch: число каналов входного изображения
        base: число каналов первой стадии энкодера
        blocks: число EG-MHSA блоков на bottleneck
        heads: число голов attention
        dropout: вероятность dropout в EG-MHSA
    returns:
        неинициализированная модель
    """
    return EdgeAttnUNet(
        in_ch=in_ch, base=base, arch=arch, blocks=blocks, heads=heads, dropout=dropout
    )


def count_params(model: nn.Module) -> int:
    """Подсчет числа обучаемых параметров модели.
    params:
        model: модель PyTorch
    returns:
        число параметров с requires_grad=True
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
