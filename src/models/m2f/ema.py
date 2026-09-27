from __future__ import annotations

import torch


class ModelEMA:
    """Экспоненциальное сглаживание весов модели (EMA) с разогревом decay."""    

    def __init__(self, model: torch.nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.updates = 0
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}
        self.backup: dict = {}

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        self.updates += 1
        decay = min(self.decay, (1 + self.updates) / (10 + self.updates))
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(decay).add_(v.detach(), alpha=1 - decay)
            else:
                self.shadow[k] = v.detach().clone()

    def apply_to(self, model: torch.nn.Module) -> None:
        self.backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow, strict=True)

    def restore(self, model: torch.nn.Module) -> None:
        model.load_state_dict(self.backup, strict=True)
        self.backup = {}
