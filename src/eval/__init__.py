from .submission import _write_csv
from .val_items import _val_items

__all__ = ["_val_items", "_write_csv", "cmd_baseline", "cmd_check", "cmd_infer", "cmd_sweep"]


def __getattr__(name: str):
    if name in {"cmd_baseline", "cmd_check", "cmd_infer", "cmd_sweep"}:
        from .evaluate import cmd_baseline, cmd_check, cmd_infer, cmd_sweep

        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")