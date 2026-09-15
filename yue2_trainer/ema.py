"""Exponential moving average of the trainable LoRA weights (``ema_decay``).

The average is warm-started the usual way: the t-th update uses decay_t = min(decay, (1 + t) / (10 + t)), so the
zero-effect initial weights are not frozen into it. Over the first steps it is close to a plain average and it
reaches the nominal decay after about 10 / (1 - decay) updates. Evaluation, probes, samples, checkpoints and the
LoRA a run returns use the averaged weights; training itself continues on the live ones, which are returned as
well so the two can be compared.
"""
from __future__ import annotations

import contextlib
from typing import Optional

import torch


class WeightAverage:
    def __init__(self, params, decay: float):
        self.params = list(params)
        self.decay = float(decay)
        self.updates = 0
        self.shadow = [p.detach().to(torch.float32).clone() for p in self.params]
        self._live = None       # the live weights while the average is applied

    def next_decay(self) -> float:
        """Decay the next update applies (warm start)."""
        return min(self.decay, (1.0 + self.updates) / (10.0 + self.updates))

    @torch.no_grad()
    def update(self):
        decay = self.next_decay()
        self.updates += 1
        for shadow, param in zip(self.shadow, self.params):
            shadow.mul_(decay).add_(param.detach().to(torch.float32), alpha=1.0 - decay)

    @contextlib.contextmanager
    def applied(self):
        """The averaged weights in the parameters for the duration (evaluation, probes, samples, saving)."""
        live = [param.detach().clone() for param in self.params]
        with torch.no_grad():
            for param, shadow in zip(self.params, self.shadow):
                param.copy_(shadow.to(param.dtype))
        self._live = live
        try:
            yield
        finally:
            self._live = None
            with torch.no_grad():
                for param, data in zip(self.params, live):
                    param.copy_(data)

    def state(self) -> dict:
        """Resume state (CPU tensors): the average and the live weights, so a resumed run continues both.
        Correct inside ``applied()`` too (checkpoints and best-eval snapshots are taken there)."""
        live = self._live if self._live is not None else self.params
        return {"decay": self.decay, "updates": self.updates,
                "shadow": [s.detach().to("cpu") for s in self.shadow],
                "live": [p.detach().to("cpu", torch.float32) for p in live]}

    def load(self, state: Optional[dict]) -> bool:
        """Restore ``state``. False (the average then restarts from the current weights) when it does not fit."""
        if not state or len(state.get("shadow") or []) != len(self.shadow):
            return False
        if any(tuple(s.shape) != tuple(p.shape) for s, p in zip(state["shadow"], self.shadow)):
            return False
        with torch.no_grad():
            for shadow, saved in zip(self.shadow, state["shadow"]):
                shadow.copy_(saved.to(shadow.device, torch.float32))
            live = state.get("live") or []
            if len(live) == len(self.params):
                for param, saved in zip(self.params, live):
                    param.copy_(saved.to(param.device, param.dtype))
        self.updates = int(state.get("updates", 0))
        return True


def averaged(average: Optional[WeightAverage]):
    """Context that applies ``average`` when there is one."""
    return average.applied() if average is not None else contextlib.nullcontext()


__all__ = ["WeightAverage", "averaged"]
