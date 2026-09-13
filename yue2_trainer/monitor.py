"""Console step/loss reporting and optional TensorBoard logging for the trainers."""
from __future__ import annotations

import datetime as _dt
import json
import logging
import math
import re
import time
from pathlib import Path
from typing import Optional

LOG = logging.getLogger("yue2_trainer")


def _fmt_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


class TrainMonitor:
    """Prints ``step i/N loss ... avg ... lr ... grad ... elapsed ... eta ...`` and mirrors it to TensorBoard.

    ``tensorboard_dir`` is the parent folder; each run gets ``<run_name>_<timestamp>`` under it
    (``tensorboard --logdir <tensorboard_dir>`` shows all runs side by side).
    """

    def __init__(self, kind: str, total_steps: int, log_every: int = 1, tensorboard_dir: Optional[str] = None,
                 run_name: str = "", config: Optional[dict] = None, window: int = 20):
        self.kind = kind
        self.total = total_steps
        self.log_every = max(1, int(log_every))
        self.window = window
        self.start = time.perf_counter()
        self.losses: list[float] = []
        self.writer = None
        self.log_dir = None
        if tensorboard_dir:
            safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", run_name.strip()) or kind
            self.log_dir = Path(tensorboard_dir) / f"{safe}_{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.log_dir.mkdir(parents=True, exist_ok=True)
                self.writer = SummaryWriter(log_dir=str(self.log_dir))
                if config:
                    self.writer.add_text("config", "```\n" + json.dumps(config, indent=1, default=str) + "\n```", 0)
                LOG.info("YuE2 trainer: TensorBoard logging to %s  (tensorboard --logdir \"%s\")", self.log_dir,
                         Path(tensorboard_dir))
            except ImportError:
                LOG.warning("YuE2 trainer: tensorboard is not installed (pip install tensorboard); console logging only")
                self.writer = None

    def step(self, index: int, loss: float, lr: float, grad_norm: Optional[float] = None,
             extra: Optional[dict] = None):
        """``index`` is 1-based."""
        self.losses.append(loss)
        avg = sum(self.losses[-self.window:]) / len(self.losses[-self.window:])
        elapsed = time.perf_counter() - self.start
        eta = elapsed / index * (self.total - index) if index else 0.0
        if self.writer is not None:
            self.writer.add_scalar("loss/step", loss, index)
            self.writer.add_scalar(f"loss/avg{self.window}", avg, index)
            self.writer.add_scalar("lr", lr, index)
            if grad_norm is not None and math.isfinite(grad_norm):
                self.writer.add_scalar("grad_norm", grad_norm, index)
            for key, value in (extra or {}).items():
                self.writer.add_scalar(key, value, index)
        if index % self.log_every == 0 or index == 1 or index == self.total:
            grad = f" grad {grad_norm:.3f}" if grad_norm is not None and math.isfinite(grad_norm) else ""
            LOG.info("YuE2 %s step %d/%d  loss %.4f  avg%d %.4f  lr %.2e%s  elapsed %s  eta %s",
                     self.kind, index, self.total, loss, self.window, avg, lr, grad,
                     _fmt_seconds(elapsed), _fmt_seconds(eta))

    def close(self, info: Optional[dict] = None):
        if self.writer is not None:
            if info:
                self.writer.add_text("result", "```\n" + json.dumps(info, indent=1, default=str) + "\n```", 0)
            self.writer.flush()
            self.writer.close()
            self.writer = None
        if self.losses:
            LOG.info("YuE2 %s finished: %d steps in %s, first loss %.4f, last %.4f, min %.4f", self.kind,
                     len(self.losses), _fmt_seconds(time.perf_counter() - self.start),
                     self.losses[0], self.losses[-1], min(self.losses))


__all__ = ["TrainMonitor"]
