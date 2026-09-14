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
                 run_name: str = "", config: Optional[dict] = None, window: int = 20, eval_label: str = "fixed-set",
                 start_step: int = 0):
        self.kind = kind
        self.eval_label = eval_label
        self.total = total_steps
        self.start_step = start_step          # first step of a resumed run (for the ETA)
        self.log_every = max(1, int(log_every))
        self.window = window
        self.start = time.perf_counter()
        self.losses: list[float] = []
        self.evals: list[tuple[int, float]] = []
        self.drift: list[tuple[int, float]] = []
        self.probes: list[dict] = []
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
        done = index - self.start_step
        eta = elapsed / done * (self.total - index) if done > 0 else 0.0
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

    def eval(self, index: int, loss: float, drift: Optional[float] = None):
        """Fixed-set validation loss at step ``index`` (0 = before training).

        ``drift`` is the loss on the fixed regularization crops (base-model scores): it measures how far
        the LoRA has moved the model away from what the base model writes.
        """
        self.evals.append((index, loss))
        if self.writer is not None:
            self.writer.add_scalar("loss/eval_heldout" if self.eval_label == "held-out" else "loss/eval_fixed", loss, index)
        start = self.evals[0][1]
        best = min(v for _, v in self.evals)
        LOG.info("YuE2 %s eval step %d/%d  %s loss %.4f  (start %.4f, best %.4f, change %+.1f%%)",
                 self.kind, index, self.total, self.eval_label, loss, start, best,
                 (loss / start - 1.0) * 100.0 if start else 0.0)
        if drift is not None:
            self.drift.append((index, drift))
            if self.writer is not None:
                self.writer.add_scalar("loss/eval_regularizer", drift, index)
            LOG.info("YuE2 %s eval step %d/%d  regularizer loss %.4f  (start %.4f, change %+.1f%%)",
                     self.kind, index, self.total, drift, self.drift[0][1],
                     (drift / self.drift[0][1] - 1.0) * 100.0 if self.drift[0][1] else 0.0)

    def probe(self, index: int, tokens: int, ended: bool, seconds: float, path: Optional[str] = None):
        """A score generated with the current LoRA at step ``index``: its length and whether it finished."""
        self.probes.append({"step": index, "tokens": tokens, "ended": ended, "seconds": seconds})
        if self.writer is not None:
            self.writer.add_scalar("probe/abc_tokens", tokens, index)
            self.writer.add_scalar("probe/ended", 1.0 if ended else 0.0, index)
        first = self.probes[0]
        verdict = "ended normally" if ended else "HIT THE TOKEN BUDGET without ending (over-trained or album-length scores)"
        LOG.info("YuE2 %s probe step %d/%d  %d ABC tokens in %s, %s%s%s", self.kind, index, self.total, tokens,
                 _fmt_seconds(seconds), verdict,
                 f"  (step {first['step']}: {first['tokens']} tokens)" if len(self.probes) > 1 else "",
                 f"  -> {path}" if path else "")

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
        if len(self.evals) > 1:
            LOG.info("YuE2 %s %s loss: %.4f before training -> %.4f at the end (best %.4f at step %d)", self.kind,
                     self.eval_label, self.evals[0][1], self.evals[-1][1], min(v for _, v in self.evals),
                     min(self.evals, key=lambda e: e[1])[0])
        if len(self.drift) > 1:
            LOG.info("YuE2 %s regularizer loss: %.4f before training -> %.4f at the end", self.kind,
                     self.drift[0][1], self.drift[-1][1])
        if self.probes:
            LOG.info("YuE2 %s probes: %s", self.kind, "; ".join(
                f"step {p['step']}: {p['tokens']} tokens{'' if p['ended'] else ' (budget hit)'}" for p in self.probes))


__all__ = ["TrainMonitor"]
