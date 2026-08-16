from __future__ import annotations

import time
from dataclasses import dataclass

import torch


@dataclass
class InferenceMetrics:
    elapsed_seconds: float
    gpu_allocated_gb: float | None = None
    gpu_reserved_gb: float | None = None
    gpu_peak_allocated_gb: float | None = None


class InferenceTimer:
    def __init__(self) -> None:
        self._start_time: float | None = None

    def start(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        self._start_time = time.perf_counter()

    def stop(self) -> InferenceMetrics:
        if self._start_time is None:
            raise RuntimeError("InferenceTimer.start() must be called first.")

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - self._start_time

        if not torch.cuda.is_available():
            return InferenceMetrics(
                elapsed_seconds=elapsed,
            )

        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        peak = torch.cuda.max_memory_allocated() / 1024**3

        return InferenceMetrics(
            elapsed_seconds=elapsed,
            gpu_allocated_gb=allocated,
            gpu_reserved_gb=reserved,
            gpu_peak_allocated_gb=peak,
        )