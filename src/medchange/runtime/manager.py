from __future__ import annotations

import threading
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeStatus:
    busy: bool
    total_requests: int
    successful_requests: int
    failed_requests: int
    cache_hits: int
    active_request_id: str | None


class RuntimeBusyError(
    RuntimeError
):
    pass


class InferenceRuntimeManager:
    def __init__(
        self,
        lock_timeout_seconds: float = 2.0,
    ) -> None:
        self.lock_timeout_seconds = (
            lock_timeout_seconds
        )

        self._inference_lock = (
            threading.Lock()
        )

        self._state_lock = (
            threading.RLock()
        )

        self._busy = False
        self._active_request_id = None

        self._total_requests = 0
        self._successful_requests = 0
        self._failed_requests = 0
        self._cache_hits = 0

    def record_cache_hit(
        self,
    ) -> None:
        with self._state_lock:
            self._cache_hits += 1

    def run(
        self,
        request_id: str,
        function,
        *args,
        **kwargs,
    ):
        with self._state_lock:
            self._total_requests += 1

        acquired = (
            self._inference_lock.acquire(
                timeout=(
                    self.lock_timeout_seconds
                )
            )
        )

        if not acquired:
            with self._state_lock:
                self._failed_requests += 1

            raise RuntimeBusyError(
                "MedChange inference engine is busy."
            )

        try:
            with self._state_lock:
                self._busy = True
                self._active_request_id = (
                    request_id
                )

            result = function(
                *args,
                **kwargs,
            )

            with self._state_lock:
                self._successful_requests += 1

            return result

        except Exception:
            with self._state_lock:
                self._failed_requests += 1

            raise

        finally:
            with self._state_lock:
                self._busy = False
                self._active_request_id = None

            self._inference_lock.release()

    def status(
        self,
    ) -> RuntimeStatus:
        with self._state_lock:
            return RuntimeStatus(
                busy=(
                    self._busy
                ),

                total_requests=(
                    self._total_requests
                ),

                successful_requests=(
                    self._successful_requests
                ),

                failed_requests=(
                    self._failed_requests
                ),

                cache_hits=(
                    self._cache_hits
                ),

                active_request_id=(
                    self._active_request_id
                ),
            )