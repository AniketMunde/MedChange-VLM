from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from pathlib import Path
from threading import RLock
from typing import Any


class ResultCache:
    def __init__(
        self,
        max_entries: int = 32,
    ) -> None:
        if max_entries <= 0:
            raise ValueError(
                "max_entries must be > 0"
            )

        self.max_entries = (
            max_entries
        )

        self._items: OrderedDict[
            str,
            dict[str, Any],
        ] = OrderedDict()

        self._lock = RLock()

    def get(
        self,
        key: str,
    ) -> dict[str, Any] | None:
        with self._lock:
            value = (
                self._items.get(
                    key
                )
            )

            if value is None:
                return None

            self._items.move_to_end(
                key
            )

            return value

    def set(
        self,
        key: str,
        value: dict[str, Any],
    ) -> None:
        with self._lock:
            self._items[
                key
            ] = value

            self._items.move_to_end(
                key
            )

            while (
                len(
                    self._items
                )
                > self.max_entries
            ):
                self._items.popitem(
                    last=False
                )

    def clear(
        self,
    ) -> None:
        with self._lock:
            self._items.clear()

    def __len__(
        self,
    ) -> int:
        with self._lock:
            return len(
                self._items
            )


def file_sha256(
    path: str | Path,
) -> str:
    digest = hashlib.sha256()

    with Path(path).open(
        "rb"
    ) as file:
        while True:
            chunk = file.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


def build_request_cache_key(
    *,
    prior_path: str | Path,
    current_path: str | Path,
    pair_id: str,
    prior_study_id: str,
    current_study_id: str,
    safety_policy: str,
    safety_threshold: float,
) -> str:

    payload = {
        "prior_sha256":
            file_sha256(
                prior_path
            ),

        "current_sha256":
            file_sha256(
                current_path
            ),

        "pair_id":
            pair_id,

        "prior_study_id":
            prior_study_id,

        "current_study_id":
            current_study_id,

        "safety_policy":
            safety_policy,

        "safety_threshold":
            float(
                safety_threshold
            ),
    }

    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
    )

    return hashlib.sha256(
        serialized.encode(
            "utf-8"
        )
    ).hexdigest()