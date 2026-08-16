from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


class EmbeddingCache:
    """
    Disk-backed BiomedCLIP embedding cache.
    """

    def __init__(
        self,
        cache_dir: str | Path,
    ) -> None:

        self.cache_dir = Path(
            cache_dir
        )

        self.cache_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    def _path(
        self,
        image_index: str,
    ) -> Path:

        safe_name = (
            Path(
                image_index
            ).stem
        )

        return (
            self.cache_dir
            / f"{safe_name}.npy"
        )

    def contains(
        self,
        image_index: str,
    ) -> bool:

        return self._path(
            image_index
        ).exists()

    def load(
        self,
        image_index: str,
    ) -> torch.Tensor:

        path = self._path(
            image_index
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Cached embedding missing: "
                f"{image_index}"
            )

        array = np.load(
            path
        )

        return torch.from_numpy(
            array
        ).unsqueeze(0)

    def save(
        self,
        image_index: str,
        embedding: torch.Tensor,
    ) -> None:

        array = (
            embedding
            .detach()
            .float()
            .cpu()
            .numpy()
            .reshape(-1)
        )

        np.save(
            self._path(
                image_index
            ),
            array,
        )