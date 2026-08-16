from __future__ import annotations

from pathlib import Path


class NIHImageResolver:
    """
    Resolve NIH Image Index filenames inside a locally
    downloaded NIH image dataset.

    The index is built once so repeated temporal pair
    lookups remain efficient.
    """

    def __init__(
        self,
        dataset_root: str | Path,
    ) -> None:

        self.dataset_root = Path(
            dataset_root
        )

        if not self.dataset_root.exists():
            raise FileNotFoundError(
                f"NIH dataset root does not exist: "
                f"{self.dataset_root}"
            )

        self._index: dict[
            str,
            Path,
        ] = {}

    def build_index(
        self,
    ) -> None:
        """
        Recursively index all PNG/JPG/JPEG files.
        """

        supported = {
            ".png",
            ".jpg",
            ".jpeg",
        }

        index: dict[
            str,
            Path,
        ] = {}

        for path in (
            self.dataset_root.rglob(
                "*"
            )
        ):
            if (
                not path.is_file()
                or path.suffix.lower()
                not in supported
            ):
                continue

            name = (
                path.name.lower()
            )

            if name in index:
                raise ValueError(
                    "Duplicate NIH image filename "
                    f"detected: {path.name}"
                )

            index[
                name
            ] = path

        if not index:
            raise ValueError(
                "No image files were found inside "
                f"{self.dataset_root}"
            )

        self._index = index

    @property
    def num_images(
        self,
    ) -> int:
        return len(
            self._index
        )

    def resolve(
        self,
        image_index: str,
    ) -> Path:

        if not self._index:
            self.build_index()

        key = (
            str(
                image_index
            )
            .strip()
            .lower()
        )

        path = (
            self._index.get(
                key
            )
        )

        if path is None:
            raise FileNotFoundError(
                "NIH image was not found: "
                f"{image_index}"
            )

        return path

    def contains(
        self,
        image_index: str,
    ) -> bool:

        if not self._index:
            self.build_index()

        return (
            str(
                image_index
            )
            .strip()
            .lower()
            in self._index
        )