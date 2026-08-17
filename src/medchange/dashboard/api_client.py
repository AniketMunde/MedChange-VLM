from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests


DEFAULT_API_URL = "http://127.0.0.1:8000"


class MedChangeAPIError(RuntimeError):
    pass


@dataclass(frozen=True)
class APIConfig:
    base_url: str = DEFAULT_API_URL
    timeout_seconds: int = 180


class MedChangeAPIClient:
    def __init__(
        self,
        config: APIConfig | None = None,
    ) -> None:
        self.config = config or APIConfig()

    def _url(
        self,
        path: str,
    ) -> str:
        return (
            self.config.base_url.rstrip("/")
            + "/"
            + path.lstrip("/")
        )

    def health(
        self,
    ) -> dict[str, Any]:
        try:
            response = requests.get(
                self._url("/health"),
                timeout=10,
            )

            response.raise_for_status()

            return response.json()

        except requests.RequestException as exc:
            raise MedChangeAPIError(
                f"Unable to reach MedChange API: {exc}"
            ) from exc

    def model_info(
        self,
    ) -> dict[str, Any]:
        try:
            response = requests.get(
                self._url("/model-info"),
                timeout=10,
            )

            response.raise_for_status()

            return response.json()

        except requests.RequestException as exc:
            raise MedChangeAPIError(
                f"Unable to retrieve model information: {exc}"
            ) from exc

    def analyze_pair(
        self,
        *,
        prior_path: Path,
        current_path: Path,
        pair_id: str,
        prior_study_id: str,
        current_study_id: str,
        safety_policy: str,
        safety_threshold: float,
    ) -> dict[str, Any]:
        try:
            with (
                prior_path.open("rb") as prior_file,
                current_path.open("rb") as current_file,
            ):
                files = {
                    "prior": (
                        prior_path.name,
                        prior_file,
                        "image/png",
                    ),
                    "current": (
                        current_path.name,
                        current_file,
                        "image/png",
                    ),
                }

                data = {
                    "pair_id": pair_id,
                    "prior_study_id": prior_study_id,
                    "current_study_id": current_study_id,
                    "safety_policy": safety_policy,
                    "safety_threshold": str(
                        safety_threshold
                    ),
                }

                response = requests.post(
                    self._url("/analyze-pair"),
                    files=files,
                    data=data,
                    timeout=self.config.timeout_seconds,
                )

            if response.status_code >= 400:
                try:
                    detail = response.json().get(
                        "detail",
                        response.text,
                    )
                except Exception:
                    detail = response.text

                raise MedChangeAPIError(
                    f"API returned {response.status_code}: "
                    f"{detail}"
                )

            return response.json()

        except requests.Timeout as exc:
            raise MedChangeAPIError(
                "Inference request timed out."
            ) from exc

        except requests.RequestException as exc:
            raise MedChangeAPIError(
                f"Inference request failed: {exc}"
            ) from exc