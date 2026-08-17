from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class APIErrorPayload:
    code: str
    message: str
    retryable: bool = False


class MedChangeServiceError(RuntimeError):
    def __init__(
        self,
        *,
        code: str,
        message: str,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)

        self.code = code
        self.message = message
        self.retryable = retryable