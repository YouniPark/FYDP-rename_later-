from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Any


@dataclass
class TimestampedValue:
    timestamp: float
    value: Any


class AppStateCache:
    def __init__(self) -> None:
        self._lock = Lock()
        self.latest_classifier_result: TimestampedValue | None = None
        self.latest_face_result: TimestampedValue | None = None
        self.latest_cue_json: TimestampedValue | None = None

    def set_classifier(self, timestamp: float, result: bool) -> None:
        with self._lock:
            self.latest_classifier_result = TimestampedValue(timestamp=timestamp, value=result)

    def set_face_result(self, timestamp: float, result: dict[str, Any]) -> None:
        with self._lock:
            self.latest_face_result = TimestampedValue(timestamp=timestamp, value=result)

    def set_cue_json(self, timestamp: float, cue_json: dict[str, Any]) -> None:
        with self._lock:
            self.latest_cue_json = TimestampedValue(timestamp=timestamp, value=cue_json)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "classifier": self.latest_classifier_result,
                "face": self.latest_face_result,
                "cue": self.latest_cue_json,
            }
