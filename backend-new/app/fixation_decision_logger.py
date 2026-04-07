from __future__ import annotations

import asyncio
import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import Settings
from app.face_service.mapping import load_name_to_people_id


class FixationDecisionCsvLogger:
    _fieldnames = [
        "timestamp_utc",
        "lsl_local_time",
        "person_id",
        "person_name",
        "eeg_status",
        "ml_outcome",
        "ml_score",
        "familiarity_verdict",
        "cue_decision",
    ]

    def __init__(self, settings: Settings) -> None:
        self._enabled = settings.save_fixation_decision_log
        self._csv_path = Path(settings.fixation_decision_log_csv_path)
        self._mapping_csv_path = Path(settings.face_id_mapping_csv_path)
        self._lock = asyncio.Lock()
        self._id_to_name: dict[str, str] | None = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _ensure_mapping_loaded(self) -> None:
        if self._id_to_name is not None:
            return

        try:
            name_to_id = load_name_to_people_id(self._mapping_csv_path)
            self._id_to_name = {str(people_id): name for name, people_id in name_to_id.items()}
        except Exception:
            self._id_to_name = {}

    def _resolve_name(self, face_id: str | None) -> str:
        if face_id is None:
            return "UNKNOWN"
        self._ensure_mapping_loaded()
        assert self._id_to_name is not None
        return self._id_to_name.get(str(face_id), "UNKNOWN")

    @staticmethod
    def _to_eeg_status(eeg_result: dict[str, Any]) -> str:
        status = str(eeg_result.get("status", "")).strip().lower()
        reason = str(eeg_result.get("reason", "")).strip().lower()
        if status == "rejected":
            return "rejected based on PTP amplitude"
        if status == "out_of_buffer_range" or reason == "insufficient_eeg_buffer_history":
            return "out of buffer range"
        if status == "no_eeg" or reason == "eeg_stream_not_connected":
            return "disconnected"
        return "processed"

    @staticmethod
    def _to_ml_outcome(eeg_result: dict[str, Any]) -> str:
        if not eeg_result.get("ml_analyzed", False):
            return ""
        is_unfamiliar = eeg_result.get("is_unfamiliar")
        if isinstance(is_unfamiliar, bool):
            return "Unfamiliar" if is_unfamiliar else "Familiar"
        return ""

    @staticmethod
    def _to_ml_score(eeg_result: dict[str, Any]) -> str:
        if not eeg_result.get("ml_analyzed", False):
            return ""
        score = eeg_result.get("ml_score")
        threshold = eeg_result.get("ml_threshold")
        if isinstance(score, (int, float)) and isinstance(threshold, (int, float)):
            if threshold != 0:
                return f"{float(score):.6f} vs {float(threshold):.6f}"
            else:
                return f"{float(score):.6f}"
        return ""

    @staticmethod
    def _to_unfamiliar_verdict(eeg_result: dict[str, Any]) -> str:
        verdict = eeg_result.get("is_unfamiliar")
        if isinstance(verdict, bool):
            return "Unfamiliar" if verdict else "Familiar"
        return ""

    @staticmethod
    def _to_cue_decision(send_cue: bool) -> str:
        return "send" if send_cue else "do not send"

    @staticmethod
    def _current_lsl_local_time() -> str:
        try:
            from pylsl import local_clock  # type: ignore

            return f"{float(local_clock()):.6f}"
        except Exception:
            return ""

    def _append_row_sync(self, row: dict[str, str]) -> None:
        self._csv_path.parent.mkdir(parents=True, exist_ok=True)
        has_header = self._csv_path.exists() and self._csv_path.stat().st_size > 0

        with self._csv_path.open("a", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=self._fieldnames)
            if not has_header:
                writer.writeheader()
            writer.writerow(row)

    async def log_event(
        self,
        *,
        face_id: str | None,
        eeg_result: dict[str, Any],
        send_cue: bool,
    ) -> None:
        if not self._enabled:
            return

        row = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "lsl_local_time": self._current_lsl_local_time(),
            "person_id": str(face_id) if face_id is not None else "",
            "person_name": self._resolve_name(face_id),
            "eeg_status": self._to_eeg_status(eeg_result),
            "ml_outcome": self._to_ml_outcome(eeg_result),
            "ml_score": self._to_ml_score(eeg_result),
            "familiarity_verdict": self._to_unfamiliar_verdict(eeg_result),
            "cue_decision": self._to_cue_decision(send_cue),
        }

        async with self._lock:
            await asyncio.to_thread(self._append_row_sync, row)
