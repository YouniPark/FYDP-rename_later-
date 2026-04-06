from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal


FaceLabel = str  # Known name or "Unknown". None means no-face and is not stored.


@dataclass
class FaceSample:
    label: FaceLabel
    confidence: float
    observed_ts: float


@dataclass
class TemporalVoteResult:
    label: str | None
    confidence: float | None
    decided_at: datetime
    sample_count: int


class TemporalFaceMemory:
    """Rolling time-window vote. Unknown is counted; no-face is ignored."""

    def __init__(self, window_seconds: float, tie_break_strategy: Literal["most_recent"] = "most_recent"):
        if window_seconds <= 0:
            raise ValueError("window_seconds must be > 0")
        if tie_break_strategy != "most_recent":
            raise ValueError("Only 'most_recent' tie break is supported")

        self.window_seconds = window_seconds
        self.tie_break_strategy = tie_break_strategy
        self._samples: list[FaceSample] = []

    def add_detection(self, label: str | None, confidence: float | None, observed_ts: float) -> TemporalVoteResult:
        # Keep no-face out of the memory window but still return a current decision.
        if label is not None:
            safe_conf = 0.0 if confidence is None else float(max(0.0, min(1.0, confidence)))
            self._samples.append(FaceSample(label=label, confidence=safe_conf, observed_ts=observed_ts))

        self._prune(observed_ts)
        return self.get_vote(now_ts=observed_ts)

    def get_vote(self, now_ts: float | None = None) -> TemporalVoteResult:
        if now_ts is None:
            now_ts = datetime.now(tz=timezone.utc).timestamp()

        self._prune(now_ts)

        if not self._samples:
            return TemporalVoteResult(
                label=None,
                confidence=None,
                decided_at=datetime.now(tz=timezone.utc),
                sample_count=0,
            )

        counts: dict[str, int] = {}
        latest_ts_by_label: dict[str, float] = {}
        conf_sum: dict[str, float] = {}
        conf_count: dict[str, int] = {}

        for sample in self._samples:
            counts[sample.label] = counts.get(sample.label, 0) + 1
            latest_ts_by_label[sample.label] = max(
                latest_ts_by_label.get(sample.label, float("-inf")),
                sample.observed_ts,
            )
            conf_sum[sample.label] = conf_sum.get(sample.label, 0.0) + sample.confidence
            conf_count[sample.label] = conf_count.get(sample.label, 0) + 1

        max_votes = max(counts.values())
        tied_labels = [label for label, count in counts.items() if count == max_votes]

        winner = max(tied_labels, key=lambda label: latest_ts_by_label[label])
        avg_conf = conf_sum[winner] / conf_count[winner]

        return TemporalVoteResult(
            label=winner,
            confidence=avg_conf,
            decided_at=datetime.now(tz=timezone.utc),
            sample_count=len(self._samples),
        )

    def _prune(self, now_ts: float) -> None:
        cutoff = now_ts - self.window_seconds
        self._samples = [sample for sample in self._samples if sample.observed_ts >= cutoff]
