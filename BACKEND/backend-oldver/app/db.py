from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any


class Database:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS face_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    external_id TEXT,
                    payload_json TEXT NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS cue_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key TEXT,
                    payload_json TEXT NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS event_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_lsl_timestamp REAL NOT NULL,
                    classifier_result INTEGER,
                    metadata_json TEXT,
                    created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS recognition_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    result_json TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS versions (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at REAL NOT NULL
                );
                """
            )
            for key in ("faces_version", "cues_version"):
                conn.execute(
                    "INSERT OR IGNORE INTO versions(key, value, updated_at) VALUES(?, ?, ?)",
                    (key, "0", time.time()),
                )

    def _set_version(self, key: str, value: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE versions SET value = ?, updated_at = ? WHERE key = ?",
                (value, time.time(), key),
            )

    def _increment_version(self, key: str) -> str:
        cur = self.get_versions()[key]
        new_value = str(int(cur) + 1)
        self._set_version(key, new_value)
        return new_value

    def get_versions(self) -> dict[str, str]:
        with self._connect() as conn:
            rows = conn.execute("SELECT key, value FROM versions").fetchall()
        return {row["key"]: row["value"] for row in rows}

    def add_or_update_face(self, payload: dict[str, Any], external_id: str | None = None) -> str:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO face_records(external_id, payload_json, updated_at) VALUES (?, ?, ?)",
                (external_id, json.dumps(payload), time.time()),
            )
        return self._increment_version("faces_version")

    def add_or_update_cue(self, payload: dict[str, Any], key: str | None = None) -> str:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO cue_records(key, payload_json, updated_at) VALUES (?, ?, ?)",
                (key, json.dumps(payload), time.time()),
            )
        return self._increment_version("cues_version")

    def list_faces(self) -> tuple[list[dict[str, Any]], float | None, str]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM face_records ORDER BY updated_at DESC").fetchall()
        records = [self._row_payload(row) for row in rows]
        last_updated = max((row["updated_at"] for row in rows), default=None)
        version = self.get_versions().get("faces_version", "0")
        return records, last_updated, version

    def list_cues(self) -> tuple[list[dict[str, Any]], float | None, str]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM cue_records ORDER BY updated_at DESC").fetchall()
        records = [self._row_payload(row) for row in rows]
        last_updated = max((row["updated_at"] for row in rows), default=None)
        version = self.get_versions().get("cues_version", "0")
        return records, last_updated, version

    def insert_event_record(self, event_lsl_timestamp: float, classifier_result: bool | None, metadata: dict[str, Any]) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO event_records(event_lsl_timestamp, classifier_result, metadata_json, created_at) VALUES (?, ?, ?, ?)",
                (event_lsl_timestamp, int(classifier_result) if classifier_result is not None else None, json.dumps(metadata), time.time()),
            )

    def insert_recognition_record(self, timestamp: float, result: dict[str, Any]) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO recognition_records(timestamp, result_json, created_at) VALUES (?, ?, ?)",
                (timestamp, json.dumps(result), time.time()),
            )

    @staticmethod
    def _row_payload(row: sqlite3.Row) -> dict[str, Any]:
        payload = json.loads(row["payload_json"])
        payload["_id"] = row["id"]
        payload["_updated_at"] = row["updated_at"]
        return payload


def snapshot_version_hash(records: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256(json.dumps(records, sort_keys=True).encode("utf-8")).hexdigest()
    return digest[:12]
