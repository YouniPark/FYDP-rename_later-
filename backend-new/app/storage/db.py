import json
from datetime import datetime
from pathlib import Path
from typing import Any

from app.storage.models import CueRecord, FaceRecord


class LocalDB:
    def __init__(self, data_dir: str, face_images_dir: str, cue_images_dir: str) -> None:
        self.data_dir = Path(data_dir)
        self.face_images_dir = Path(face_images_dir)
        self.cue_images_dir = Path(cue_images_dir)
        self.face_manifest_path = self.data_dir / "face_db.json"
        self.cue_manifest_path = self.data_dir / "cue_db.json"

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.face_images_dir.mkdir(parents=True, exist_ok=True)
        self.cue_images_dir.mkdir(parents=True, exist_ok=True)

    def load_face_db(self) -> dict[str, FaceRecord]:
        self.ensure_dirs()
        if not self.face_manifest_path.exists():
            return {}
        payload = json.loads(self.face_manifest_path.read_text(encoding="utf-8"))
        return {record["face_id"]: FaceRecord.model_validate(record) for record in payload.get("faces", [])}

    def load_cue_db(self) -> dict[str, CueRecord]:
        self.ensure_dirs()
        if not self.cue_manifest_path.exists():
            return {}
        payload = json.loads(self.cue_manifest_path.read_text(encoding="utf-8"))
        return {record["face_id"]: CueRecord.model_validate(record) for record in payload.get("cues", [])}

    def save_face_db(self, face_db: dict[str, FaceRecord]) -> None:
        payload = {"faces": [rec.model_dump(mode="json") for rec in face_db.values()]}
        self.face_manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def save_cue_db(self, cue_db: dict[str, CueRecord]) -> None:
        payload = {"cues": [rec.model_dump(mode="json") for rec in cue_db.values()]}
        self.cue_manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def store_face_image(self, face_id: str, image_bytes: bytes, ext: str = "jpg") -> str:
        self.ensure_dirs()
        suffix = ext.lstrip(".").lower() or "jpg"
        relative_path = f"faces/{face_id}_{int(datetime.utcnow().timestamp() * 1000)}.{suffix}"
        target = self.data_dir / relative_path
        target.write_bytes(image_bytes)
        return relative_path

    def resolve_data_file(self, relative_path: str) -> Path:
        target = (self.data_dir / relative_path).resolve()
        if self.data_dir.resolve() not in target.parents and target != self.data_dir.resolve():
            raise ValueError("Path traversal rejected")
        return target

    def upsert_face_record(self, face_db: dict[str, FaceRecord], face_id: str, metadata: dict[str, Any], image_path: str | None) -> FaceRecord:
        record = FaceRecord(face_id=face_id, metadata=metadata, image_path=image_path, updated_at=datetime.utcnow())
        face_db[face_id] = record
        self.save_face_db(face_db)
        return record

    def upsert_cue_record(self, cue_db: dict[str, CueRecord], face_id: str, cue: dict[str, Any]) -> CueRecord:
        record = CueRecord(face_id=face_id, cue=cue, updated_at=datetime.utcnow())
        cue_db[face_id] = record
        self.save_cue_db(cue_db)
        return record
