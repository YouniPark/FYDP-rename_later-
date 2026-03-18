import json
from datetime import datetime
from pathlib import Path
from typing import Any

from app.storage.models import CueRecord, FaceRecord


class LocalDB:
    def __init__(
        self,
        data_dir: str,
        people_json_path: Path,
        images_dir: Path,
        headshots_dir: Path,
        auditory_cue_dir: Path,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.people_json_path = Path(people_json_path)
        self.images_dir = Path(images_dir)
        self.headshots_dir = Path(headshots_dir)
        self.auditory_cue_dir = Path(auditory_cue_dir)

        ## backups
        self.face_manifest_path = self.data_dir / "face_db.json"
        self.cue_manifest_path = self.data_dir / "cue_db.json"

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.people_json_path.parent.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.headshots_dir.mkdir(parents=True, exist_ok=True)
        self.auditory_cue_dir.mkdir(parents=True, exist_ok=True)

    def _read_people(self) -> list[dict[str, Any]]:
        self.ensure_dirs()

        if not self.people_json_path.exists():
            return []
        payload = json.loads(self.people_json_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("people.json must be a JSON array")
        return [entry for entry in payload if isinstance(entry, dict)]

    def _write_people(self, people: list[dict[str, Any]]) -> None:
        self.ensure_dirs()
        self.people_json_path.write_text(json.dumps(people, indent=2), encoding="utf-8")

    @staticmethod
    def _face_id_to_person_id(face_id: str) -> int | str:
        try:
            return int(face_id)
        except ValueError:
            return face_id

    @staticmethod
    def _person_id_to_face_id(person_id: Any) -> str:
        return str(person_id)

    @staticmethod
    def _updated_at(entry: dict[str, Any]) -> datetime:
        created_at = entry.get("created_at")
        if isinstance(created_at, (int, float)):
            try:
                return datetime.utcfromtimestamp(created_at)
            except (OverflowError, OSError, ValueError):
                return datetime.utcnow()
        return datetime.utcnow()

    def _find_person(self, people: list[dict[str, Any]], face_id: str) -> tuple[int, dict[str, Any] | None]:
        person_id = self._face_id_to_person_id(face_id)
        for idx, person in enumerate(people):
            if person.get("id") == person_id:
                return idx, person
        return -1, None

    def load_face_db(self) -> dict[str, FaceRecord]:
        people = self._read_people()
        if people:
            records: dict[str, FaceRecord] = {}
            for person in people:
                face_id = self._person_id_to_face_id(person.get("id"))
                if not face_id:
                    continue
                metadata = {
                    "name": person.get("name"),
                    "relationship": person.get("relationship"),
                    "headshot": person.get("headshot"),
                    "audio_status": person.get("audio_status"),
                    "audio_error": person.get("audio_error"),
                    "created_at": person.get("created_at"),
                }
                records[face_id] = FaceRecord(
                    face_id=face_id,
                    metadata={k: v for k, v in metadata.items() if v is not None},
                    image_path=person.get("image"),
                    updated_at=self._updated_at(person),
                )
            return records

        # backup
        if not self.face_manifest_path.exists():
            return {}
        payload = json.loads(self.face_manifest_path.read_text(encoding="utf-8"))
        return {record["face_id"]: FaceRecord.model_validate(record) for record in payload.get("faces", [])}

    def load_cue_db(self) -> dict[str, CueRecord]:
        people = self._read_people()
        if people:
            records: dict[str, CueRecord] = {}
            for person in people:
                face_id = self._person_id_to_face_id(person.get("id"))
                if not face_id:
                    continue
                cue_payload = {
                    "auditory cue (male)": person.get("auditory cue (male)"),
                    "auditory cue (female)": person.get("auditory cue (female)"),
                    "audio_status": person.get("audio_status"),
                    "audio_error": person.get("audio_error"),
                    "headshot": person.get("headshot"),
                    "image": person.get("image"),
                    "name": person.get("name"),
                    "relationship": person.get("relationship"),
                    "created_at": person.get("created_at"),
                }
                records[face_id] = CueRecord(
                    face_id=face_id,
                    cue={k: v for k, v in cue_payload.items() if v is not None},
                    updated_at=self._updated_at(person),
                )
            return records

        # fallback
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
        filename = f"{face_id}_{int(datetime.utcnow().timestamp() * 1000)}.{suffix}"
        target = self.images_dir / filename
        target.write_bytes(image_bytes)
        return f"images/{filename}"

    def resolve_data_file(self, relative_path: str) -> Path:
        # Resolve against PeopleDatabase to support values from people.json
        base_dir = self.people_json_path.parent.resolve()
        target = (base_dir / relative_path).resolve()
        if base_dir not in target.parents and target != base_dir:
            raise ValueError("Path traversal rejected")
        return target

    def upsert_face_record(self, face_db: dict[str, FaceRecord], face_id: str, metadata: dict[str, Any], image_path: str | None) -> FaceRecord:
        record = FaceRecord(face_id=face_id, metadata=metadata, image_path=image_path, updated_at=datetime.utcnow())
        face_db[face_id] = record
        people = self._read_people()
        person_id = self._face_id_to_person_id(face_id)
        idx, person = self._find_person(people, face_id)
        if person is None:
            person = {"id": person_id, "created_at": int(datetime.utcnow().timestamp())}
            people.append(person)
            idx = len(people) - 1

        people[idx] = {
            **person,
            "id": person_id,
            "image": image_path if image_path is not None else person.get("image"),
            "name": metadata.get("name", person.get("name")),
            "relationship": metadata.get("relationship", person.get("relationship")),
            "headshot": metadata.get("headshot", person.get("headshot")),
            "audio_status": metadata.get("audio_status", person.get("audio_status")),
            "audio_error": metadata.get("audio_error", person.get("audio_error")),
            "created_at": person.get("created_at", int(datetime.utcnow().timestamp())),
        }
        self._write_people(people)
        return record

    def upsert_cue_record(self, cue_db: dict[str, CueRecord], face_id: str, cue: dict[str, Any]) -> CueRecord:
        record = CueRecord(face_id=face_id, cue=cue, updated_at=datetime.utcnow())
        cue_db[face_id] = record
        
        people = self._read_people()
        person_id = self._face_id_to_person_id(face_id)
        idx, person = self._find_person(people, face_id)
        if person is None:
            person = {"id": person_id, "created_at": int(datetime.utcnow().timestamp())}
            people.append(person)
            idx = len(people) - 1

        merged = {
            **person,
            "id": person_id,
            "auditory cue (male)": cue.get("auditory cue (male)", person.get("auditory cue (male)")),
            "auditory cue (female)": cue.get("auditory cue (female)", person.get("auditory cue (female)")),
            "audio_status": cue.get("audio_status", person.get("audio_status")),
            "audio_error": cue.get("audio_error", person.get("audio_error")),
            "created_at": person.get("created_at", int(datetime.utcnow().timestamp())),
        }

        # Preserve useful person-level fields if user sends them via cue payload.
        for key in ("name", "relationship", "image", "headshot"):
            merged[key] = cue.get(key, person.get(key))

        people[idx] = merged
        self._write_people(people)

        return record
