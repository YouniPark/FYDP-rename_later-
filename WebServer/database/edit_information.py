from __future__ import annotations

import json
import time
from pathlib import Path
from threading import Lock

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse

from WebServer.database.add_new_person import (
    DEFAULT_DB_DIR,
    ensure_db_layout,
    generate_and_store_auditory_cues,
)

router = APIRouter()

people_file_lock = Lock()


def _people_json_path() -> Path:
    return DEFAULT_DB_DIR / "people.json"


def load_people() -> list[dict]:
    ensure_db_layout(DEFAULT_DB_DIR)
    try:
        with people_file_lock:
            return json.loads(_people_json_path().read_text(encoding="utf-8"))
    except Exception:
        return []


def save_people(people: list[dict]) -> None:
    with people_file_lock:
        _people_json_path().write_text(json.dumps(people, indent=2), encoding="utf-8")


def public_image_url(image_rel: str | None) -> str | None:
    if not image_rel:
        return None
    return "/localdb/" + image_rel.replace("\\", "/")


def _resolve_db_path(path_str: str | None) -> Path | None:
    if not path_str:
        return None

    p = Path(path_str)

    if p.is_absolute():
        return p

    return DEFAULT_DB_DIR / p


def _delete_file_if_exists(path_str: str | None) -> None:
    p = _resolve_db_path(path_str)
    if p is None:
        return

    try:
        if p.exists() and p.is_file():
            p.unlink()
    except Exception:
        pass


def _clear_and_delete_audio_files(person: dict) -> None:
    _delete_file_if_exists(person.get("auditory cue (male)"))
    _delete_file_if_exists(person.get("auditory cue (female)"))

    person["auditory cue (male)"] = None
    person["auditory cue (female)"] = None
    person["audio_error"] = None


@router.post("/database/update")
async def update_person(
    background_tasks: BackgroundTasks,
    person_id: int = Form(...),
    name: str = Form(...),
    relationship: str = Form(...),
    image: UploadFile | None = File(default=None),
):
    name = name.strip()
    relationship = relationship.strip()

    if not name:
        raise HTTPException(status_code=400, detail="Name is required.")
    if not relationship:
        raise HTTPException(status_code=400, detail="Relationship is required.")

    people = load_people()

    idx = None
    for i, p in enumerate(people):
        try:
            if int(p.get("id", -1)) == int(person_id):
                idx = i
                break
        except Exception:
            continue

    if idx is None:
        raise HTTPException(status_code=404, detail="Person not found.")

    person = people[idx]

    old_name = str(person.get("name", "")).strip()
    old_relationship = str(person.get("relationship", "")).strip()

    name_changed = old_name != name
    relationship_changed = old_relationship != relationship
    should_regenerate_audio = name_changed or relationship_changed

    # Update text fields
    person["name"] = name
    person["relationship"] = relationship

    # Update image if a new one is uploaded
    if image is not None:
        ext = Path(image.filename or "").suffix or ".png"
        ts = int(time.time())
        filename = f"{person_id}_{ts}{ext}"

        img_path = DEFAULT_DB_DIR / "images" / filename
        img_path.write_bytes(await image.read())

        # Delete old image file
        old_image = person.get("image")
        person["image"] = str(Path("images") / filename)

        if old_image and old_image != person["image"]:
            _delete_file_if_exists(old_image)

    # If name or relationship changed, remove old audio and regenerate
    if should_regenerate_audio:
        _clear_and_delete_audio_files(person)
        person["audio_status"] = "pending"

        ts = int(time.time())

        save_people(people)

        background_tasks.add_task(
            generate_and_store_auditory_cues,
            int(person_id),
            name,
            relationship,
            ts,
            DEFAULT_DB_DIR,
        )
    else:
        save_people(people)

    return JSONResponse(
        {
            "ok": True,
            "person": {
                "id": person.get("id"),
                "name": person.get("name"),
                "relationship": person.get("relationship"),
                "image": public_image_url(person.get("image")),
                "audio_status": person.get("audio_status"),
            },
        }
    )


@router.post("/database/delete")
async def delete_person(person_id: int = Form(...)):
    people = load_people()

    idx = None
    for i, p in enumerate(people):
        try:
            if int(p.get("id", -1)) == int(person_id):
                idx = i
                break
        except Exception:
            continue

    if idx is None:
        raise HTTPException(status_code=404, detail="Person not found.")

    person = people.pop(idx)

    # Delete files from disk
    try:
        _delete_file_if_exists(person.get("image"))
        _delete_file_if_exists(person.get("headshot"))
        _delete_file_if_exists(person.get("auditory cue (male)"))
        _delete_file_if_exists(person.get("auditory cue (female)"))
    except Exception:
        pass

    save_people(people)

    return JSONResponse({"ok": True})