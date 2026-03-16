from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.config import settings
from WebServer.setting.setting_config import (
    DURATION_TO_TIME,
    FONT_SIZE_TO_PX,
    IMAGE_SIZE_TO_SCALE,
)

SETTINGS_PATH = settings.setting_dir / "settings.json"
PEOPLE_PATH = settings.people_json_path
VALID_VOICE_TYPES = {"male", "female"}


def load_json(json_path: Path) -> Any:
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_binary_file(file_path: Path) -> bytes:
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    with file_path.open("rb") as f:
        return f.read()


def find_person_by_id(people_data: list[dict[str, Any]], people_id: int) -> dict[str, Any]:
    for person in people_data:
        if person.get("id") == people_id:
            return person
    raise ValueError(f"Person with id={people_id} not found")


def validate_settings(settings_json: dict[str, Any]) -> tuple[str, str, str, str, list[str]]:
    required_keys = [
        "font_size",
        "image_size",
        "duration_time",
        "voice_type",
        "cue_selection",
    ]
    missing_keys = [key for key in required_keys if key not in settings_json]
    if missing_keys:
        raise ValueError(f"Missing required setting(s) in settings.json: {missing_keys}")

    font_size_label = settings_json["font_size"]
    image_size_label = settings_json["image_size"]
    duration_label = settings_json["duration_time"]
    voice_type = settings_json["voice_type"]
    cue_selection = settings_json["cue_selection"]

    if font_size_label not in FONT_SIZE_TO_PX:
        raise ValueError(f"Invalid font_size: {font_size_label}")
    if image_size_label not in IMAGE_SIZE_TO_SCALE:
        raise ValueError(f"Invalid image_size: {image_size_label}")
    if duration_label not in DURATION_TO_TIME:
        raise ValueError(f"Invalid duration_time: {duration_label}")
    if voice_type not in VALID_VOICE_TYPES:
        raise ValueError(f"Invalid voice_type: {voice_type}")
    if not isinstance(cue_selection, list):
        raise ValueError("cue_selection must be a list")

    return font_size_label, image_size_label, duration_label, voice_type, cue_selection


def resolve_image_path(person: dict[str, Any], people_id: int) -> Path:
    image_rel_path = person.get("image")
    if not image_rel_path:
        raise ValueError(f"No image found for people_id={people_id}")
    return settings.images_dir / Path(image_rel_path).name


def resolve_audio_path(person: dict[str, Any], people_id: int, voice_type: str) -> Path:
    audio_field = f"auditory cue ({voice_type})"
    audio_rel_path = person.get(audio_field)
    if not audio_rel_path:
        raise ValueError(
            f"No audio found in field '{audio_field}' for people_id={people_id}"
        )
    return settings.auditory_cue_dir / Path(audio_rel_path).name


def get_person_cue_data(people_id: int) -> dict[str, Any]:
    settings_json = load_json(SETTINGS_PATH)
    people_data = load_json(PEOPLE_PATH)

    (
        font_size_label,
        image_size_label,
        duration_label,
        voice_type,
        cue_selection,
    ) = validate_settings(settings_json)

    person = find_person_by_id(people_data, people_id)

    cues: dict[str, Any] = {}

    if "name" in cue_selection:
        cues["name"] = person.get("name")

    if "relationship" in cue_selection:
        cues["relationship"] = person.get("relationship")

    if "image" in cue_selection:
        cues["image"] = read_binary_file(resolve_image_path(person, people_id))

    if "audio" in cue_selection:
        cues["audio"] = read_binary_file(resolve_audio_path(person, people_id, voice_type))

    return {
        "people_id": people_id,
        "font_size_px": FONT_SIZE_TO_PX[font_size_label],
        "image_scale": IMAGE_SIZE_TO_SCALE[image_size_label],
        "duration_seconds": DURATION_TO_TIME[duration_label],
        "cues": cues,
    }


def cue_preparation(is_unfamiliar: bool, people_id: int | None,) -> tuple[bool, dict[str, Any]]:
    if is_unfamiliar:
        return False, {}

    if people_id is None:
        # TO-DO: return message saying the person in question does not exist in the database 
        return False, {}

    try:
        cue_payload = get_person_cue_data(people_id)
    except Exception:
        return False, {}

    return True, cue_payload