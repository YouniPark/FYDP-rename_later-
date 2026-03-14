from app.db import Database


def test_db_version_increments(tmp_path) -> None:
    db = Database(str(tmp_path / "test.db"))

    initial = db.get_versions()
    assert initial["faces_version"] == "0"
    assert initial["cues_version"] == "0"

    v1 = db.add_or_update_face({"name": "alice"})
    assert v1 == "1"

    v2 = db.add_or_update_cue({"id": "cue-1", "text": "hello"})
    assert v2 == "1"

    current = db.get_versions()
    assert current["faces_version"] == "1"
    assert current["cues_version"] == "1"
