from app.models import DBName, WSIncomingMessage


def test_ws_event_timing_validates_required_field() -> None:
    msg = WSIncomingMessage(type="event_timing", event_lsl_timestamp=123.4)
    assert msg.event_lsl_timestamp == 123.4


def test_ws_pull_db_requires_db() -> None:
    msg = WSIncomingMessage(type="pull_db", db=DBName.faces)
    assert msg.db == DBName.faces
