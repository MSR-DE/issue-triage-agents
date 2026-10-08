"""The comments fetch resumes from a time cursor, which GitHub needs in UTC."""
from datetime import datetime, timedelta, timezone

from ingestion.comments import EPOCH, iso


def test_cursor_is_converted_to_utc():
    ist = timezone(timedelta(hours=5, minutes=30))
    assert iso(datetime(2025, 1, 1, 5, 30, tzinfo=ist)) == "2025-01-01T00:00:00Z"


def test_empty_table_starts_at_epoch():
    assert iso(None) == EPOCH
