"""Facts the Investigator computes in code: the reporter's version and each fix's status."""
from datetime import datetime, timezone

from retrieval.fix_lookup import fix_status, reported_version, version_tuple

MERGED = datetime(2025, 2, 13, tzinfo=timezone.utc)


def test_reported_version():
    assert reported_version("### Version\nuv 0.4.18 (abc 2024-10-01)") == "0.4.18"
    assert reported_version("works with uv-0.5.30 but not 0.5.31") == "0.5.30"
    assert reported_version("uv v0.6.0") == "0.6.0"
    assert reported_version("Python 3.12.4, no uv version given") is None
    assert reported_version(None) is None


def test_version_tuple_compares_numerically():
    assert version_tuple("0.10.2") > version_tuple("0.9.30")
    assert version_tuple("v1.2.3") == (1, 2, 3)
    assert version_tuple("nightly") is None


def test_fix_status():
    assert fix_status(MERGED, None, "0.5.30") == "merged 2025-02-13, not yet in a release when the issue was opened"
    assert fix_status(MERGED, "0.6.0", "0.5.30") == "released in 0.6.0, newer than the reporter's uv 0.5.30"
    assert fix_status(MERGED, "0.5.30", "0.5.31") == "released in 0.5.30; the reporter's uv 0.5.31 already includes it"
    assert fix_status(MERGED, "0.6.0", None) == "released in 0.6.0"
