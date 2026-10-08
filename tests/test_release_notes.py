"""Release notes -> changelog entries (one per bullet, PR links split out)."""
from ingestion.releases import parse_notes

BODY = """## Release Notes

Released on 2026-09-22.

### Enhancements

- Add `--output-format json` to `uv pip install` ([#21893](https://github.com/astral-sh/uv/pull/21893))
- Select package versions with wheels compatible with each fork
  ([#21835](https://github.com/astral-sh/uv/pull/21835), [#21836](https://github.com/astral-sh/uv/pull/21836))

### Bug fixes

- Handle cycles in `uv pip tree` ([#8689](https://github.com/astral-sh/uv/pull/8689))
- Upgrade PyPy to v7.3.22

## Install uv 0.12.23

- not a changelog bullet ([#1](https://github.com/astral-sh/uv/pull/1))
"""


def test_entries_sections_and_prs():
    entries = parse_notes(BODY)
    assert entries == [
        ("Enhancements", "Add `--output-format json` to `uv pip install`", [21893]),
        ("Enhancements", "Select package versions with wheels compatible with each fork", [21835, 21836]),
        ("Bug fixes", "Handle cycles in `uv pip tree`", [8689]),
        ("Bug fixes", "Upgrade PyPy to v7.3.22", []),
    ]


def test_only_release_notes_section():
    assert all(prs != [1] for _, _, prs in parse_notes(BODY))


def test_empty_or_missing_body():
    assert parse_notes(None) == []
    assert parse_notes("## Install uv 0.0.5\n- x") == []
