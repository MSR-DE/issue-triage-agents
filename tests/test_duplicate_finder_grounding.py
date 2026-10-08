"""The Duplicate Finder's safety rules, tested end to end with a scripted fake Groq API
(no network, no tokens) and an in-memory stand-in for Postgres:
  - read_issue refuses an issue no search in this run returned
  - the verdict's made-up issue numbers are dropped and reported as invented
  - the tool-round cap holds
"""
import json
from datetime import datetime, timezone

import groq
import httpx
import pytest

from triage.duplicate_finder import MAX_TOOL_ROUNDS, make_finder

CREATED = datetime(2025, 1, 10, tzinfo=timezone.utc)
ISSUES = {  # number -> (title, body)
    101: ("uv pip install fails without a virtualenv", "global install outside a venv"),
    102: ("Add flag for global installs in CI", "--system for CI images"),
    103: ("Unrelated: lockfile ordering", "ordering of entries"),
}


class FakeRows:
    def __init__(self, rows):
        self.rows = rows

    def fetchall(self):
        return self.rows

    def fetchone(self):
        return self.rows[0] if self.rows else None


class FakeConn:
    """Answers the two queries retrieval/lookup.py runs."""

    def execute(self, sql, params):
        if "ANY(%s)" in sql:                       # search results: several issues
            return FakeRows([(n, *ISSUES[n]) for n in params[1] if n in ISSUES])
        n = params[1]                              # read_issue: one issue
        return FakeRows([(n, *ISSUES[n], CREATED)] if n in ISSUES else [])


class FakeIndex:
    def search(self, query, before, k=10):
        assert before == CREATED                   # the cutoff always comes from state
        return [101, 102, 103][:k]


def scripted_groq(script):
    """A fake Groq API: agent turns follow `script` (a list of tool calls or None),
    the verdict call returns `script.verdict`."""
    calls = {"agent": 0, "requests": []}

    def handler(request):
        body = json.loads(request.content)
        calls["requests"].append(body)
        schema = (body.get("response_format") or {}).get("json_schema", {}).get("name")
        if schema == "duplicate_verdict":
            msg = {"role": "assistant", "content": json.dumps({"duplicates": script["verdict"]})}
            finish = "stop"
        else:
            step = script["agent"][calls["agent"]] if calls["agent"] < len(script["agent"]) else None
            calls["agent"] += 1
            if step:
                name, args = step
                msg = {"role": "assistant", "content": None, "tool_calls": [{
                    "id": f"call{calls['agent']}", "type": "function",
                    "function": {"name": name, "arguments": json.dumps(args)}}]}
                finish = "tool_calls"
            else:
                msg = {"role": "assistant", "content": "done"}
                finish = "stop"
        return httpx.Response(200, json={
            "id": "x", "object": "chat.completion", "created": 0, "model": "m",
            "choices": [{"index": 0, "finish_reason": finish, "message": msg}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110}})

    return handler, calls


@pytest.fixture
def run_finder(monkeypatch):
    def run(script):
        handler, calls = scripted_groq(script)
        original = groq.Groq.__init__

        def patched(self, *args, **kwargs):
            kwargs["http_client"] = httpx.Client(transport=httpx.MockTransport(handler))
            original(self, *args, **kwargs)

        monkeypatch.setattr(groq.Groq, "__init__", patched)
        finder = make_finder(FakeConn(), FakeIndex())
        result = finder.invoke({"number": 200, "title": "pip install without venv",
                                "body": "can't install globally", "created_at": CREATED})
        return result, calls
    return run


def tool_results(result, name):
    return [m.content for m in result["messages"] if getattr(m, "name", None) == name]


def test_invented_numbers_are_dropped(run_finder):
    result, _ = run_finder({"agent": [None], "verdict": [101, 999]})
    assert result["duplicates"] == [101]
    assert result["invented"] == [999]


def test_read_refuses_issues_no_search_returned(run_finder):
    result, _ = run_finder({"agent": [("read_issue", {"number": 999}), None], "verdict": []})
    assert any("Not allowed" in text for text in tool_results(result, "read_issue"))
    assert result["duplicates"] == []


def test_read_works_for_seen_issues(run_finder):
    result, _ = run_finder({"agent": [("read_issue", {"number": 102}), None], "verdict": [102]})
    assert any("Add flag for global installs" in t for t in tool_results(result, "read_issue"))
    assert result["duplicates"] == [102]


def test_tool_round_cap(run_finder):
    endless = [("read_issue", {"number": 101})] * 10
    result, calls = run_finder({"agent": endless, "verdict": [101]})
    assert result["rounds"] == MAX_TOOL_ROUNDS
    assert calls["agent"] == MAX_TOOL_ROUNDS          # no agent turn after the cap
    assert result["duplicates"] == [101]


def test_verdict_sees_no_tool_definitions(run_finder):
    # Groq rejects a tool-less call that contains tool-call history ("Tool choice is none");
    # the verdict gets a plain-text transcript instead.
    _, calls = run_finder({"agent": [("read_issue", {"number": 101}), None], "verdict": [101]})
    verdict = [r for r in calls["requests"] if r.get("response_format")][-1]
    assert "tools" not in verdict
    assert all(m["role"] in ("system", "user") for m in verdict["messages"])
