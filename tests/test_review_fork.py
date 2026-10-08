"""The human-review pause, as the live demo uses it: the real triage graph with stand-in
agents and an in-memory checkpointer (no LLM, no database).

  - find_pause finds the saved pause, also after decisions were made on it
  - every decision on the same pause takes effect (each runs on its own fork)
  - why the fork is needed: resuming the same checkpoint twice replays the first decision
"""
from datetime import datetime, timezone

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

import triage.graph as tg

THREAD = "test:astral-sh/uv#200"
ISSUE = {"number": 200, "title": "uv sync hangs behind a proxy", "body": "HTTPS_PROXY set",
         "created_at": datetime(2025, 6, 1, tzinfo=timezone.utc)}


class FakeLabeler:
    def invoke(self, messages):
        raw = AIMessage(content='{"label": "bug"}', usage_metadata={
            "input_tokens": 10, "output_tokens": 2, "total_tokens": 12})
        return {"parsed": {"label": "bug"}, "raw": raw}


class Returns:
    """Stand-in for an agent subgraph: .invoke(state) -> a fixed answer."""

    def __init__(self, answer):
        self.answer = answer

    def invoke(self, state):
        return self.answer


def fake_drafter(state):
    return {"draft": "Thanks! See #101.", "draft_problems": [], "draft_tokens": 5}


@pytest.fixture
def graph(monkeypatch):
    monkeypatch.setattr(tg, "labeler", FakeLabeler())
    g = tg.build_graph(with_review=True, checkpointer=InMemorySaver(),
                       finder=Returns({"duplicates": [101]}), drafter=fake_drafter,
                       investigator=Returns({"fixed_by": [9001]}))
    g.invoke(dict(ISSUE), {"configurable": {"thread_id": THREAD}})
    return g


def test_find_pause_returns_the_proposal(graph):
    proposal, checkpoint = tg.find_pause(graph, THREAD)
    assert checkpoint
    assert proposal["proposed_label"] == "bug"
    assert proposal["proposed_duplicates"] == [101]
    assert proposal["proposed_fixes"] == [9001]
    assert proposal["draft"] == "Thanks! See #101."


def test_find_pause_unknown_thread(graph):
    assert tg.find_pause(graph, "test:never-triaged") == (None, None)


def test_every_decision_on_the_same_pause_takes_effect(graph):
    _, checkpoint = tg.find_pause(graph, THREAD)
    first = tg.resume_from_pause(graph, THREAD, checkpoint, {"action": "approve"})
    assert first["decision"] == "approved" and first["final_label"] == "bug"

    second = tg.resume_from_pause(graph, THREAD, checkpoint, {"action": "reject"})
    assert second["decision"] == "rejected" and second["final_reply"] == ""

    edit = {"action": "edit", "label": "enhancement", "duplicates": [], "fixes": [9001],
            "reply": "Fixed by #9001."}
    third = tg.resume_from_pause(graph, THREAD, checkpoint, edit)
    assert third["decision"] == "edited"
    assert third["final_label"] == "enhancement" and third["final_fixes"] == [9001]


def test_pause_is_still_found_after_decisions(graph):
    proposal, checkpoint = tg.find_pause(graph, THREAD)
    tg.resume_from_pause(graph, THREAD, checkpoint, {"action": "approve"})
    tg.resume_from_pause(graph, THREAD, checkpoint, {"action": "reject"})
    assert tg.find_pause(graph, THREAD) == (proposal, checkpoint)


def test_plain_resume_of_the_same_checkpoint_replays_the_first_decision(graph):
    # Why resume_from_pause forks. If a LangGraph upgrade makes this test fail, the
    # library changed how it resumes a checkpoint; re-check resume_from_pause.
    _, checkpoint = tg.find_pause(graph, THREAD)
    cfg = {"configurable": {"thread_id": THREAD, "checkpoint_id": checkpoint}}
    assert graph.invoke(Command(resume={"action": "approve"}), cfg)["decision"] == "approved"
    assert graph.invoke(Command(resume={"action": "reject"}), cfg)["decision"] == "approved"
