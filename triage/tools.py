"""Duplicate Finder tools: thin @tool wrappers over retrieval/lookup.py.

The model chooses only the query and which candidate to read. The cutoff date and
the "only read what search returned" rule come from the graph state.
"""
from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from evals.dataset import REPO
from retrieval.lookup import read_issue as lookup_issue
from retrieval.lookup import search_candidates

SEARCH_TOOL = "search_similar_issues"
AGENT_K = 5      # agent searches; the automatic first search shows 10


def seen_numbers(messages):
    """Every issue number any search in this run has returned (kept as the
    ToolMessage artifact, which the model never sees). Also used by the grounding check."""
    seen = set()
    for m in messages:
        if isinstance(m, ToolMessage) and m.name == SEARCH_TOOL and m.artifact:
            seen.update(m.artifact)
    return seen


def format_candidates(cands, already_seen=()):
    """One line per candidate; ones already shown are listed by number only (saves tokens)."""
    if not cands:
        return "No earlier issues found."
    lines = []
    for c in cands:
        if c["number"] in already_seen:
            lines.append(f"#{c['number']} (already listed above)")
        else:
            lines.append(f"#{c['number']} {c['title']} -- {c['snippet']}")
    return "\n".join(lines)


def make_tools(conn, index):
    """conn and index are fixed here (the embedding model loads once), never passed by the model."""

    @tool(SEARCH_TOOL, response_format="content_and_artifact")
    def search_similar_issues(query: str, state: Annotated[dict, InjectedState]):
        """Search earlier uv issues for ones about the same problem as the issue
        being triaged. Returns up to 5 issues, most similar first, as
        "#number title -- start of body". Write the query as a short description
        of the underlying problem in uv's own terms (commands, flags, settings,
        error messages), not as a copy of the reporter's wording. Only issues
        created before the one being triaged are searched."""
        cands = search_candidates(conn, index, REPO, query,
                                  before=state["created_at"], k=AGENT_K)
        text = format_candidates(cands, already_seen=seen_numbers(state["messages"]))
        return text, [c["number"] for c in cands]

    @tool
    def read_issue(number: int, state: Annotated[dict, InjectedState]) -> str:
        """Read the title and the start of the body of one earlier issue, to check
        whether it really describes the same problem. Only works for issue
        numbers that a search in this conversation returned."""
        if number not in seen_numbers(state["messages"]):
            return (f"Not allowed: #{number} was not returned by a search in this "
                    f"conversation. Read only issues that search_similar_issues listed.")
        issue = lookup_issue(conn, REPO, number)
        if issue is None:
            return f"Issue #{number} not found."
        return f"#{issue['number']} {issue['title']}\n{issue['body']}"

    return [search_similar_issues, read_issue]