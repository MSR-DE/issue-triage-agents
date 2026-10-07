"""Investigator tools: thin @tool wrappers over retrieval/fix_lookup.py.

Same rules as the Duplicate Finder's tools: the model chooses only the query and
which PR to read; the cutoff (the issue's created_at) and the reporter's version
come from the graph state, and read_pr only opens PRs a search in this run returned.
"""
from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from evals.dataset import REPO
from retrieval.fix_lookup import read_pr as lookup_pr
from retrieval.fix_lookup import search_fix_candidates

SEARCH_TOOL = "search_fixes"
READ_TOOL = "read_pr"
AGENT_K = 5      # agent searches; the automatic first search shows more


def seen_prs(messages):
    """Every PR number any search in this run returned (the ToolMessage artifact)."""
    seen = set()
    for m in messages:
        if isinstance(m, ToolMessage) and m.name == SEARCH_TOOL and m.artifact:
            seen.update(m.artifact)
    return seen


def format_fixes(cands, already_seen=()):
    if not cands:
        return "No merged pull requests found."
    lines = []
    for c in cands:
        if c["number"] in already_seen:
            lines.append(f"PR #{c['number']} (already listed above)")
        else:
            lines.append(f"PR #{c['number']} {c['title']} -- {c['text']} [{c['status']}]")
    return "\n".join(lines)


def make_fix_tools(conn, index):

    @tool(SEARCH_TOOL, response_format="content_and_artifact")
    def search_fixes(query: str, state: Annotated[dict, InjectedState]):
        """Search uv pull requests merged in the days before the issue was opened,
        for a fix of the problem it describes. Returns up to 5 PRs, most similar first, as
        "PR #number title -- release note or description [release status]". Write
        the query as the problem or its fix in uv's terms (commands, flags,
        settings, error messages)."""
        cands = search_fix_candidates(conn, index, REPO, query, before=state["created_at"],
                                      reported=state.get("reported_version"), k=AGENT_K,
                                      since=state.get("since"))       # set by the seed step
        text = format_fixes(cands, already_seen=seen_prs(state["messages"]))
        return text, [c["number"] for c in cands]

    @tool(READ_TOOL)
    def read_pr(number: int, state: Annotated[dict, InjectedState]) -> str:
        """Read one pull request's title, release note and description, to check
        whether it fixes exactly the problem in the issue. Only works for PR
        numbers that a search in this conversation returned."""
        if number not in seen_prs(state["messages"]):
            return (f"Not allowed: PR #{number} was not returned by a search in this "
                    f"conversation. Read only PRs that search_fixes listed.")
        pr = lookup_pr(conn, REPO, number, before=state["created_at"],
                       reported=state.get("reported_version"))
        if pr is None:
            return f"PR #{number} not found."
        notes = f"\nRelease note: {pr['notes']}" if pr["notes"] else ""
        return f"PR #{pr['number']} {pr['title']} [{pr['status']}]{notes}\n{pr['body']}"

    return [search_fixes, read_pr]
