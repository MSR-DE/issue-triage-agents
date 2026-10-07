"""Triage graph: (label + find duplicates + investigate "already fixed?", in parallel)
-> draft reply -> human review (pauses) -> post (dry run).

    python -m triage.graph run 19540       # labels, finds duplicates, drafts, then pauses
    python -m triage.graph review 19540    # shows the proposal, asks, resumes

The pause is saved in Postgres (checkpointer), so `review` works from a new
process, after a restart, or days later.
"""
import argparse
from datetime import datetime
from typing import TypedDict

# Must be imported first: it loads .env (DATABASE_URL, GROQ_API_KEY, LANGFUSE_*).
from ingestion.db import DATABASE_URL, get_db_connection

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langfuse import get_client
from langfuse.langchain import CallbackHandler
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from evals.dataset import REPO
# The frozen v2 prompt lives in one place only.
from evals.labeler import (LABELS, SCHEMA, SECURE_SYSTEM_PROMPT, SYSTEM_PROMPT,
                           build_user_message)
from triage.checks import check_reply
from triage.drafter import make_drafter
from triage.duplicate_finder import make_finder
from triage.investigator import INDEX_VARIANT, make_investigator

MODEL = "openai/gpt-oss-20b"
# Issue text in <issue> tags + "untrusted" note for the labeler. Must keep dev ≈ 84%
# (run_id dev-20b-graph-v2sec) and should block the 2 label-forcing attacks.
SECURE_LABELER = True


# Each node returns only the fields it changed.
class TriageState(TypedDict, total=False):
    number: int
    title: str
    body: str
    created_at: datetime    # input: the duplicate search only looks before it
    label: str              # written by label_issue (the model's proposal)
    raw_output: str         # written by label_issue (the model's raw JSON)
    prompt_tokens: int      # written by label_issue
    completion_tokens: int  # written by label_issue
    duplicates: list[int]   # written by find_duplicates (grounded, best first)
    fixes: list[int]        # written by investigate: merged PRs that may already fix it
    draft: str              # written by draft_reply (gpt-oss-120b)
    draft_problems: list[str]  # written by draft_reply (output checks, triage/checks.py)
    draft_tokens: int       # written by draft_reply
    decision: str           # written by review: approved / edited / rejected
    final_label: str        # written by review: what a human signed off on
    final_duplicates: list[int]  # written by review
    final_fixes: list[int]  # written by review
    final_reply: str        # written by review: the reply a human signed off on ("" = none)


# Same settings as the direct Groq call in evals/labeler.py, plus a safety cap
# on output tokens (gpt-oss's reasoning counts toward it).
llm = ChatGroq(model=MODEL, temperature=0, reasoning_effort="low",
               max_tokens=512, max_retries=2)

# LangChain takes the schema's name from a "title" key.
# include_raw=True keeps the raw reply, so token counts aren't lost.
labeler = llm.with_structured_output(
    {**SCHEMA["schema"], "title": SCHEMA["name"]},
    method="json_schema",
    strict=True,
    include_raw=True,
)


# ------ Nodes ------ #

def label_issue(state: TriageState) -> dict:
    messages = [
        SystemMessage(content=SECURE_SYSTEM_PROMPT if SECURE_LABELER else SYSTEM_PROMPT),
        HumanMessage(content=build_user_message(state["title"], state["body"],
                                                untrusted=SECURE_LABELER)),
    ]
    out = labeler.invoke(messages)

    # Covers: no JSON, cut-off JSON (parsed == {}), or a label outside the four.
    parsed = out["parsed"]
    label = parsed.get("label") if parsed else None
    if label not in LABELS:
        raw = out["raw"]
        finish = raw.response_metadata.get("finish_reason")
        raise ValueError(f"bad labeler output (finish={finish}): {raw.content!r}")

    usage = out["raw"].usage_metadata
    return {
        "label": label,
        "raw_output": out["raw"].content,
        "prompt_tokens": usage["input_tokens"],
        "completion_tokens": usage["output_tokens"],
    }


def review(state: TriageState) -> dict:
    # Pauses the graph here. The dict is what the reviewer sees; the run is
    # saved by the checkpointer and resumes with Command(resume=decision).
    # On resume this node runs again from the top and interrupt() returns the
    # decision, so nothing with side effects may come before it.
    proposed_dups = state.get("duplicates", [])
    proposed_fixes = state.get("fixes", [])
    decision = interrupt({
        "number": state["number"],
        "title": state["title"],
        "proposed_label": state["label"],
        "proposed_duplicates": proposed_dups,
        "proposed_fixes": proposed_fixes,
        "draft": state.get("draft", ""),
        "draft_problems": state.get("draft_problems", []),
    })

    # The reviewer's input is checked too, not just the model's.
    action = decision.get("action")
    if action == "approve":
        return {"decision": "approved", "final_label": state["label"],
                "final_duplicates": proposed_dups, "final_fixes": proposed_fixes,
                "final_reply": state.get("draft", "")}
    if (action == "edit" and decision.get("label") in LABELS
            and all(isinstance(n, int) for n in decision.get("duplicates", []))
            and all(isinstance(n, int) for n in decision.get("fixes", []))
            and isinstance(decision.get("reply", ""), str)):
        return {"decision": "edited", "final_label": decision["label"],
                "final_duplicates": decision.get("duplicates", []),
                "final_fixes": decision.get("fixes", []),
                "final_reply": decision.get("reply", "")}
    if action == "reject":
        return {"decision": "rejected", "final_label": None, "final_duplicates": [],
                "final_fixes": [], "final_reply": ""}
    raise ValueError(f"invalid review decision: {decision!r}")


def post(state: TriageState) -> dict:
    # Dry run: nothing is written to GitHub yet (read-only token, by design).
    print(f"[dry run] would add label '{state['final_label']}' to {REPO}#{state['number']}")
    reply = state.get("final_reply", "")
    if reply:
        # Re-check what is actually being posted (the reviewer may have edited it).
        allowed = state.get("final_duplicates", []) + state.get("final_fixes", [])
        for problem in check_reply(reply, allowed):
            print(f"[warning] reply {problem}")
        print(f"[dry run] would comment:\n{reply}")
    return {}


def after_review(state: TriageState) -> str:
    # Conditional edge: only an approved or edited label goes on to post.
    return "post" if state.get("final_label") else END


# ------ Graph ------ #

def build_graph(with_review=False, checkpointer=None, finder=None, drafter=None,
                investigator=None):
    """with_review=False: label only (what the labeler evals measure; no pause).
    with_review=True: label + find_duplicates (+ investigate, if an Investigator is
    given) in parallel -> draft_reply -> review -> post; needs a checkpointer, a
    Duplicate Finder (make_finder) and a drafter (make_drafter)."""
    builder = StateGraph(TriageState)
    builder.add_node("label_issue", label_issue)
    builder.add_edge(START, "label_issue")

    if not with_review:
        builder.add_edge("label_issue", END)
        return builder.compile(checkpointer=checkpointer)

    def find_duplicates(state: TriageState) -> dict:
        # The agent has its own state; pass in what it needs, take back the answer.
        # Called inside this node, it runs as a subgraph: same trace, same checkpointer.
        r = finder.invoke({"number": state["number"], "title": state["title"],
                           "body": state["body"], "created_at": state["created_at"]})
        return {"duplicates": r["duplicates"]}

    def investigate(state: TriageState) -> dict:
        # Same pattern: the Investigator runs as a subgraph with its own state.
        r = investigator.invoke({"number": state["number"], "title": state["title"],
                                 "body": state["body"], "created_at": state["created_at"]})
        return {"fixes": r["fixed_by"]}

    builder.add_node("find_duplicates", find_duplicates)
    builder.add_node("draft_reply", drafter)
    builder.add_node("review", review)
    builder.add_node("post", post)
    builder.add_edge(START, "find_duplicates")                    # fan out: all start at once
    branches = ["label_issue", "find_duplicates"]
    if investigator is not None:
        builder.add_node("investigate", investigate)
        builder.add_edge(START, "investigate")
        branches.append("investigate")
    builder.add_edge(branches, "draft_reply")                     # fan in: waits for all
    builder.add_edge("draft_reply", "review")
    builder.add_conditional_edges("review", after_review, ["post", END])
    builder.add_edge("post", END)
    return builder.compile(checkpointer=checkpointer)


# ------ CLI ------ #

def load_issue(conn, number):
    """Title, body and created_at of one uv issue from Postgres."""
    row = conn.execute(
        "SELECT title, body, created_at FROM issues WHERE repo = %s AND issue_number = %s",
        (REPO, number),
    ).fetchone()
    if row is None:
        raise SystemExit(f"issue #{number} not found in the issues table")
    return row


def run_config(number):
    # thread_id = which saved run this is; one thread per issue.
    # langfuse_session_id groups the run and review traces in Langfuse.
    thread = f"{REPO}#{number}"
    return {
        "configurable": {"thread_id": thread},
        "callbacks": [CallbackHandler()],
        "metadata": {"langfuse_session_id": thread},
    }


def cmd_run(graph, conn, number):
    config = run_config(number)
    if graph.get_state(config).next:
        raise SystemExit(f"#{number} is already waiting for review: "
                         f"python -m triage.graph review {number}")

    title, body, created_at = load_issue(conn, number)
    result = graph.invoke({"number": number, "title": title, "body": body,
                           "created_at": created_at},
                          {**config, "run_name": f"triage #{number}"})
    if "__interrupt__" in result:
        flags = f", {len(result['draft_problems'])} flag(s) on the draft" if result["draft_problems"] else ""
        print(f"#{number} {title!r}: proposed '{result['label']}', "
              f"duplicates {result['duplicates'] or 'none'}, "
              f"possible fixes {result.get('fixes') or 'none'}{flags}, waiting for review.")
        print(f"  python -m triage.graph review {number}")


def ask_decision(label, duplicates, draft, fixes=()):
    while True:
        answer = input("[a]pprove, [e]dit, [r]eject? ").strip().lower()
        if answer == "a":
            return {"action": "approve"}
        if answer == "r":
            return {"action": "reject"}
        if answer == "e":
            return {"action": "edit", "label": ask_label(label),
                    "duplicates": ask_duplicates(duplicates),
                    "fixes": ask_duplicates(list(fixes), what="possible fixes (PR numbers"),
                    "reply": ask_reply(draft)}


def ask_reply(current):
    raw = input("reply (type a new one; '-' for no reply; Enter keeps the draft): ").strip()
    if not raw:
        return current
    return "" if raw == "-" else raw


def ask_label(current):
    while True:
        label = input(f"label ({' / '.join(LABELS)}, Enter keeps '{current}'): ").strip().lower()
        if not label:
            return current
        if label in LABELS:
            return label
        print("not one of the four labels")


def ask_duplicates(current, what="duplicates (issue numbers"):
    shown = ", ".join(map(str, current)) or "none"
    while True:
        raw = input(f"{what}, comma-separated; '-' for none; Enter keeps {shown}): ").strip()
        if not raw:
            return current
        if raw == "-":
            return []
        try:
            return [int(x.strip().lstrip("#")) for x in raw.split(",") if x.strip()]
        except ValueError:
            print("use issue numbers like 1526, 1374")


def cmd_review(graph, conn, number):
    config = run_config(number)
    snapshot = graph.get_state(config)
    if not snapshot.interrupts:
        raise SystemExit(f"nothing waiting for review on #{number}")

    proposal = snapshot.interrupts[0].value
    print(f"#{proposal['number']} {proposal['title']!r}")
    print(f"  label: {proposal['proposed_label']}")
    for n in proposal["proposed_duplicates"] or []:
        row = conn.execute("SELECT title FROM issues WHERE repo = %s AND issue_number = %s",
                           (REPO, n)).fetchone()
        print(f"  possible duplicate: #{n} {row[0] if row else ''!r}")
    if not proposal["proposed_duplicates"]:
        print("  possible duplicates: none")
    for n in proposal.get("proposed_fixes") or []:
        row = conn.execute("SELECT title, merged_at FROM pull_requests WHERE repo = %s AND pr_number = %s",
                           (REPO, n)).fetchone()
        print(f"  possible fix: PR #{n} {row[0] if row else ''!r} (merged {row[1]:%Y-%m-%d})" if row
              else f"  possible fix: PR #{n}")
    if not proposal.get("proposed_fixes"):
        print("  possible fixes: none")
    print("  draft reply:\n    " + (proposal["draft"] or "(none)").replace("\n", "\n    "))
    for problem in proposal["draft_problems"]:
        print(f"  [check] draft {problem}")
    decision = ask_decision(proposal["proposed_label"], proposal["proposed_duplicates"],
                            proposal["draft"], proposal.get("proposed_fixes") or [])

    result = graph.invoke(Command(resume=decision),
                          {**config, "run_name": f"review #{number}"})
    print(f"#{number}: {result['decision']} -> {result['final_label']}, "
          f"duplicates {result['final_duplicates'] or 'none'}, "
          f"fixes {result.get('final_fixes') or 'none'}, "
          f"reply {'yes' if result['final_reply'] else 'none'}")


def main(index_factory=None, fix_index_factory=None):
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["run", "review"])
    p.add_argument("number", type=int, help="uv issue number, e.g. 19540")
    args = p.parse_args()

    # Two connections: the checkpointer's (autocommit, owned by LangGraph) and ours
    # for issue lookups and the Duplicate Finder's tools. setup() creates the
    # checkpoint tables the first time and is a no-op after that.
    with PostgresSaver.from_conn_string(DATABASE_URL) as checkpointer, \
            get_db_connection() as conn:
        checkpointer.setup()
        if index_factory is None:
            from retrieval.vector import VectorIndex      # loads the embedding model
            index = VectorIndex(conn, REPO)
        else:
            index = index_factory(conn)
        if fix_index_factory is None:
            from retrieval.fix_index import FixIndex   # reuses the loaded embedding model
            fix_index = FixIndex(conn, REPO, INDEX_VARIANT, index.model)
        else:
            fix_index = fix_index_factory(conn)
        graph = build_graph(with_review=True, checkpointer=checkpointer,
                            finder=make_finder(conn, index), drafter=make_drafter(conn),
                            investigator=make_investigator(conn, fix_index))
        if args.command == "run":
            cmd_run(graph, conn, args.number)
        else:
            cmd_review(graph, conn, args.number)

    # Traces upload in the background; flush or the last ones can be lost.
    get_client().flush()


if __name__ == "__main__":
    main()