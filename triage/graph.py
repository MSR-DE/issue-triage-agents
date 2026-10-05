"""Triage graph, step 2: label -> human review (pauses) -> post (dry run).

    python -m triage.graph run 19540       # labels, then pauses for review
    python -m triage.graph review 19540    # shows the proposal, asks, resumes

The pause is saved in Postgres (checkpointer), so `review` works from a new
process, after a restart, or days later.
"""
import argparse
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
from evals.labeler import LABELS, SCHEMA, SYSTEM_PROMPT, build_user_message

MODEL = "openai/gpt-oss-20b"


# Each node returns only the fields it changed.
class TriageState(TypedDict, total=False):
    number: int
    title: str
    body: str
    label: str              # written by label_issue (the model's proposal)
    raw_output: str         # written by label_issue (the model's raw JSON)
    prompt_tokens: int      # written by label_issue
    completion_tokens: int  # written by label_issue
    decision: str           # written by review: approved / edited / rejected
    final_label: str        # written by review: what a human signed off on


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
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=build_user_message(state["title"], state["body"])),
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
    decision = interrupt({
        "number": state["number"],
        "title": state["title"],
        "proposed_label": state["label"],
    })

    # The reviewer's input is checked too, not just the model's.
    action = decision.get("action")
    if action == "approve":
        return {"decision": "approved", "final_label": state["label"]}
    if action == "edit" and decision.get("label") in LABELS:
        return {"decision": "edited", "final_label": decision["label"]}
    if action == "reject":
        return {"decision": "rejected", "final_label": None}
    raise ValueError(f"invalid review decision: {decision!r}")


def post(state: TriageState) -> dict:
    # Dry run: nothing is written to GitHub yet (read-only token, by design).
    print(f"[dry run] would add label '{state['final_label']}' to {REPO}#{state['number']}")
    return {}


def after_review(state: TriageState) -> str:
    # Conditional edge: only an approved or edited label goes on to post.
    return "post" if state.get("final_label") else END


# ------ Graph ------ #

def build_graph(with_review=False, checkpointer=None):
    """with_review=False: label only (what the evals measure; no pause).
    with_review=True: label -> review -> post; needs a checkpointer."""
    builder = StateGraph(TriageState)
    builder.add_node("label_issue", label_issue)
    builder.add_edge(START, "label_issue")

    if with_review:
        builder.add_node("review", review)
        builder.add_node("post", post)
        builder.add_edge("label_issue", "review")
        builder.add_conditional_edges("review", after_review, ["post", END])
        builder.add_edge("post", END)
    else:
        builder.add_edge("label_issue", END)

    return builder.compile(checkpointer=checkpointer)


# ------ CLI ------ #

def load_issue(conn, number):
    """Title and body of one uv issue from Postgres."""
    row = conn.execute(
        "SELECT title, body FROM issues WHERE repo = %s AND issue_number = %s",
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


def cmd_run(graph, number):
    config = run_config(number)
    if graph.get_state(config).next:
        raise SystemExit(f"#{number} is already waiting for review: "
                         f"python -m triage.graph review {number}")

    with get_db_connection() as conn:
        title, body = load_issue(conn, number)

    result = graph.invoke({"number": number, "title": title, "body": body},
                          {**config, "run_name": f"triage #{number}"})
    if "__interrupt__" in result:
        print(f"#{number} {title!r}: proposed '{result['label']}', waiting for review.")
        print(f"  python -m triage.graph review {number}")


def ask_decision(proposed):
    while True:
        answer = input(f"[a]pprove '{proposed}', [e]dit, [r]eject? ").strip().lower()
        if answer == "a":
            return {"action": "approve"}
        if answer == "r":
            return {"action": "reject"}
        if answer == "e":
            while True:
                label = input(f"new label ({' / '.join(LABELS)}): ").strip().lower()
                if label in LABELS:
                    return {"action": "edit", "label": label}
                print("not one of the four labels")


def cmd_review(graph, number):
    config = run_config(number)
    snapshot = graph.get_state(config)
    if not snapshot.interrupts:
        raise SystemExit(f"nothing waiting for review on #{number}")

    proposal = snapshot.interrupts[0].value
    print(f"#{proposal['number']} {proposal['title']!r}")
    decision = ask_decision(proposal["proposed_label"])

    result = graph.invoke(Command(resume=decision),
                          {**config, "run_name": f"review #{number}"})
    print(f"#{number}: {result['decision']} -> {result['final_label']}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["run", "review"])
    p.add_argument("number", type=int, help="uv issue number, e.g. 19540")
    args = p.parse_args()

    # One Postgres connection for the checkpointer; setup() creates its
    # tables the first time and is a no-op after that.
    with PostgresSaver.from_conn_string(DATABASE_URL) as checkpointer:
        checkpointer.setup()
        graph = build_graph(with_review=True, checkpointer=checkpointer)
        if args.command == "run":
            cmd_run(graph, args.number)
        else:
            cmd_review(graph, args.number)

    # Traces upload in the background; flush or the last ones can be lost.
    get_client().flush()


if __name__ == "__main__":
    main()