"""Triage graph, step 1: one node that labels an issue. No agents yet.

    python -m triage.graph 19540
"""
import argparse
from typing import TypedDict

# Must be imported first: it loads .env (GROQ_API_KEY, LANGFUSE_*).
from ingestion.db import get_db_connection

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langfuse import get_client
from langfuse.langchain import CallbackHandler
from langgraph.graph import END, START, StateGraph

from evals.dataset import REPO
# The frozen v2 prompt lives in one place only.
from evals.labeler import LABELS, SCHEMA, SYSTEM_PROMPT, build_user_message

MODEL = "openai/gpt-oss-20b"


# Each node returns only the fields it changed.
class TriageState(TypedDict, total=False):
    number: int
    title: str
    body: str
    label: str      # written by label_issue


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

    return {"label": label}


# ------ Graph ------ #

def build_graph():
    builder = StateGraph(TriageState)
    builder.add_node("label_issue", label_issue)
    builder.add_edge(START, "label_issue")
    builder.add_edge("label_issue", END)
    return builder.compile()


def load_issue(conn, number):
    """Title and body of one uv issue from Postgres."""
    row = conn.execute(
        "SELECT title, body FROM issues WHERE repo = %s AND issue_number = %s",
        (REPO, number),
    ).fetchone()
    if row is None:
        raise SystemExit(f"issue #{number} not found in the issues table")
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("number", type=int, help="uv issue number, e.g. 19540")
    args = p.parse_args()

    with get_db_connection() as conn:
        title, body = load_issue(conn, args.number)

    graph = build_graph()

    # One handler at invoke time traces the whole run as one Langfuse trace.
    result = graph.invoke(
        {"number": args.number, "title": title, "body": body},
        config={"callbacks": [CallbackHandler()], "run_name": f"triage #{args.number}"},
    )
    print(f"#{result['number']} {result['title']!r} -> {result['label']}")

    # Traces upload in the background; flush or the last ones can be lost.
    get_client().flush()


if __name__ == "__main__":
    main()