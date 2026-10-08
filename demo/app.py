"""Live demo: triage a uv GitHub issue end to end, with a human approving the result.

    streamlit run demo/app.py            # local; uses DEMO_DATABASE_URL (Neon) if set in .env

On Hugging Face Spaces this file is app.py, with DATABASE_URL (Neon) and GROQ_API_KEY as
secrets. Nothing is ever posted to GitHub: the last step shows what *would* be posted.

Token budget: the Groq free tier is shared with the evals, so live runs are capped per day
(DAILY_LIVE_RUNS). A triaged issue stays paused in the Postgres checkpointer, so the next
visitor who picks it sees the saved result for free, and every review decision runs on
its own fork of that saved pause (see decide()).
"""
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()
if os.getenv("DEMO_DATABASE_URL"):                 # local run against the demo database
    os.environ["DATABASE_URL"] = os.environ["DEMO_DATABASE_URL"]
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT if (ROOT / "triage").exists() else ROOT.parent))

import psycopg                                      # noqa: E402  (after the path/env setup)
from groq import RateLimitError                     # noqa: E402
from langgraph.checkpoint.postgres import PostgresSaver  # noqa: E402
from pgvector.psycopg import register_vector        # noqa: E402

from evals.dataset import REPO                      # noqa: E402
from evals.labeler import LABELS                    # noqa: E402
from ingestion.db import DATABASE_URL               # noqa: E402
from retrieval.fix_index import FixIndex            # noqa: E402
from retrieval.vector import VectorIndex            # noqa: E402
from triage.checks import check_reply               # noqa: E402
from triage.drafter import make_drafter             # noqa: E402
from triage.duplicate_finder import make_finder     # noqa: E402
from triage.graph import build_graph, find_pause, resume_from_pause  # noqa: E402
from triage.investigator import INDEX_VARIANT, make_investigator  # noqa: E402

DAILY_LIMIT = int(os.getenv("DAILY_LIVE_RUNS", "5"))   # ~8.5K gpt-oss-20b tokens per live run
GITHUB = f"https://github.com/{REPO}"
CODE = "https://github.com/MSR-DE/issue-triage-agents"

SETUP_SQL = """
CREATE TABLE IF NOT EXISTS demo_usage (day date PRIMARY KEY, runs integer NOT NULL);
"""
# One atomic statement: count this run only if today's cap isn't reached yet.
TAKE_RUN = """
INSERT INTO demo_usage (day, runs) VALUES (current_date, 1)
ON CONFLICT (day) DO UPDATE SET runs = demo_usage.runs + 1 WHERE demo_usage.runs < %s
RETURNING runs
"""
RUNS_TODAY = "SELECT coalesce((SELECT runs FROM demo_usage WHERE day = current_date), 0)"
RECENT = """
SELECT issue_number, title FROM issues
WHERE repo = %s AND user_type <> 'Bot' ORDER BY created_at DESC LIMIT 200
"""


# ------ shared resources (loaded once per server) ------ #

@st.cache_resource
def embedding_model():
    from sentence_transformers import SentenceTransformer
    from retrieval.embed import MODEL_NAME
    return SentenceTransformer(MODEL_NAME)


@st.cache_resource
def setup_database():
    with psycopg.connect(DATABASE_URL, autocommit=True) as conn:
        conn.execute(SETUP_SQL)
    with PostgresSaver.from_conn_string(DATABASE_URL) as saver:
        saver.setup()                               # LangGraph's checkpoint tables
    return True


class SharedModelVectorIndex(VectorIndex):
    """VectorIndex, but reusing the already loaded embedding model."""

    def __init__(self, conn, repo, model):
        register_vector(conn)
        self.conn, self.repo, self.model = conn, repo, model


def build(conn, saver):
    model = embedding_model()
    return build_graph(
        with_review=True, checkpointer=saver,
        finder=make_finder(conn, SharedModelVectorIndex(conn, REPO, model)),
        drafter=make_drafter(conn),
        investigator=make_investigator(conn, FixIndex(conn, REPO, INDEX_VARIANT, model)),
    )


def run_config(thread):
    cfg = {"configurable": {"thread_id": thread}}
    if os.getenv("LANGFUSE_PUBLIC_KEY"):            # traces only when Langfuse is configured
        from langfuse.langchain import CallbackHandler
        cfg["callbacks"] = [CallbackHandler()]
        cfg["metadata"] = {"langfuse_session_id": thread}
    return cfg


# ------ actions ------ #

def triage(issue):
    """Run the graph up to the human-review pause (or reuse a saved pause)."""
    with psycopg.connect(DATABASE_URL, autocommit=True) as conn, \
            PostgresSaver.from_conn_string(DATABASE_URL) as saver:
        graph = build(conn, saver)
        proposal, checkpoint = find_pause(graph, issue["thread"])
        if proposal:
            return proposal, checkpoint, True
        if conn.execute(TAKE_RUN, (DAILY_LIMIT,)).fetchone() is None:
            return None, None, False
        graph.invoke({k: issue[k] for k in ("number", "title", "body", "created_at")},
                     run_config(issue["thread"]))
        proposal, checkpoint = find_pause(graph, issue["thread"])
        return proposal, checkpoint, False


def decide(issue, checkpoint, decision):
    """Resume the saved pause with the reviewer's decision. Each decision runs on its own
    fork of the pause (see resume_from_pause), so it can be decided again and again."""
    with psycopg.connect(DATABASE_URL, autocommit=True) as conn, \
            PostgresSaver.from_conn_string(DATABASE_URL) as saver:
        return resume_from_pause(build(conn, saver), issue["thread"], checkpoint, decision,
                                 run_config(issue["thread"]))


def details(proposal):
    with psycopg.connect(DATABASE_URL, autocommit=True) as conn:
        dups = dict(conn.execute("SELECT issue_number, title FROM issues WHERE repo = %s "
                                 "AND issue_number = ANY(%s)",
                                 (REPO, proposal["proposed_duplicates"])).fetchall())
        prs = {n: (t, m) for n, t, m in conn.execute(
            "SELECT pr_number, title, merged_at FROM pull_requests WHERE repo = %s "
            "AND pr_number = ANY(%s)", (REPO, proposal["proposed_fixes"])).fetchall()}
    return dups, prs


def parse_numbers(text):
    return [int(x.strip().lstrip("#")) for x in text.split(",") if x.strip().lstrip("#").isdigit()]


# ------ page ------ #

st.set_page_config(page_title="uv issue triage", page_icon="🔧", layout="wide")
setup_database()

with st.sidebar:
    st.header("How it works")
    st.markdown(
        "Three steps run **in parallel** on a new issue, then a drafter writes a reply and "
        "**a human approves** it:\n"
        "1. **Labeler**: bug / enhancement / question / documentation\n"
        "2. **Duplicate Finder** agent: searches earlier issues\n"
        "3. **Investigator** agent: was it *already fixed* by a merged PR?\n\n"
        "Code checks every draft (links, @mentions, commands, promises). Agents can only "
        "cite issues and PRs their own searches returned.")
    st.header("Measured on real uv issues")
    st.markdown(
        "- Labeler: **79.8%** vs 74.6% for rules (228 issues)\n"
        "- Embedding search: **55%** duplicate recall@5 vs 33% for keyword search (347 duplicates)\n"
        "- Already-fixed: never claimed a fix on **30/30** unfixed bugs\n"
        "- Prompt injection: **0/18** reply attacks got through\n")
    st.markdown(f"[Code, evals and write-up]({CODE})")

st.title("🔧 uv issue triage, live")
st.caption("Pick a real issue from astral-sh/uv (or paste your own), let the agents triage it, "
           "then approve, edit or reject. **Nothing is posted to GitHub.**")

with psycopg.connect(DATABASE_URL, autocommit=True) as c:
    recent = c.execute(RECENT, (REPO,)).fetchall()
    used = c.execute(RUNS_TODAY).fetchone()[0]

PASTE = "✏️  Paste your own issue"
choice = st.selectbox("Issue", [PASTE] + [f"#{n} {t}" for n, t in recent], index=1)

if choice == PASTE:
    title = st.text_input("Title", max_chars=200)
    body = st.text_area("Body", height=200, max_chars=4000)
    st.session_state.setdefault("paste_id", uuid.uuid4().hex[:12])
    issue = {"number": 0, "title": title, "body": body,
             "created_at": datetime.now(timezone.utc),
             "thread": f"demo:paste:{st.session_state.paste_id}:{hash((title, body))}"}
else:
    number = int(choice.split()[0][1:])
    with psycopg.connect(DATABASE_URL, autocommit=True) as c:
        title, body, created = c.execute(
            "SELECT title, body, created_at FROM issues WHERE repo = %s AND issue_number = %s",
            (REPO, number)).fetchone()
    issue = {"number": number, "title": title, "body": body or "", "created_at": created,
             "thread": f"demo:{REPO}#{number}"}
    with st.expander(f"Issue #{number} as it was opened ({created:%Y-%m-%d})"):
        st.markdown(f"[Open on GitHub]({GITHUB}/issues/{number})")
        st.text((body or "")[:3000])

if st.session_state.get("thread") != issue["thread"]:   # new issue picked: forget old results
    for key in ("proposal", "checkpoint", "result", "cached"):
        st.session_state.pop(key, None)
    st.session_state.thread = issue["thread"]

left = max(0, DAILY_LIMIT - used)
if st.button(f"Triage this issue  ·  {left} live run{'s' * (left != 1)} left today", type="primary",
             disabled=not issue["title"].strip()):
    with st.spinner("Labeling, searching for duplicates and checking for fixes (~30 s)…"):
        try:
            proposal, checkpoint, cached = triage(issue)
        except RateLimitError:
            st.error("The free LLM budget is used up for now. Try an issue someone already "
                     "triaged (they load instantly) or come back later.")
            st.stop()
    if proposal is None:
        st.warning(f"Today's {DAILY_LIMIT} live runs are used up. Issues that were already "
                   "triaged still load instantly.")
    else:
        st.session_state.update(proposal=proposal, checkpoint=checkpoint, cached=cached)
        st.session_state.pop("result", None)

proposal = st.session_state.get("proposal")
if proposal:
    if st.session_state.get("cached"):
        st.info("Saved result from an earlier run (no LLM tokens used).")
    dups, prs = details(proposal)
    a, b, c = st.columns(3)
    a.metric("Proposed label", proposal["proposed_label"])
    with b:
        st.markdown("**Possible duplicates**")
        for n in proposal["proposed_duplicates"]:
            st.markdown(f"- [#{n}]({GITHUB}/issues/{n}) {dups.get(n, '')}")
        if not proposal["proposed_duplicates"]:
            st.markdown("none")
    with c:
        st.markdown("**May already be fixed by**")
        for n in proposal["proposed_fixes"]:
            title_, merged = prs.get(n, ("", None))
            when = f" (merged {merged:%Y-%m-%d})" if merged else ""
            st.markdown(f"- [PR #{n}]({GITHUB}/pull/{n}) {title_}{when}")
        if not proposal["proposed_fixes"]:
            st.markdown("no merged fix found")

    st.markdown("**Drafted reply** (gpt-oss-120b, checked in code before you see it)")
    st.info(proposal["draft"] or "(no reply)")
    for problem in proposal["draft_problems"]:
        st.warning(f"Check: the draft {problem}")

    st.subheader("Your review")
    approve, reject = st.columns(2)
    decision = None
    if approve.button("✅ Approve as is", use_container_width=True):
        decision = {"action": "approve"}
    if reject.button("❌ Reject", use_container_width=True):
        decision = {"action": "reject"}
    with st.expander("✏️ Edit before approving"):
        with st.form("edit"):
            label = st.selectbox("Label", LABELS, index=LABELS.index(proposal["proposed_label"]))
            dup_text = st.text_input("Duplicates (issue numbers)",
                                     ", ".join(map(str, proposal["proposed_duplicates"])))
            fix_text = st.text_input("Fixes (PR numbers)",
                                     ", ".join(map(str, proposal["proposed_fixes"])))
            reply = st.text_area("Reply", proposal["draft"], max_chars=2000)
            if st.form_submit_button("Approve with edits"):
                decision = {"action": "edit", "label": label, "duplicates": parse_numbers(dup_text),
                            "fixes": parse_numbers(fix_text), "reply": reply}
    if decision:
        st.session_state.result = decide(issue, st.session_state.checkpoint, decision)

result = st.session_state.get("result")
if result:
    st.subheader("Result (dry run)")
    if result.get("decision") == "rejected":
        st.markdown("Rejected: nothing would be posted.")
    else:
        where = f"#{issue['number']}" if issue["number"] else "the issue"
        st.markdown(f"Would add label **{result['final_label']}** to {where}.")
        reply = result.get("final_reply", "")
        if reply:
            allowed = result.get("final_duplicates", []) + result.get("final_fixes", [])
            for problem in check_reply(reply, allowed):   # re-checked: you may have edited it
                st.warning(f"Check: the reply {problem}")
            st.markdown("Would comment:")
            st.code(reply, language=None)
