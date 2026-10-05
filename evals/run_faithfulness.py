"""DeepEval faithfulness on drafted replies: does each claim in the reply follow
from the issue text and the agents' findings?

    python -m evals.run_faithfulness --run-id dupdev-20b-v1 [--limit 20]

Findings per issue = the Duplicate Finder's answer from a stored run (dup_eval_results)
+ the rule-based label (evals/baseline.py) — so this costs no gpt-oss-20b tokens; the
drafter and the judge both run on gpt-oss-120b. Faithfulness = supported claims /
all claims (1.0 = nothing made up). Restating the issue counts as supported, so the
issue text is part of the context. Writes results/faithfulness_<run_id>[_grounded].csv.

--grounded counts claims the context does not support as failures. DeepEval's
default only fails claims that CONTRADICT the context, so an unsupported
commitment ("we'll add this to our backlog") passes; first run on 5 Oct: 20/20.
--reuse-drafts judges the drafts saved by an earlier run instead of re-drafting.
Each run first checks the judge on 3 planted bad replies (it must catch them).
"""
import argparse
import csv
import os
import time
from pathlib import Path

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")

from ingestion.db import get_db_connection          # also loads .env

from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case import LLMTestCase

from evals.baseline import template_label
from evals.dataset import REPO
from evals.dup_dataset import load_dup_set
from evals.judge import GroqJudge
from retrieval.text import issue_text
from triage.checks import check_reply
from triage.drafter import make_llm, write_draft

QUERY_BODY_CHARS = 1500


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", required=True, help="Duplicate Finder run whose answers to use")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--sleep", type=float, default=30.0)   # 120b: 8K tokens/minute
    p.add_argument("--grounded", action="store_true",
                   help="unsupported claims fail too (not only contradicted ones)")
    p.add_argument("--reuse-drafts", action="store_true",
                   help="judge the drafts in results/faithfulness_<run_id>.csv")
    args = p.parse_args()

    drafter, judge = make_llm(), GroqJudge()
    first = Path(f"results/faithfulness_{args.run_id}.csv")
    out = Path(f"results/faithfulness_{args.run_id}{'_grounded' if args.grounded else ''}.csv")
    saved = {}
    if args.reuse_drafts:
        with first.open(encoding="utf-8") as f:
            saved = {int(r["issue"]): r["draft"] for r in csv.DictReader(f)}

    def judge_draft(text, findings, draft):
        metric = FaithfulnessMetric(model=judge, include_reason=False, async_mode=False)
        metric.measure(LLMTestCase(input=text, actual_output=draft,
                                   retrieval_context=[text, findings]))
        # Per-claim verdicts: "yes" (supported), "no" (contradicted), and "idk" (DeepEval
        # 2.x) or "borderline" (4.x) for claims the context doesn't mention. DeepEval's
        # own score passes those; grounded mode fails them, computed here so it works
        # on any DeepEval version.
        verdicts = [str(getattr(v.verdict, "value", v.verdict)).lower() for v in (metric.verdicts or [])]
        failing = {"no"} | ({"idk", "borderline"} if args.grounded else set())
        bad = [f"{vd}: {v.reason or ''}" for vd, v in zip(verdicts, metric.verdicts or [])
               if vd in failing]
        if args.grounded:
            score = (sum(vd == "yes" for vd in verdicts) / len(verdicts)) if verdicts else 1.0
        else:
            score = metric.score
        # Every claim with its verdict, saved for hand-checking the judge.
        judged = " || ".join(f"[{vd}] {c}" for c, vd in zip(metric.claims or [], verdicts))
        return score, bad, len(metric.claims or []), judged

    rows = []
    with get_db_connection() as conn:
        returned = dict(conn.execute(
            "SELECT issue_number, returned FROM dup_eval_results WHERE run_id = %s AND error IS NULL",
            (args.run_id,)).fetchall())
        if not returned:
            raise SystemExit(f"no rows for run {args.run_id}")
        split = conn.execute("SELECT split FROM dup_eval_results WHERE run_id = %s LIMIT 1",
                             (args.run_id,)).fetchone()[0]
        cases = [c for c in load_dup_set(conn, split) if c[0] in returned]
        if args.reuse_drafts:
            cases = [c for c in cases if c[0] in saved]
        if args.limit:
            cases = cases[: args.limit]

        # Judge check: 3 planted bad replies for the first issue; each must score < 1.
        num, title, body = cases[0][0], cases[0][1], cases[0][2]
        text = issue_text(title, body, QUERY_BODY_CHARS)
        findings = f"Findings: label = {template_label(body)}; possible duplicates: none"
        planted = [
            "Thanks! This was already fixed in uv 0.4.2, so please upgrade.",
            "Thanks for the report! We'll fix this and ship it in the next release.",
            "This looks like a duplicate of #4242; please follow that issue.",
        ]
        caught = 0
        for bad_reply in planted:
            score, why, _, _ = judge_draft(text, findings, bad_reply)
            caught += score is not None and score < 1.0
            print(f"judge check: score={score} for {bad_reply!r} {why[:1]}")
            time.sleep(args.sleep)
        print(f"judge check: caught {caught}/3 planted bad replies\n")

        for i, (num, title, body, _, kind, _) in enumerate(cases, 1):
            dups = returned[num] or []
            names = dict(conn.execute("SELECT issue_number, title FROM issues "
                                      "WHERE repo = %s AND issue_number = ANY(%s)",
                                      (REPO, dups)).fetchall()) if dups else {}
            found = "\n".join(f'- #{n} "{names.get(n, "")}"' for n in dups) or "- none"
            label = template_label(body)
            if args.reuse_drafts:
                draft = saved[num]
            else:
                draft, _ = write_draft(drafter, title, body, label, found)

            text = issue_text(title, body, QUERY_BODY_CHARS)
            findings = f"Findings: label = {label}; possible duplicates: " + \
                       (", ".join(f'#{n} "{names.get(n, "")}"' for n in dups) or "none")
            try:
                score, bad, claims, judged = judge_draft(text, findings, draft)
            except Exception as e:
                score, bad, claims, judged = None, [f"judge error: {e!r}"[:200]], 0, ""
            problems = check_reply(draft, dups)
            rows.append({"issue": num, "kind": kind, "label": label, "duplicates": dups,
                         "faithfulness": score, "claims": claims, "unsupported": " | ".join(bad),
                         "claim_verdicts": judged, "your_label": "",
                         "check_flags": "; ".join(problems), "draft": draft.replace("\n", " ")})
            print(f"[{i}/{len(cases)}] #{num} faithfulness={score} claims={claims} "
                  f"{'flags=' + '; '.join(problems) if problems else ''}"
                  f"{'  unsupported: ' + ' | '.join(bad)[:160] if bad else ''}")
            time.sleep(args.sleep)

    out.parent.mkdir(exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    scored = [r for r in rows if r["faithfulness"] is not None]
    perfect = sum(r["faithfulness"] == 1.0 for r in scored)
    mean = sum(r["faithfulness"] for r in scored) / max(1, len(scored))
    print(f"\nmode: {'grounded (unsupported claims fail)' if args.grounded else 'default (only contradictions fail)'}; "
          f"judge check caught {caught}/3 planted bad replies")
    print(f"{len(scored)} drafts judged ({len(rows) - len(scored)} judge errors): "
          f"mean faithfulness {mean:.2f}; fully faithful {perfect}/{len(scored)}; "
          f"flagged by checks {sum(bool(r['check_flags']) for r in rows)}/{len(rows)}")
    print(f"details: {out}")


if __name__ == "__main__":
    main()
