"""One-off: freeze the Investigator's eval samples to CSV (evals/fix_tune.csv, fix_eval.csv).

uv has only ~34 issues that were already fixed when reported (fix_links tier
already_fixed), too few for a dev/test split. So (decided 7 Oct 2026):
  tune  3 already-fixed + 2 controls: look at these by hand while writing the prompts
  eval  everything else, run ONCE with the frozen prompts; reported as small-n with
        uncertainty ranges (evals/score_inv_run.py prints Wilson 95% intervals)
Positives: already_fixed issues minus EXCLUDE (answer-key links a hand check found
wrong). Controls: real problems NOT fixed when reported (fixed later: a fixed_later
link, or a "Fixes #N" PR merged after the issue opened; no already_fixed link), same
date range. Seeded random, no overlap.

    python -m ingestion.fixes show 40      # check every link first, fill EXCLUDE
    python -m evals.pin_fixes
"""
import csv
import random
from pathlib import Path

from ingestion.db import get_db_connection
from evals.dataset import REPO

HERE = Path(__file__).parent
FILES = {"tune": HERE / "fix_tune.csv", "eval": HERE / "fix_eval.csv"}
TUNE = (3, 2)            # (positives, controls)
EVAL_CONTROLS = 30
SEED = 42
# Hand check of all 34 links, 7 Oct 2026 (7 wrong, all hedged; ingestion/fixes.py now
# skips such comments): 16165 "purportedly fixed", 2409 "partially fixed", 14904 / 8228
# "I thought (we) fixed", 8887 "had hoped to have fixed", 3923 "not to break what we
# fixed", 3248 "I presume?". The pinned CSVs were made before the full check; these
# rows were removed from them by hand (key errors, decided before any agent run).
EXCLUDE = {16165, 2409, 14904, 8228, 8887, 3923, 3248}

POSITIVES = """
SELECT f.issue_number, array_agg(f.pr_number ORDER BY f.pr_number)
FROM fix_links f JOIN issues i ON i.repo = f.repo AND i.issue_number = f.issue_number
WHERE f.repo = %s AND f.tier = 'already_fixed' AND i.user_type <> 'Bot'
GROUP BY f.issue_number ORDER BY f.issue_number
"""
CONTROLS = """
SELECT DISTINCT i.issue_number
FROM issues i
WHERE i.repo = %(repo)s AND i.user_type <> 'Bot'
  AND (EXISTS (SELECT 1 FROM fix_links f WHERE f.repo = i.repo
               AND f.issue_number = i.issue_number AND f.tier = 'fixed_later')
       OR EXISTS (SELECT 1 FROM pr_closes c JOIN pull_requests p
                    ON p.repo = c.repo AND p.pr_number = c.pr_number
                  WHERE c.repo = i.repo AND c.issue_number = i.issue_number
                    AND p.merged_at > i.created_at))
  AND NOT EXISTS (SELECT 1 FROM fix_links f WHERE f.repo = i.repo
                  AND f.issue_number = i.issue_number AND f.tier = 'already_fixed')
  AND i.created_at BETWEEN
      (SELECT min(j.created_at) FROM issues j JOIN fix_links f
         ON f.repo = j.repo AND f.issue_number = j.issue_number
       WHERE f.repo = %(repo)s AND f.tier = 'already_fixed')
  AND (SELECT max(j.created_at) FROM issues j JOIN fix_links f
         ON f.repo = j.repo AND f.issue_number = j.issue_number
       WHERE f.repo = %(repo)s AND f.tier = 'already_fixed')
ORDER BY i.issue_number
"""


def write(path, positives, controls):
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["issue_number", "kind", "fix_prs"])
        for number, prs in sorted(positives):
            w.writerow([number, "fixed", ";".join(map(str, prs))])
        for number in sorted(controls):
            w.writerow([number, "control", ""])
    print(f"{path.name}: {len(positives)} already-fixed + {len(controls)} controls")


def main():
    if any(p.exists() for p in FILES.values()):
        raise SystemExit("fix_tune.csv / fix_eval.csv already exist, not overwriting")
    with get_db_connection() as conn:
        positives = [p for p in conn.execute(POSITIVES, (REPO,)).fetchall() if p[0] not in EXCLUDE]
        controls = [r[0] for r in conn.execute(CONTROLS, {"repo": REPO}).fetchall()]

    rng = random.Random(SEED)
    rng.shuffle(positives)
    rng.shuffle(controls)
    n_p, n_c = TUNE
    write(FILES["tune"], positives[:n_p], controls[:n_c])
    write(FILES["eval"], positives[n_p:], controls[n_c:n_c + EVAL_CONTROLS])
    print(f"(pool: {len(positives)} already-fixed after excluding {sorted(EXCLUDE)}, "
          f"{len(controls)} possible controls)")


if __name__ == "__main__":
    main()
