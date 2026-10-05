"""One-off: freeze the Duplicate Finder's eval samples to CSV (dev + test).

Duplicates: a seeded random sample of the headline answer key (explicit +
single_link, 344 in uv): 20 dev (debugging and prompt work), 100 test (final runs
only), no overlap. Controls: issues nobody marked as a duplicate, from the same
date range: 10 dev, 30 test. They check that the agent can say "no duplicate".
Caveat: a few controls may be real duplicates that nobody marked.

Run once from the project root:  python -m evals.pin_duplicates
"""
import csv
import random
from pathlib import Path

from ingestion.db import get_db_connection
from evals.dataset import REPO

HERE = Path(__file__).parent
FILES = {"dev": HERE / "dup_dev.csv", "test": HERE / "dup_test.csv"}
SIZES = {"dev": (20, 10), "test": (100, 30)}     # (duplicates, controls)
SEED = 42

# Headline duplicates with all their maintainer-linked originals.
DUPLICATES = """
SELECT issue_number, array_agg(original_number ORDER BY original_number)
FROM duplicate_links
WHERE repo = %s AND tier IN ('explicit', 'single_link')
GROUP BY issue_number
ORDER BY issue_number
"""

# Never marked as a duplicate in any way: no duplicate label, not closed as
# duplicate, not in duplicate_links. Non-bot, same date range as the duplicates.
CONTROLS = """
SELECT i.issue_number
FROM issues i
WHERE i.repo = %(repo)s
  AND i.user_type <> 'Bot'
  AND i.state_reason IS DISTINCT FROM 'duplicate'
  AND NOT EXISTS (SELECT 1 FROM issue_labels l
                  WHERE l.repo = i.repo AND l.issue_number = i.issue_number
                    AND l.label_name = 'duplicate')
  AND NOT EXISTS (SELECT 1 FROM duplicate_links d
                  WHERE d.repo = i.repo AND d.issue_number = i.issue_number)
  AND i.created_at BETWEEN
      (SELECT min(j.created_at) FROM issues j JOIN duplicate_links d
         ON d.repo = j.repo AND d.issue_number = j.issue_number WHERE d.repo = %(repo)s)
  AND (SELECT max(j.created_at) FROM issues j JOIN duplicate_links d
         ON d.repo = j.repo AND d.issue_number = j.issue_number WHERE d.repo = %(repo)s)
ORDER BY i.issue_number
"""


def main():
    if any(p.exists() for p in FILES.values()):
        raise SystemExit("dup_dev.csv / dup_test.csv already exist, not overwriting")

    with get_db_connection() as conn:
        dups = conn.execute(DUPLICATES, (REPO,)).fetchall()
        controls = [r[0] for r in conn.execute(CONTROLS, {"repo": REPO}).fetchall()]

    rng = random.Random(SEED)
    n_dup = sum(d for d, _ in SIZES.values())
    n_ctl = sum(c for _, c in SIZES.values())
    picked_dups = rng.sample(dups, n_dup)
    picked_ctls = rng.sample(controls, n_ctl)

    for split in ("dev", "test"):
        n_d, n_c = SIZES[split]
        split_dups, picked_dups = picked_dups[:n_d], picked_dups[n_d:]
        split_ctls, picked_ctls = picked_ctls[:n_c], picked_ctls[n_c:]
        with FILES[split].open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["issue_number", "kind", "originals"])
            for number, originals in sorted(split_dups):
                w.writerow([number, "duplicate", ";".join(map(str, originals))])
            for number in sorted(split_ctls):
                w.writerow([number, "control", ""])
        print(f"{FILES[split].name}: {n_d} duplicates + {n_c} controls")

    print(f"(pool: {len(dups)} headline duplicates, {len(controls)} possible controls)")


if __name__ == "__main__":
    main()
