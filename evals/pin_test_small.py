"""One-off: pin a smaller Duplicate Finder test sample (evals/dup_test_small.csv).

The full test sample (99 scorable duplicates + 30 controls) costs ~570K tokens, about
3 days of gpt-oss-20b's free tier. This draws 50 duplicates + 15 controls from it at
random (seed 42), before any test results exist, so it stays an unbiased test set at
about half the cost. Duplicates whose only original is a meta issue are left out.

    python -m evals.pin_test_small
"""
import csv
import random
from pathlib import Path

from ingestion.duplicates import META_ISSUES

HERE = Path(__file__).parent
SRC, DST = HERE / "dup_test.csv", HERE / "dup_test_small.csv"
N_DUP, N_CTL, SEED = 50, 15, 42


def main():
    if DST.exists():
        raise SystemExit(f"{DST.name} already exists; refusing to overwrite a pinned set")
    with SRC.open(newline="") as f:
        rows = list(csv.DictReader(f))
    dups = [r for r in rows if r["kind"] == "duplicate"
            and {int(x) for x in r["originals"].split(";") if x} - META_ISSUES]
    ctls = [r for r in rows if r["kind"] == "control"]
    rng = random.Random(SEED)
    picked = rng.sample(dups, N_DUP) + rng.sample(ctls, N_CTL)
    picked.sort(key=lambda r: int(r["issue_number"]))
    with DST.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["issue_number", "kind", "originals"])
        w.writeheader()
        w.writerows(picked)
    print(f"wrote {DST.name}: {N_DUP} duplicates + {N_CTL} controls "
          f"(from {len(dups)} scorable duplicates + {len(ctls)} controls, seed {SEED})")


if __name__ == "__main__":
    main()
