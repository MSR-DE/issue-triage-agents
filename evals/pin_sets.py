"""One-off: freeze the test and dev sets (issue number + gold label) to CSV files.

After this, load_test_set / load_dev_set read the files instead of re-running the
"most recent 300" query, so new issues from a later sync or a maintainer relabel
can't change the sets. Run once from the project root:  python -m evals.pin_sets
"""
import csv
from pathlib import Path

from ingestion.db import get_db_connection
from evals.dataset import DEV_SQL, TEST_SQL, _load

HERE = Path(__file__).parent
FILES = {"test": (HERE / "test_issues.csv", TEST_SQL), "dev": (HERE / "dev_issues.csv", DEV_SQL)}


def main():
    with get_db_connection() as conn:
        for name, (path, sql) in FILES.items():
            if path.exists():
                print(f"{path.name} already exists, not overwriting")
                continue
            rows = _load(conn, sql)
            with path.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["issue_number", "label"])
                for issue_number, _title, _body, label in rows:
                    writer.writerow([issue_number, label])
            print(f"{path.name}: {len(rows)} issues")


if __name__ == "__main__":
    main()
