from collections import Counter

from ingestion.db import get_db_connection
from evals.dataset import load_dev_set, load_test_set, TYPE_LABELS
from evals.baseline import template_label
from evals.metrics import report


def main():
    with get_db_connection() as conn:
        sets = {"dev": load_dev_set(conn), "test": load_test_set(conn)}

    for name, issues in sets.items():
        labels = [label for _, _, _, label in issues]
        top_label, top_count = Counter(labels).most_common(1)[0]

        print(f"\n===== {name}: {len(issues)} issues =====")
        print(f"majority baseline (always '{top_label}'): {top_count / len(labels):.1%}\n")
        print("template baseline:")
        pairs = [(label, template_label(body)) for _, _, body, label in issues]
        report(pairs, TYPE_LABELS)


if __name__ == "__main__":
    main()