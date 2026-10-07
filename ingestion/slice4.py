"""All Slice 4 data in one go (each step can also run alone; all are re-runnable):
tables -> releases -> PRs (from stored pages) -> comments -> fix_links answer key.

    python -m ingestion.slice4
"""
from ingestion import comments, fixes, prs, releases, schema_slice4


def main():
    schema_slice4.main()
    if not releases.fetch_releases():
        return
    releases.load_releases()
    prs.load_prs()
    if not comments.fetch_comments():
        print("Comments stopped early; re-run to resume. fix_links not rebuilt.")
        return
    fixes.build()


if __name__ == "__main__":
    main()
