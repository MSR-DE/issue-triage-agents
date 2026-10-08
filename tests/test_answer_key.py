"""The "already fixed?" answer key: which maintainer comments say a PR fixed an issue."""
import pytest

from ingestion.fixes import HEDGE, VERSION_ONLY, fixing_prs, normalize
from ingestion.prs import closed_issues


@pytest.mark.parametrize("comment, prs", [
    ("This was fixed in #11329, please upgrade.", {11329}),
    ("Fixed already in #11329 will be out today", {11329}),
    ("I believe this was fixed by https://github.com/astral-sh/uv/pull/8030", {8030}),
    ("Resolved via [#123](https://github.com/astral-sh/uv/pull/123).", {123}),
    ("#456 should fix this.", {456}),
    ("astral-sh/uv#789 fixes it", {789}),
    ("We fixed this in #4242 and #4243", {4242, 4243}),
    ("fixed by #1, #2 & #3.", {1, 2, 3}),
    ("I fixed this with PR #77", {77}),
    ("#88 also fixed this", {88}),
    ("fixed in #5 but #6 is unrelated", {5}),
])
def test_fix_comments_found(comment, prs):
    assert fixing_prs(comment) == prs


@pytest.mark.parametrize("comment", [
    "> fixed in #999\nquoted reporter text",        # a quote is the reporter's words
    "```\nfixed in #55\n```",                       # code block
    "Ok, now that #10430 is addressed, I re-ran this.",
    "That's on the todo-list, e.g., #9867.",
    "Duplicate of #12",
    "Fixed in uv 0.5.22",                            # no PR to score against
])
def test_non_fix_comments_ignored(comment):
    assert fixing_prs(comment) == set()


# The hand check of all 34 already-fixed links (7 Oct 2026) found 7 wrong; all were hedged.
@pytest.mark.parametrize("comment", [
    "I thought we fixed this in #14325",
    "Annoying, I thought I fixed this in #8191 — thanks for the report!",
    "I had hoped to have fixed that in #8665, but apparently this branch is still wrong.",
    "the one purportedly fixed by #14331",
    "This would also be resolved by #3049 / #1511 I presume?",
    "This is partially fixed by #2401",
    "careful not to break the things we fixed in #3130",
])
def test_hedged_comments_are_skipped(comment):
    assert HEDGE.search(normalize(comment))


@pytest.mark.parametrize("comment", [
    "Fixed already in #11329",
    "This was fixed in #12160 (not yet released).",
    "I believe this was fixed in #8191 and will be released today",
])
def test_real_fixes_are_not_hedged(comment):
    assert not HEDGE.search(normalize(comment))


def test_version_only_comments():
    assert VERSION_ONLY.search(normalize("This was fixed in 0.5.22"))
    assert not VERSION_ONLY.search(normalize("fixed in #3"))


@pytest.mark.parametrize("body, issues", [
    ("Fixes #12", [12]),
    ("closes: #3 and fixes https://github.com/astral-sh/uv/issues/45", [3, 45]),
    ("Resolves astral-sh/uv#7", [7]),
    ("Closes #9.", [9]),
    ("fixed the #8 thing", []),
    ("prefix #10", []),
    (None, []),
])
def test_pr_closing_keywords(body, issues):
    assert closed_issues(body) == issues
