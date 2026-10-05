"""Output checks on a drafted reply, in code, before a maintainer sees it.

A reply may only mention the issue numbers the agents found. It must not contain
links, @mentions, commands or code, or promise a fix or a release. Anything else
is flagged (security layer 5: an injected issue can't smuggle these into a reply
without the reviewer seeing a warning).
"""
import re

MAX_CHARS = 800

ISSUE_REF = re.compile(r"(?:#|/issues/|/pull/)(\d+)")
URL = re.compile(r"https?://|www\.|\b[\w-]+\.(?:com|io|org|net|dev|xyz|sh|ru|cn)\b", re.I)
MENTION = re.compile(r"(?<![\w.`])@[A-Za-z0-9][A-Za-z0-9-]*")
# Code blocks, or inline code with shell syntax or download/delete commands. A plain
# command name in backticks (`uv sync`) is fine: flagging it put warnings on most
# clean drafts in the first injection run (10 of 18).
CODE = re.compile(r"```|`[^`\n]*(?:[|;&<>]|\$\(|\b(?:curl|wget|sudo|rm|chmod|eval|bash|sh)\b)[^`\n]*`")
COMMAND = re.compile(r"^\s*(?:\$|>|sudo |curl |wget |pip |uv |python )", re.M | re.I)
PROMISE = re.compile(
    r"\b(?:will be (?:fixed|released|shipped|resolved)|we(?:'ll| will) (?:fix|release|ship)"
    r"|next release|fixed (?:soon|shortly)|eta\b|by (?:tomorrow|next week))", re.I)


def check_reply(reply, allowed_numbers):
    """List of problems (empty = passes). allowed_numbers: issues the agents found."""
    problems = []
    extra = sorted({int(n) for n in ISSUE_REF.findall(reply)} - set(allowed_numbers))
    if extra:
        problems.append(f"mentions issues the agents didn't find: {extra}")
    if URL.search(reply):
        problems.append("contains a link")
    if MENTION.search(reply):
        problems.append(f"contains an @mention: {MENTION.search(reply).group()}")
    if CODE.search(reply) or COMMAND.search(reply):
        problems.append("contains code or a command")
    if PROMISE.search(reply):
        problems.append(f"promises something: {PROMISE.search(reply).group()!r}")
    if len(reply) > MAX_CHARS:
        problems.append(f"too long ({len(reply)} chars)")
    if not reply.strip():
        problems.append("empty")
    return problems
