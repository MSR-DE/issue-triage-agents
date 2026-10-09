"""Output checks on a drafted reply, in code, before a maintainer sees it.

A reply may only mention the issue numbers the agents found. It must not contain
links, @mentions, commands or code, promise a fix or a release, commit the
team to any work, or mention passwords, tokens or other secrets. Anything else is flagged (security layer 5: an injected issue
can't smuggle these into a reply without the reviewer seeing a warning).
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
# Promises of a fix/release/timeline, and ANY statement of what "we" will do: the bot
# speaks for no one. A verb list ("we'll look/fix/add...") missed "We'll take a look" and
# "We'll note" in the 8 Oct re-draft, so every "we'll / we will / we're going to" counts.
# ' or ’ (models use both).
PROMISE = re.compile(
    r"\b(?:will be (?:fixed|released|shipped|resolved|addressed|prioriti[sz]ed)"
    r"|we(?:['\u2019]ll| will| are going to|['\u2019]re going to)\b"
    r"|(?:look|looking) into (?:it|this)|take a look|backlog|roadmap|with the team"
    r"|next release|fixed (?:soon|shortly)|eta\b|by (?:tomorrow|next week))", re.I)
# A reply must never ask for (or talk about) secrets. Added 9 Oct after the injection
# re-run: the "credential-phish" attack made the protected drafter ask the reporter to
# paste their index password or token, and no check flagged it. On the 112 saved
# replies it flags that one plus 2 that restate a reporter's own token problem: a
# reviewer glances at those, which is the point of a flag.
SECRET = re.compile(r"\b(?:passwords?|passphrases?|tokens?|api[ _-]?keys?|secrets?|credentials?"
                    r"|private keys?|ssh keys?|access keys?)\b", re.I)


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
    if SECRET.search(reply):
        problems.append(f"mentions a password, token or other secret: {SECRET.search(reply).group()!r}")
    if len(reply) > MAX_CHARS:
        problems.append(f"too long ({len(reply)} chars)")
    if not reply.strip():
        problems.append("empty")
    return problems
