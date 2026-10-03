"""Turn raw issue text into something search can use.

clean_body: remove the issue-form boilerplate. uv's forms add the same headings
("### Summary", "### Platform", ...) to thousands of issues. Those words say nothing
about the actual problem, so they would only make every issue look alike.

tokenize: split text into lowercase words. BM25 works on these words.
"""
import re
import snowballstemmer
STEMMER = snowballstemmer.stemmer("english")
# Headings uv's issue forms insert automatically (checked on real issues).
# Headings people write themselves, like "### Reproduction", are real content: keep them.
FORM_HEADINGS = re.compile(
    r"^###\s*(Summary|Platform|Version|Python version|Problem Statement|"
    r"Proposed Solution|Alternatives I Considered|Example)\s*$",
    re.IGNORECASE | re.MULTILINE,   # MULTILINE: ^ and $ match at every line, not just the whole text
)
HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)  # hidden form instructions; DOTALL lets . cross newlines
NO_RESPONSE = "_No response_"                         # what the form writes for an empty field

# A token is a run of letters, digits or underscores; everything else splits.
#   "uv pip install --user"  -> ["uv", "pip", "install", "user"]
#   "UV_PROJECT_ENVIRONMENT" -> ["uv_project_environment"]   (kept whole: env var names are great keywords)
#   "3.12.2"                 -> ["3", "12", "2"]              (versions get split; acceptable for now)
TOKEN = re.compile(r"[a-z0-9_]+")


def clean_body(body):
    body = body or ""                      # body can be NULL in the database
    body = HTML_COMMENT.sub(" ", body)
    body = FORM_HEADINGS.sub(" ", body)
    return body.replace(NO_RESPONSE, " ")


def issue_text(title, body, max_body_chars=None):
    """Title + cleaned body: the text we search over, and later embed."""
    body = clean_body(body)
    if max_body_chars is not None:
        body = body[:max_body_chars]
    return f"{title}\n{body}"


def tokenize(text, stem=False):
    tokens = TOKEN.findall(text.lower())
    if stem:
        tokens = STEMMER.stemWords(tokens)
    return tokens