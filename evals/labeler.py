import json

LABELS = ["bug", "enhancement", "question", "documentation"]

PROMPT_VERSION = "v2"

SYSTEM_PROMPT = """You triage GitHub issues for uv, a Python package and project manager.
Pick exactly one label:
- bug: uv behaves incorrectly: a crash, panic, wrong result, or a regression from documented or previous behaviour.
- enhancement: a request for new behaviour, a new option or flag, or a change to how something works by design.
- question: the reporter needs help using uv, is unsure whether something is expected, or hits a problem caused by their setup or a misunderstanding rather than a defect in uv.
- documentation: the docs are wrong, missing or unclear.

Each issue comes with the issue form the reporter used, detected from its headings.
In this repository the form is strong evidence:
- bug report form: usually labeled bug.
- question form: usually labeled question.
- feature request form: usually labeled enhancement.
- none detected: decide from the text alone.
Keep the label the form suggests unless the text clearly contradicts it. Examples of a clear contradiction:
a bug report where the reporter is really asking how to do something or whether behaviour is intended (question);
a feature request that actually describes a crash or wrong result (bug);
any form where the whole point is that the docs are wrong or missing (documentation).
Reply with JSON only: {"label": "<one of the four labels>"}"""

# Structured output: the API forces the reply to match this schema.
SCHEMA = {
    "name": "issue_label",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {"label": {"type": "string", "enum": LABELS}},
        "required": ["label"],
        "additionalProperties": False,
    },
}


def detect_form(body):
    """Which uv issue form was used, from its headings (same logic as the baseline)."""
    body = body or ""
    if "### Example" in body or "### Problem Statement" in body:
        return "feature request"
    if "### Platform" in body:
        if "### Python version" in body:
            return "bug report"
        return "question"
    return "none detected"


def build_user_message(title, body, max_chars=1500):
    form = detect_form(body)
    body = (body or "")[:max_chars]
    return f"Issue form: {form}\nTitle: {title}\n\nBody:\n{body}"


def label_issue(client, model, title, body):
    """One API call for one issue. Returns (label, raw_text, usage)."""
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_message(title, body)},
        ],
        response_format={"type": "json_schema", "json_schema": SCHEMA},
        reasoning_effort="low",
        temperature=0,
    )
    raw = resp.choices[0].message.content
    label = json.loads(raw)["label"]
    if label not in LABELS:
        raise ValueError(f"unexpected label: {label!r}")
    return label, raw, resp.usage