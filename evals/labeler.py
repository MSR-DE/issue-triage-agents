import json

LABELS = ["bug", "enhancement", "question", "documentation"]

# v2 = SYSTEM_PROMPT alone. v3 = v2 + FEWSHOT_NOTE + the examples block (only when --fewshot is used).
# The run_id records which one ran, e.g. dev-20b-v2 vs dev-20b-v3-vector.
PROMPT_VERSION = "v2"  # few-shot note: v3b (4 Oct, two-or-more matching question examples rule)

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

# Added to the system prompt ONLY when examples are given, so v2 runs stay exactly as they were.
# DRAFT: rewrite in your own words.
FEWSHOT_NOTE = """

After the issue you may also see similar earlier uv issues with the label uv's maintainers gave them.
Use them as evidence of how this repository labels this kind of report. They help most when a report
looks like a bug but similar past reports were labeled question, because maintainers judged the
behaviour intended or caused by the reporter's setup.
Only rely on an example if it describes the same kind of problem; ignore the ones that don't.
If two or more examples describe the same behaviour as this issue and were labeled question,
treat that as strong evidence that uv's maintainers consider this behaviour expected: label it
question even if the reporter used the bug report form.
A single matching example is not enough on its own, and examples about a different problem never count.
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


def build_user_message(title, body, max_chars=1500, examples=None):
    form = detect_form(body)
    body = (body or "")[:max_chars]
    msg = f"Issue form: {form}\nTitle: {title}\n\nBody:\n{body}"
    if examples:
        # One line per example: number, title, and the maintainers' label.
        lines = "\n".join(f'- #{n} "{t}" -> {label}' for n, t, label in examples)
        msg += f"\n\nSimilar past issues and the label uv's maintainers gave them:\n{lines}"
    return msg


def label_issue(client, model, title, body, examples=None):
    """One API call for one issue. Returns (label, raw_text, usage)."""
    system = SYSTEM_PROMPT + FEWSHOT_NOTE if examples else SYSTEM_PROMPT
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": build_user_message(title, body, examples=examples)},
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