def template_label(body):
    body = body or ""
    if "### Example" in body or "### Problem Statement" in body:
        return "enhancement"
    if "### Platform" in body:
        if "### Python version" in body:
            return "bug"
        return "question"
    return "bug"