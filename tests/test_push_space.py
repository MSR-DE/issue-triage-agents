"""deploy/push_space.py with a fake Hugging Face API (no network, no token):
the Space gets exactly the files the app needs, the database secret is the demo (Neon)
database and never the local one, and secret values are never printed."""
import sys
import types
from pathlib import Path

import pytest

import deploy.push_space as push


class FakeApi:
    calls = []

    def __init__(self, token):
        self.token = token

    def create_repo(self, repo_id, **kw):
        FakeApi.calls.append(("create_repo", repo_id, kw))

    def add_space_secret(self, repo_id, key, value):
        FakeApi.calls.append(("secret", key, value))

    def add_space_variable(self, repo_id, key, value):
        FakeApi.calls.append(("variable", key, value))

    def upload_folder(self, repo_id, folder_path, **kw):
        files = sorted(str(p.relative_to(folder_path)).replace("\\", "/")
                       for p in Path(folder_path).rglob("*") if p.is_file())
        readme = (Path(folder_path) / "README.md").read_text(encoding="utf-8")
        FakeApi.calls.append(("upload", files, readme, kw))
        return types.SimpleNamespace(oid="abcdef1234")


@pytest.fixture
def run(monkeypatch, capsys):
    FakeApi.calls = []
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(HfApi=FakeApi))
    monkeypatch.setattr(push, "load_dotenv", lambda: None)
    monkeypatch.setenv("HF_SPACE", "someone/uv-triage")
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    monkeypatch.setenv("DATABASE_URL", "postgresql://localhost:5433/LOCAL_DB")
    monkeypatch.setenv("DEMO_DATABASE_URL", "postgresql://neon.example/DEMO_DB")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_SECRET_VALUE")

    def go(*argv):
        monkeypatch.setattr(sys, "argv", ["push_space", *argv])
        push.main()
        return FakeApi.calls, capsys.readouterr().out
    return go


def test_uploads_exactly_what_the_app_needs(run):
    calls, _ = run()
    _, files, readme, kw = next(c for c in calls if c[0] == "upload")
    for needed in ("app.py", "Dockerfile", "requirements.txt", "README.md",
                   "triage/graph.py", "triage/investigator.py", "retrieval/fix_index.py",
                   "retrieval/vector.py", "ingestion/db.py", "evals/labeler.py"):
        assert needed in files
    assert all(f.endswith(".py") or f in ("Dockerfile", "requirements.txt", "README.md")
               for f in files)                      # no .env, data or results
    assert "sdk: docker" in readme and "app_port: 8501" in readme
    assert kw["delete_patterns"] == ["*.py"]
    assert not any(c[0] == "secret" for c in calls)  # code-only by default (what CI runs)


def test_secrets_use_the_demo_database_and_are_never_printed(run):
    calls, out = run("--secrets")
    secrets = {c[1]: c[2] for c in calls if c[0] == "secret"}
    assert secrets["DATABASE_URL"] == "postgresql://neon.example/DEMO_DB"
    assert secrets["GROQ_API_KEY"] == "gsk_SECRET_VALUE"
    assert "DEMO_DB" not in out and "SECRET_VALUE" not in out and "hf_test" not in out


def test_missing_demo_database_stops_before_anything_is_set(run, monkeypatch):
    monkeypatch.delenv("DEMO_DATABASE_URL")
    with pytest.raises(SystemExit):
        run("--secrets")
    assert not any(c[0] in ("secret", "upload") for c in FakeApi.calls)


def test_skips_without_a_token(run, monkeypatch):
    monkeypatch.delenv("HF_TOKEN")
    calls, out = run()
    assert calls == [] and "skipping" in out
