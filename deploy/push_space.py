"""Upload the live demo to its Hugging Face Space (Docker), and optionally set its secrets.

    python -m deploy.push_space             # code only; CI runs this on every push to main
    python -m deploy.push_space --secrets   # once, from your PC: also copies the Space's
                                            # secrets from .env (values are never printed)

Needs HF_TOKEN (a Hugging Face write token) and HF_SPACE (e.g. "your-name/uv-issue-triage"),
from .env locally or from the CI secrets. Without HF_TOKEN it prints a note and exits 0, so
CI stays green before the Space exists.

What --secrets sets on the Space:
    DATABASE_URL      <- DEMO_DATABASE_URL (the Neon database; never your local one)
    GROQ_API_KEY      <- GROQ_API_KEY
    LANGFUSE_*        <- copied if present (optional: traces of the demo's runs)
    DAILY_LIVE_RUNS   variable, default 5 (live runs per day, ~8.5K gpt-oss-20b tokens each;
                      cached results are free). Change it in the Space settings any time.
"""
import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
# demo file -> name in the Space
DEMO_FILES = {"app.py": "app.py", "Dockerfile": "Dockerfile",
              "requirements.txt": "requirements.txt", "SPACE_README.md": "README.md"}
PACKAGES = ["triage", "retrieval", "ingestion", "evals"]   # the app imports from all four
OPTIONAL_SECRETS = ["LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST",
                    "LANGFUSE_BASE_URL"]


def stage(folder):
    """Copy what the Space needs into `folder`: the demo files and the packages' .py files
    (no data, no results, no .env)."""
    for src, name in DEMO_FILES.items():
        shutil.copy(ROOT / "demo" / src, folder / name)
    for package in PACKAGES:
        (folder / package).mkdir()
        for f in (ROOT / package).glob("*.py"):
            shutil.copy(f, folder / package / f.name)


def set_secrets(api, space):
    secrets = {"DATABASE_URL": os.getenv("DEMO_DATABASE_URL"),
               "GROQ_API_KEY": os.getenv("GROQ_API_KEY")}
    missing = [k for k, v in secrets.items() if not v]
    if missing:
        sys.exit(f"missing from .env: {', '.join('DEMO_DATABASE_URL' if k == 'DATABASE_URL' else k for k in missing)}")
    secrets.update({k: os.environ[k] for k in OPTIONAL_SECRETS if os.getenv(k)})
    for key, value in secrets.items():
        api.add_space_secret(space, key, value)
    api.add_space_variable(space, "DAILY_LIVE_RUNS", os.getenv("DAILY_LIVE_RUNS", "5"))
    print(f"secrets set on {space}: {', '.join(secrets)} (+ variable DAILY_LIVE_RUNS)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--secrets", action="store_true", help="also set the Space's secrets from .env")
    args = p.parse_args()
    load_dotenv()
    token, space = os.getenv("HF_TOKEN"), os.getenv("HF_SPACE")
    if not token or not space:
        print("HF_TOKEN / HF_SPACE not set: skipping the Space deploy.")
        return

    from huggingface_hub import HfApi
    api = HfApi(token=token)
    api.create_repo(space, repo_type="space", space_sdk="docker", exist_ok=True)
    if args.secrets:
        set_secrets(api, space)
    with tempfile.TemporaryDirectory() as tmp:
        stage(Path(tmp))
        sha = os.getenv("GITHUB_SHA", "")[:7]
        info = api.upload_folder(
            repo_id=space, repo_type="space", folder_path=tmp,
            commit_message=f"Deploy from GitHub {sha}".strip() if sha else "Deploy from local copy",
            delete_patterns=["*.py"])     # .py files deleted in the repo disappear here too
    print(f"uploaded to https://huggingface.co/spaces/{space} ({info.oid[:7] if info else 'no changes'});"
          " the Space rebuilds itself (a few minutes).")


if __name__ == "__main__":
    main()
