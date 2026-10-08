"""Test setup. The project's modules read secrets from .env at import time and stop if
they're missing; tests never connect to anything, so dummy values are enough."""
import os
import sys
from pathlib import Path

os.environ.setdefault("DATABASE_URL", "postgresql://tests@localhost:1/none")
os.environ.setdefault("GITHUB_TOKEN", "tests-dummy")
os.environ.setdefault("GROQ_API_KEY", "tests-dummy")
os.environ.setdefault("LANGFUSE_TRACING_ENABLED", "false")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
