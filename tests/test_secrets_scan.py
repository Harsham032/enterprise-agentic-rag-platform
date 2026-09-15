"""The credential scanner that gates every push.

A scanner is only worth running if it catches real secrets and stays quiet on
ordinary code. A noisy one gets ignored, which is worse than not having one.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def scanner() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "secrets_scan", REPO_ROOT / "scripts" / "secrets_scan.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["secrets_scan"] = module
    spec.loader.exec_module(module)
    return module


def _scan(scanner: ModuleType, tmp_path: Path, content: str) -> list[tuple[int, str]]:
    path = tmp_path / "candidate.txt"
    path.write_text(content, encoding="utf-8")
    return scanner.scan_file(path)


@pytest.mark.parametrize(
    "line",
    [
        'API_KEY = "sk-abcdefghijklmnopqrstuvwxyz012345"',  # pragma: allowlist secret
        "DATABASE_URL=postgresql://user:hunter2secret@db.internal/app",  # pragma: allowlist secret
        'password = "correct-horse-battery-staple"',  # pragma: allowlist secret
        "CLIENT_SECRET=aVeryRealLookingSecretValue",  # pragma: allowlist secret
        "-----BEGIN RSA PRIVATE KEY-----",  # pragma: allowlist secret
        "token = 'ghp_abcdefghijklmnopqrstuvwxyz0123456789'",  # pragma: allowlist secret
        "AWS_ACCESS_KEY=AKIAIOSFODNN7EXAMPLE",  # pragma: allowlist secret
    ],
)
def test_real_credentials_are_caught(scanner: ModuleType, tmp_path: Path, line: str) -> None:
    assert _scan(scanner, tmp_path, line + "\n"), line


@pytest.mark.parametrize(
    "line",
    [
        # Passing a variable through is not a hardcoded credential.
        "self._client = OpenAI(api_key=api_key)",
        "api_key=api_key",
        "def __init__(self, api_key: str, *, model: str = 'x') -> None:",
        # Lowercase identifiers that merely contain a keyword.
        "tokens = set(content_tokens(text))",
        "max_tokens=max_tokens,",
        "sentence_tokens = len(sentence.split())",
        # Placeholders and empty values.
        "RAG_OPENAI_API_KEY=",
        "RAG_SEC_USER_AGENT=your-name your-email@example.com",
        'export RAG_NCBI_EMAIL="your.email@example.com"',
        "password = os.environ['APP_PASSWORD']",
        "RAG_DATABASE_URL=sqlite:///data/processed/rag_platform.sqlite3",
    ],
)
def test_ordinary_code_is_not_flagged(scanner: ModuleType, tmp_path: Path, line: str) -> None:
    assert _scan(scanner, tmp_path, line + "\n") == [], line


def test_repository_is_clean(scanner: ModuleType) -> None:
    """The committed tree must carry no credentials. This is the gate CI runs."""
    findings: list[str] = []
    for path in scanner.tracked_files():
        if path.suffix.lower() in scanner.SKIP_SUFFIXES or path.name == "secrets_scan.py":
            continue
        findings.extend(f"{path}:{number}: {why}" for number, why in scanner.scan_file(path))
    assert findings == []


def test_an_empty_git_index_does_not_silently_pass(
    scanner: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Before the first commit `git ls-files` is empty; scanning nothing would be a false pass."""
    import subprocess

    class EmptyResult:
        returncode = 0
        stdout = ""

    monkeypatch.setattr(subprocess, "run", lambda *_args, **_kwargs: EmptyResult())
    monkeypatch.setattr(scanner, "working_tree_files", lambda: [Path("sentinel")])
    assert scanner.tracked_files() == [Path("sentinel")]


def test_env_file_is_ignored_by_git() -> None:
    assert ".env" in (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")


def test_env_example_contains_no_real_values() -> None:
    """The template must carry placeholders only; it is committed."""
    for line in (REPO_ROOT / ".env.example").read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        if any(word in key.upper() for word in ("KEY", "SECRET", "PASSWORD", "TOKEN")):
            assert value == "", f"{key} must be blank in the template, found {value!r}"


def test_allowlist_marker_exempts_a_reviewed_line(scanner: ModuleType, tmp_path: Path) -> None:
    secret = 'API_KEY = "sk-abcdefghijklmnopqrstuvwxyz012345"'  # pragma: allowlist secret
    assert _scan(scanner, tmp_path, secret + "\n")
    assert _scan(scanner, tmp_path, f"{secret}  # {scanner.ALLOWLIST_MARKER}\n") == []
