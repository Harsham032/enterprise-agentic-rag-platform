#!/usr/bin/env python
"""Scan tracked files for credential-shaped strings before publishing.

Deliberately simple and fast so it can run as a pre-push check and in CI. It
looks for assignments to credential-named variables that carry a non-placeholder
value, and for the common fixed-prefix key formats.

Exit code 1 means something needs a human look. False positives are expected;
add them to ALLOWED_SUBSTRINGS once reviewed.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

CREDENTIAL_ASSIGNMENT = re.compile(r"""(?ix)
    \b(api[_-]?key|secret|passwd|password|access[_-]?token|auth[_-]?token
       |private[_-]?key|client[_-]?secret|database[_-]?url)\b
    \s*[:=]\s*
    ["']?(?P<value>[^\s"',;#]{8,})["']?
    """)

FIXED_PREFIX_KEYS = re.compile(
    r"(?:sk-[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}"
    r"|AKIA[0-9A-Z]{16}|AIza[0-9A-Za-z_-]{30,}|xox[baprs]-[A-Za-z0-9-]{10,})"
)

PRIVATE_KEY_BLOCK = re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |PGP )?PRIVATE KEY-----")

# Values that are obviously not real credentials.
PLACEHOLDER_VALUES = re.compile(
    r"(?i)^(?:your[_-]|change[_-]?me|placeholder|example|dummy|xxx+|\.\.\.|<[^>]+>|\$\{[^}]+\}|"
    r"none|null|true|false|\"\"|''|str|string|field\(|os\.environ|settings\.)"
)

ALLOWED_SUBSTRINGS = (
    "sqlite:///",
    "postgresql+psycopg://rag:CHANGE_ME@",
    "redis://localhost",
    "http://localhost",
    "users.noreply.github.com",
)

SKIP_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".pdf", ".joblib", ".npz", ".ipynb_checkpoints"}


def tracked_files() -> list[Path]:
    """Files git is tracking, which is exactly what a push would publish."""
    result = subprocess.run(["git", "ls-files"], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return sorted(p for p in Path().rglob("*") if p.is_file() and ".git" not in p.parts)
    return [Path(line) for line in result.stdout.splitlines() if line]


def is_placeholder(value: str) -> bool:
    return bool(PLACEHOLDER_VALUES.match(value)) or any(
        allowed in value for allowed in ALLOWED_SUBSTRINGS
    )


def scan_file(path: Path) -> list[tuple[int, str]]:
    findings: list[tuple[int, str]] = []
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return findings

    for number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if any(allowed in stripped for allowed in ALLOWED_SUBSTRINGS):
            continue
        if PRIVATE_KEY_BLOCK.search(line):
            findings.append((number, "private key block"))
            continue
        if FIXED_PREFIX_KEYS.search(line):
            findings.append((number, "fixed-prefix credential"))
            continue
        match = CREDENTIAL_ASSIGNMENT.search(line)
        if match and not is_placeholder(match.group("value")):
            findings.append((number, f"credential-shaped assignment: {stripped[:90]}"))
    return findings


def main() -> int:
    total = 0
    for path in tracked_files():
        if path.suffix.lower() in SKIP_SUFFIXES or path.name == Path(__file__).name:
            continue
        for number, description in scan_file(path):
            print(f"{path}:{number}: {description}")
            total += 1

    if Path(".env").exists():
        result = subprocess.run(
            ["git", "check-ignore", "-q", ".env"], capture_output=True, check=False
        )
        if result.returncode != 0:
            print(".env exists and is not ignored by git")
            total += 1

    print(f"\n{total} finding(s)")
    return 1 if total else 0


if __name__ == "__main__":
    raise SystemExit(main())
