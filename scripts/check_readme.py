#!/usr/bin/env python3
"""Validate the README: every relative link resolves to a real file,
required sections exist, and external URLs are well-formed."""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"

REQUIRED_SECTIONS = [
    "Why this project matters",
    "Run it yourself on Kaggle",
    "What AuK can do",
    "How the notebook works",
    "## Examples",
]

LINK_RE = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
URL_RE = re.compile(r"^https?://[^\s]+$")

errors = []


def main() -> int:
    text = README.read_text()
    for section in REQUIRED_SECTIONS:
        if section not in text:
            errors.append(f"README missing section: {section!r}")

    for target in LINK_RE.findall(text):
        if "data:" in target or target.startswith("#"):
            continue  # embedded image / anchor
        if URL_RE.match(target):
            continue  # external URL, checked by humans / markdown lint
        path = (ROOT / target).resolve()
        if not path.is_file():
            errors.append(f"README link target missing: {target}")

    if errors:
        print("README check failed:")
        for e in errors:
            print(f"  - {e}")
        return 1
    print("README check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())