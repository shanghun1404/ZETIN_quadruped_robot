#!/usr/bin/env python3
"""Verify the bundle's portable payload before building or running it."""
from __future__ import annotations

import hashlib
from pathlib import Path
import sys


def verify(directory):
    failures = []
    lines = (directory / "SHA256SUMS").read_text().splitlines()
    for line in lines:
        expected, relative = line.split("  ", 1)
        candidate = directory / relative
        if Path(relative).is_absolute() or ".." in Path(relative).parts or candidate.is_symlink():
            failures.append(f"invalid payload path: {relative}")
            continue
        if not candidate.is_file():
            failures.append(f"missing payload: {relative}")
            continue
        digest = hashlib.sha256()
        with candidate.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            failures.append(f"SHA256 mismatch: {relative}")
    return failures, len(lines)


if __name__ == "__main__":
    try:
        failures, count = verify(Path(__file__).resolve().parent)
    except (OSError, ValueError) as error:
        print(f"bundle verification failed: {error}", file=sys.stderr)
        sys.exit(1)
    if failures:
        print("\n".join(failures), file=sys.stderr)
        sys.exit(1)
    print(f"Verified {count} payload SHA256 values.")
