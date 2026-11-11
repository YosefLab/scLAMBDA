"""Atomic replacement for run files, with unique names for concurrent writers."""
from contextlib import contextmanager
from pathlib import Path
import os
import tempfile


@contextmanager
def atomic_output(path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=path.suffix, dir=path.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        yield temporary
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_text(path: Path, text: str):
    with atomic_output(path) as temporary:
        temporary.write_text(text, encoding="utf-8")
