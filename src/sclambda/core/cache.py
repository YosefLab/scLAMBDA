"""Dataset-scoped cache.

Two kinds of state get confused easily, so they live in different places:

* **Run state** — clusters, dossiers, results, images. Belongs to one analysis
  and lives in the run directory.
* **Dataset state** — the memory-mapped CSR copy of the matrix, per-gene
  reference statistics, per-cell totals and dropout rates. These are a pure
  function of the `.h5ad` file, so they belong to the *file*, and every run over
  that file should share them.

Keeping the second kind in the run directory meant a second run on the same
dataset rebuilt everything — minutes of I/O and a full duplicate of the matrix
on disk. Here it is keyed on the file's identity instead:

    <cache root>/<stem>-<fingerprint>/
        meta.json          the fingerprint this directory was built from
        csr_{data,indices,indptr}.npy
        stats-<params>.npz reference statistics, keyed by the params that change them

The fingerprint covers path, size and mtime, so editing the `.h5ad` produces a
new directory rather than silently reusing stale arrays.

Cache root resolution, in order: an explicit `--cache-dir`, `$LAMBDA_CACHE_DIR`,
`$XDG_CACHE_HOME/lambdakit`, then `~/.cache/lambdakit`.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from contextlib import contextmanager
from pathlib import Path

import numpy as np

META = "meta.json"


def resolve_root(explicit: str | Path | None = None) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    env = os.environ.get("LAMBDA_CACHE_DIR")
    if env:
        return Path(env).expanduser().resolve()
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return (base / "lambdakit").resolve()


def fingerprint(h5ad: str | Path, x_key: str = "X") -> dict:
    """Identify a dataset by content-affecting properties, not just its name.

    Path, size and mtime together are cheap and catch the cases that matter: a
    regenerated file, a different file with the same basename, a moved file.
    Hashing the bytes would be more correct and far too slow for a matrix this
    size.
    """
    p = Path(h5ad).resolve()
    st = p.stat()
    payload = f"{p}|{st.st_size}|{st.st_mtime_ns}|{x_key}"
    return {
        "path": str(p),
        "size": st.st_size,
        "mtime_ns": st.st_mtime_ns,
        "x_key": x_key,
        "key": hashlib.sha1(payload.encode()).hexdigest()[:12],
    }


def dataset_dir(h5ad: str | Path, root: str | Path | None = None, x_key: str = "X") -> Path:
    fp = fingerprint(h5ad, x_key)
    d = resolve_root(root) / f"{Path(fp['path']).stem}-{fp['key']}"
    d.mkdir(parents=True, exist_ok=True)
    meta = d / META
    if not meta.exists():
        _atomic_write_text(meta, json.dumps(fp, indent=2))
    return d


def stats_key(*, gene_col: str | None, gene_filter: str, target_sum: float) -> str:
    """Reference statistics depend on more than the file.

    Mean and variance depend on whether the matrix gets normalised and to what
    target sum; the per-cell dropout rate additionally depends on which genes
    the filter keeps, which depends on the gene column. Two runs differing in
    any of these need separate statistics, so the parameters go in the filename
    rather than silently sharing one file.
    """
    payload = f"{gene_col}|{gene_filter}|{target_sum}"
    return hashlib.sha1(payload.encode()).hexdigest()[:10]


def stats_path(dataset_dir_: Path, key: str) -> Path:
    return Path(dataset_dir_) / f"stats-{key}.npz"


# ---------------- atomic writes ----------------
#
# Parallel subagents can reach a cold cache at the same moment. Writing to a
# unique temp name and renaming means a reader never sees a half-written array;
# the worst case is two processes doing the same work, not a corrupt file.

def atomic_save_npy(path: Path, array: np.ndarray) -> None:
    from sclambda.storage.atomic import atomic_output
    with atomic_output(path) as tmp:
        with tmp.open("wb") as stream:
            np.save(stream, array)


def atomic_save_npz(path: Path, **arrays) -> None:
    from sclambda.storage.atomic import atomic_output
    with atomic_output(path) as tmp:
        with tmp.open("wb") as stream:
            np.savez_compressed(stream, **arrays)


def _atomic_write_text(path: Path, text: str) -> None:
    from sclambda.storage.atomic import write_text
    write_text(path, text)


@contextmanager
def build_lock(directory: Path, name: str = "build"):
    """Advisory lock so concurrent workers don't duplicate an expensive build.

    Best-effort: if locking is unavailable (some network filesystems), the block
    still runs. Correctness rests on the atomic renames above, not on this.
    """
    lock_path = Path(directory) / f".{name}.lock"
    fh = None
    try:
        import fcntl
        fh = open(lock_path, "w")
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
    except Exception:  # noqa: BLE001
        if fh is not None:
            fh.close()
            fh = None
    try:
        yield
    finally:
        if fh is not None:
            try:
                import fcntl
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            except Exception:  # noqa: BLE001
                pass
            fh.close()


# ---------------- inspection / management ----------------

def entries(root: str | Path | None = None) -> list[dict]:
    """Every cached dataset, with its size and whether its source still matches."""
    r = resolve_root(root)
    if not r.exists():
        return []
    out = []
    for d in sorted(r.iterdir()):
        if not d.is_dir():
            continue
        meta_file = d / META
        meta = json.loads(meta_file.read_text()) if meta_file.exists() else {}
        total = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
        src = Path(meta.get("path", ""))
        if not src.exists():
            status = "source missing"
        else:
            st = src.stat()
            fresh = st.st_size == meta.get("size") and st.st_mtime_ns == meta.get("mtime_ns")
            status = "current" if fresh else "stale (source changed)"
        out.append({
            "dir": d, "bytes": total, "status": status,
            "source": meta.get("path", "?"),
            "has_matrix": (d / "csr_data.npy").exists(),
            "n_stats": len(list(d.glob("stats-*.npz"))),
        })
    return out


def remove(directory: str | Path) -> int:
    d = Path(directory)
    total = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
    shutil.rmtree(d, ignore_errors=True)
    return total


def human(n: int) -> str:
    x = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if x < 1024 or unit == "TB":
            return f"{x:.0f} {unit}" if unit == "B" else f"{x:.1f} {unit}"
        x /= 1024
    return f"{x:.1f} TB"
