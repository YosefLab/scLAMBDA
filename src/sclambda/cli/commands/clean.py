"""`lambdakit clean` — report and reclaim disk after a run.

Three things accumulate, and they are not equally safe to delete:

* **rendered images** — regenerable from the data in seconds, and the largest
  part of a spatial run (they scale with `--px`, not with dataset size);
* **the dataset cache** — matrix-sized, shared by every run over that `.h5ad`,
  and rebuilt automatically on next use;
* **the run directory** — the annotations themselves. Deleting it destroys work
  that cannot be recomputed, because the reasoning that produced each label is
  not reproducible.

So this reports by default and only deletes when told which category, and it
refuses to remove a run whose annotations were never exported unless overridden.
Losing a finished annotation run to a cleanup command would be the worst failure
mode in the whole package.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

from sclambda.core import cache
from sclambda.storage.session import Session


def _size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def cmd_clean(args) -> int:
    sess = Session(args.run)
    run = Path(sess.run)

    images = run / "images"
    tmp_dirs = [run / "tmp"]           # legacy location from earlier versions
    try:
        cache_dir = sess.dataset_cache
    except Exception:  # noqa: BLE001
        cache_dir = None

    results = sess.all_results()
    exported = bool(sess.phase_state("export"))

    categories = {
        "images": (images, "rendered niche maps — regenerable"),
        "cache": (cache_dir, "shared dataset cache — rebuilt on next use"),
        "run": (run, "the whole run, including the annotations"),
    }

    out = [f"# Disk used by `{run}`", "", "| what | size | notes |", "|---|---:|---|"]
    for name, (path, note) in categories.items():
        if path is None:
            continue
        out.append(f"| {name} | {cache.human(_size(path))} | {note} |")
    legacy = sum(_size(d) for d in tmp_dirs)
    if legacy:
        out.append(f"| stale tmp | {cache.human(legacy)} | leftover from an older version — safe |")

    out += ["", f"{len(results)} annotation(s) committed; "
                f"export {'recorded' if exported else '**not recorded**'}."]

    targets: list[tuple[str, Path]] = []
    if args.images:
        targets.append(("images", images))
    if args.cache and cache_dir is not None:
        targets.append(("cache", cache_dir))
    if args.run_dir:
        targets.append(("run", run))
    if legacy and (args.images or args.cache or args.run_dir):
        for d in tmp_dirs:
            if d.exists():
                targets.append(("stale tmp", d))

    if not targets:
        out += ["", "Nothing deleted — this only reports unless you say what to remove:",
                "", "    lambdakit clean --run RUN --images   # rendered maps",
                "    lambdakit clean --run RUN --cache    # the shared dataset cache",
                "    lambdakit clean --run RUN --run-dir  # everything, annotations included"]
        print("\n".join(out))
        return 0

    # The one refusal: do not let a finished run be deleted before its results
    # are out. Images and cache are regenerable; annotations are not.
    if args.run_dir and results and not exported and not args.force:
        out += ["", "**Refusing to delete the run.** It holds "
                    f"{len(results)} committed annotation(s) and no export has been recorded, "
                    "so removing it would destroy work that cannot be recomputed.",
                "", "Export first:",
                "", f"    lambdakit export --run {run} --csv annotations.csv --xlsx annotations.xlsx",
                f"    lambdakit phase done --run {run} --name export",
                "", "or pass --force if you genuinely want the run gone."]
        print("\n".join(out))
        return 2

    freed = 0
    removed = []
    for name, path in targets:
        if path is None or not path.exists():
            continue
        freed += _size(path)
        shutil.rmtree(path, ignore_errors=True)
        removed.append(name)

    out += ["", f"Removed: {', '.join(removed) or 'nothing'} — freed {cache.human(freed)}."]
    if "cache" in removed:
        out.append("The next command over this dataset rebuilds the cache.")
    if "images" in removed:
        out.append("Re-render any map with `lambdakit spatial image`.")
    if "run" in removed:
        out.append("The run is gone; the dataset cache (if kept) still serves any new run.")
    print("\n".join(out))
    return 0
