"""Run directory: the on-disk state that replaces sc-lambda's in-process objects.

sc-lambda held everything in one long-lived ``Accessor`` shared across threads.
A skill instead runs many short Bash processes, some of them in parallel
subagents, so state has to be a file. Layout::

    <run>/manifest.json        dataset + run parameters
         /overlay.parquet      per-cell cluster and annotation columns
         /genes.txt            measured gene panel
         /catalog.json         cell-type or anatomical-compartment catalog
         /dossiers/<id>.json   per-cluster belief state
         /results/<id>.json    committed label + justification + transcript
         /ontology.json        optional Cell Ontology term per committed label
         /images/<id>_*.png    rendered niche maps

Per-cluster files are what make parallel subagents safe: two clusters never
write the same path. The overlay is the one shared object, and only the
orchestrator mutates it (between depth waves), mirroring sc-lambda's rule
against locking inside ``map_parallel``.

What is *not* here is anything derived purely from the `.h5ad`: the
memory-mapped matrix and the reference statistics live in a dataset-scoped
cache (see :mod:`sclambda.core.cache`) so a second run over the same file reuses
them instead of spending minutes rebuilding a duplicate.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sclambda.core import cache
from sclambda.storage.atomic import atomic_output, write_text

SCHEMA_VERSION = 1

PROGRESS = "progress.json"
MANIFEST = "manifest.json"
OVERLAY = "overlay.parquet"
OVERLAY_FALLBACK = "overlay.csv.gz"
REFSTATS = "refstats.npz"
GENES = "genes.txt"
CATALOG = "catalog.json"
ONTOLOGY = "ontology.json"


class SessionError(RuntimeError):
    pass


_write_atomic = write_text


def default_run_dir(h5ad: str | Path) -> Path:
    return Path(".lambda") / Path(h5ad).stem


class Session:
    """Handle on a run directory.

    The accessor is built lazily — most commands need it, but ``clusters``,
    ``dossier`` and ``export`` do not, and building it opens the h5ad and can
    materialise a memory-mapped CSR copy of the matrix.
    """

    def __init__(self, run_dir: str | Path):
        self.run = Path(run_dir)
        self._manifest: dict[str, Any] | None = None
        self._accessor = None

    # ---------------- manifest ----------------

    @property
    def manifest(self) -> dict[str, Any]:
        if self._manifest is None:
            path = self.run / MANIFEST
            if not path.exists():
                raise SessionError(
                    f"No run found at {self.run} (missing {MANIFEST}). Run `lambdakit init` first."
                )
            manifest = json.loads(path.read_text())
            if not isinstance(manifest, dict):
                raise SessionError("Run manifest must be a JSON object")
            version = manifest.get("schema_version", 0)
            if version not in (0, SCHEMA_VERSION):
                raise SessionError(f"Unsupported run schema {version}; update sclambda to read this run")
            self._manifest = manifest
        return self._manifest

    def save_manifest(self, data: dict[str, Any] | None = None) -> None:
        if data is not None:
            self._manifest = data
        self._manifest.setdefault("schema_version", SCHEMA_VERSION)
        self.run.mkdir(parents=True, exist_ok=True)
        _write_atomic(self.run / MANIFEST, json.dumps(self._manifest, indent=2))

    def set(self, key: str, value: Any) -> None:
        self.manifest[key] = value
        self.save_manifest()

    def get(self, key: str, default: Any = None) -> Any:
        return self.manifest.get(key, default)

    # ---------------- accessor ----------------

    @property
    def accessor(self):
        if self._accessor is None:
            self._accessor = self._build_accessor()
        return self._accessor

    def _build_accessor(self):
        from sclambda.core.accessor import Accessor, DEFAULT_GENE_FILTER

        m = self.manifest
        self.validate_source()
        acc = Accessor(
            path=m["h5ad"],
            gene_col=m.get("gene_col"),
            gene_filter=DEFAULT_GENE_FILTER,
            embedding_key=m.get("embedding_key") or "X_pca",
            chunk_size=int(m.get("chunk_size", 50_000)),
            target_sum=float(m.get("target_sum", 1e4)),
            use_mmap_csr=bool(m.get("use_mmap_csr", True)),
            cache_dir=m.get("cache_root"),
        )
        self._load_refstats(acc)
        self._load_overlay(acc)
        # Built by `lambdakit init`; makes gene-panel DE read columns, not every row.
        acc.attach_column_cache()
        if m.get("mode") == "spatial":
            from sclambda.annotation.niche.populations import ensure_spatial_cell_types
            generated = m.get("cell_type_col_generated", False) or not bool(m.get("cell_type_col"))
            m["cell_type_col"] = ensure_spatial_cell_types(acc, None if generated else m["cell_type_col"])
            m["cell_type_col_generated"] = generated
        return acc

    # ---------------- dataset cache ----------------

    def validate_source(self) -> None:
        expected = self.manifest.get("dataset_fingerprint")
        if expected and cache.fingerprint(self.manifest["h5ad"]) != expected:
            raise SessionError("The source dataset changed since this run was initialized; create a new run")

    @property
    def dataset_cache(self) -> Path:
        """Where this dataset's matrix and statistics live.

        Scoped to the `.h5ad`, not to the run, so a second analysis of the same
        file reuses both instead of rebuilding them.
        """
        m = self.manifest
        return cache.dataset_dir(m["h5ad"], root=m.get("cache_root"))

    @property
    def _stats_key(self) -> str:
        m = self.manifest
        return cache.stats_key(
            gene_col=m.get("gene_col"),
            gene_filter="default",
            target_sum=float(m.get("target_sum", 1e4)),
        )

    # ---------------- reference stats ----------------

    def refstats_path(self) -> Path:
        return cache.stats_path(self.dataset_cache, self._stats_key)

    def refstats_cached(self) -> bool:
        return self.refstats_path().exists()

    def _load_refstats(self, acc) -> None:
        path = self.refstats_path()
        if not path.exists():
            legacy = self.run / REFSTATS  # runs created before the cache moved
            if not legacy.exists():
                return
            path = legacy
        with np.load(path) as z:
            total = z["total_counts"] if "total_counts" in z.files else None
            acc.set_reference_stats(z["mean"], z["var"], int(z["n"]), total,
                                    dropout=z["dropout"] if "dropout" in z.files else None)

    def save_refstats(self, acc) -> None:
        """Persist the full-dataset pass so no later command or run repeats it."""
        path = self.refstats_path()
        with cache.build_lock(self.dataset_cache, "stats"):
            cache.atomic_save_npz(
                path,
                mean=acc.ref_mean,
                var=acc.ref_var,
                n=np.array(acc.ref_n),
                total_counts=acc.total_counts,
                dropout=acc.cell_dropout_rate,
            )

    # ---------------- overlay ----------------

    def _overlay_path(self) -> Path | None:
        stored = self.manifest.get("overlay_file")
        if stored:
            if stored not in (OVERLAY, OVERLAY_FALLBACK):
                raise SessionError(f"Invalid overlay filename: {stored}")
            path = self.run / stored
            if not path.is_file():
                raise SessionError(f"Missing overlay: {stored}")
            return path
        for name in (OVERLAY, OVERLAY_FALLBACK):
            if (self.run / name).exists():
                return self.run / name
        return None

    def _load_overlay(self, acc) -> None:
        path = self._overlay_path()
        if path is None:
            return
        df = self.read_overlay()
        try:
            acc.restore_overlay(df)
        except ValueError as exc:
            raise SessionError(str(exc)) from exc

    def save_overlay(self, acc=None) -> Path:
        acc = acc or self.accessor
        df = acc.overlay
        current = self._overlay_path()
        # Keep an existing format. This avoids a multi-file format switch during
        # a checkpoint; new runs prefer Parquet and fall back only for a missing engine.
        path = current or self.run / OVERLAY
        if path.name == OVERLAY_FALLBACK:
            with atomic_output(path) as tmp:
                df.to_csv(tmp, compression="gzip")
        else:
            try:
                with atomic_output(path) as tmp:
                    df.to_parquet(tmp)
            except ImportError:
                if current is not None:
                    raise SessionError("Reading/writing this run requires pyarrow; install sclambda")
                path = self.run / OVERLAY_FALLBACK
                with atomic_output(path) as tmp:
                    df.to_csv(tmp, compression="gzip")
        self.set("overlay_file", path.name)
        return path

    def read_overlay(self) -> pd.DataFrame:
        """Read an overlay without loading expression data."""
        self.validate_source()
        path = self._overlay_path()
        if path is None:
            raise SessionError("No overlay yet — run `lambdakit init`.")
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        return df.set_index(df.columns[0]).rename_axis(None)

    # ---------------- gene panel ----------------

    def save_genes(self, genes) -> None:
        _write_atomic(self.run / GENES, "\n".join(str(g) for g in genes))

    def read_genes(self) -> list[str]:
        path = self.run / GENES
        if not path.exists():
            return []
        return [g for g in path.read_text().splitlines() if g]

    # ---------------- catalog ----------------

    def save_catalog(self, entries: list[dict]) -> None:
        _write_atomic(self.run / CATALOG, json.dumps(entries, indent=2))

    def read_catalog(self) -> list[dict]:
        path = self.run / CATALOG
        if not path.exists():
            return []
        return json.loads(path.read_text())

    def read_ontology(self) -> dict:
        path = self.run / ONTOLOGY
        return json.loads(path.read_text()) if path.exists() else {"terms": {}}

    def write_ontology(self, data: dict) -> None:
        if self.phase_state("export") == "done":
            self.mark_phase("export", "pending", "Ontology terms changed; export again")
        _write_atomic(self.run / ONTOLOGY, json.dumps(data, indent=2))

    def current_ontology_terms(self, results: dict[str, dict] | None = None) -> dict[str, dict]:
        """Recorded terms whose label still matches the committed result."""
        results = self.all_results() if results is None else results
        return {cid: entry for cid, entry in self.read_ontology().get("terms", {}).items()
                if cid in results and results[cid].get("label") == entry.get("label")}

    # ---------------- per-cluster stores ----------------

    def _cluster_file(self, kind: str, cluster_id: str) -> Path:
        d = self.run / kind
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{_safe(cluster_id)}.json"

    def read_dossier_dict(self, cluster_id: str) -> dict | None:
        path = self._cluster_file("dossiers", cluster_id)
        return json.loads(path.read_text()) if path.exists() else None

    def write_dossier_dict(self, cluster_id: str, data: dict) -> Path:
        path = self._cluster_file("dossiers", cluster_id)
        _write_atomic(path, json.dumps(data, indent=2))
        return path

    def read_result(self, cluster_id: str) -> dict | None:
        path = self._cluster_file("results", cluster_id)
        return json.loads(path.read_text()) if path.exists() else None

    def write_result(self, cluster_id: str, data: dict) -> Path:
        path = self._cluster_file("results", cluster_id)
        previous = json.loads(path.read_text()) if path.exists() else None
        if previous != data and self.phase_state("export") == "done":
            # Invalidate before writing, so interruption cannot certify stale exports.
            self.mark_phase("export", "pending", "Annotations changed; export again")
        _write_atomic(path, json.dumps(data, indent=2))
        return path

    def all_results(self) -> dict[str, dict]:
        d = self.run / "results"
        if not d.exists():
            return {}
        out = {}
        for f in sorted(d.glob("*.json")):
            data = json.loads(f.read_text())
            out[data.get("cluster_id", f.stem)] = data
        return out

    def current_results(self) -> dict[str, dict]:
        """Committed results that still match the current critic decisions."""
        from sclambda.annotation.decisions import result_matches_dossier
        dossiers = self.all_dossiers()
        return {cid: result for cid, result in self.all_results().items()
                if result_matches_dossier(result, dossiers.get(cid))}

    def uncommitted_decisions(self) -> list[str]:
        from sclambda.annotation.decisions import result_matches_dossier
        results = self.all_results()
        return [cid for cid, dossier in self.all_dossiers().items()
                if (dossier.get("annotation") or dossier.get("critic_decision") or dossier.get("decision_mode")) and not result_matches_dossier(results.get(cid), dossier)]

    def child_clusters(self, cluster_id: str, overlay=None) -> list[str]:
        """Existing direct children, at this parent's depth and within its cells."""
        overlay = self.read_overlay() if overlay is None else overlay
        depth = cluster_id.count("_") + 1 if cluster_id else 0
        column = f"L{depth + 1}"
        if column not in overlay:
            return []
        if cluster_id:
            parent_column = f"L{depth}"
            if parent_column not in overlay:
                return []
            values = overlay.loc[overlay[parent_column] == cluster_id, column]
        else:
            values = overlay[column]
        return sorted({str(c) for c in values.dropna() if str(c) not in ("", "nan", "None")})

    def all_dossiers(self) -> dict[str, dict]:
        d = self.run / "dossiers"
        if not d.exists():
            return {}
        out = {}
        for f in sorted(d.glob("*.json")):
            data = json.loads(f.read_text())
            out[data.get("cluster_id", f.stem)] = data
        return out

    def image_dir(self) -> Path:
        d = self.run / "images"
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ---------------- phase ledger ----------------
    #
    # Per-cluster files record what has been *annotated*. They cannot record
    # what has been *done to the run as a whole* — whether the cross-cluster
    # pass ran, whether the spatial final passes ran, whether results were
    # exported. Those are one-shot phases with no per-cluster artefact, so a
    # restart would silently skip or repeat them. This ledger is where they go.

    def read_progress(self) -> dict:
        path = self.run / PROGRESS
        if not path.exists():
            return {"phases": {}}
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            return {"phases": {}}   # never let a damaged ledger block a restart

    def mark_phase(self, name: str, state: str = "done", note: str = "") -> dict:
        data = self.read_progress()
        data.setdefault("phases", {})[name] = {
            "state": state,
            "note": note,
            "at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        _write_atomic(self.run / PROGRESS, json.dumps(data, indent=2))
        return data

    def phase_state(self, name: str) -> str:
        return (self.read_progress().get("phases", {}).get(name) or {}).get("state", "")

    def stale_clusters(self, minutes: int = 30) -> list[str]:
        """Clusters with uncommitted work, untouched for a while.

        After an interrupted run these are the ones to resume: a dossier
        exists without a matching result, and nothing has touched it since.
        """
        from sclambda.annotation.decisions import result_matches_dossier

        out = []
        d = self.run / "dossiers"
        if not d.exists():
            return out
        cutoff = time.time() - minutes * 60
        for f in sorted(d.glob("*.json")):
            dossier = json.loads(f.read_text())
            cid = dossier.get("cluster_id", f.stem)
            if result_matches_dossier(self.read_result(cid), dossier):
                continue
            if f.stat().st_mtime < cutoff:
                out.append(cid)
        return out

    # ---------------- transcript ----------------

    def append_transcript(self, cluster_id: str, entry: dict) -> None:
        """Append one step to a cluster's transcript.

        Replaces sc-lambda's ``_prompt_execution_log``, which harvested
        prompt/response pairs off node outputs to build the run's markdown
        history.
        """
        d = self.run / "transcripts"
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{_safe(cluster_id)}.jsonl"
        with path.open("a") as fh:
            fh.write(json.dumps(entry) + "\n")

    def read_transcript(self, cluster_id: str) -> list[dict]:
        path = self.run / "transcripts" / f"{_safe(cluster_id)}.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _safe(cluster_id: str) -> str:
    """Filesystem-safe cluster id ('' is the root cluster in Leiden mode)."""
    return "_root_" if cluster_id == "" else str(cluster_id).replace("/", "-")
