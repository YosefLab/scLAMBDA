"""Species-aware gene-name resolution against a versioned NCBI gene_info snapshot.

No cell identities or marker panels are encoded here. Exact panel names win;
ambiguous aliases never select a feature, even if only one candidate is measured.
"""
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
import gzip
import hashlib
import io
import os
import re
import tempfile
import urllib.request

TAXA = {'human': '9606', 'homo sapiens': '9606', 'mouse': '10090',
        'mus musculus': '10090', 'rat': '10116', 'rattus norvegicus': '10116'}
FILES = {'9606': 'Mammalia/Homo_sapiens.gene_info.gz',
         '10090': 'Mammalia/Mus_musculus.gene_info.gz',
         '10116': 'Mammalia/Rattus_norvegicus.gene_info.gz'}


def cache_root(path=None):
    return Path(path or os.environ.get('SCLAMBDA_EVIDENCE_CACHE',
                str(Path.home() / '.cache' / 'sclambda' / 'evidence')))


def atomic_bytes(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix='.' + path.name)
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(data)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


@lru_cache(maxsize=8)
def _snapshot(taxon, directory, download):
    path = Path(directory) / f'{taxon}.gene_info.gz'
    if path.exists():
        return path, None
    if not download or taxon not in FILES:
        return None, 'No species gene-name snapshot available; exact panel matches remain usable'
    try:
        url = 'https://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/' + FILES[taxon]
        with urllib.request.urlopen(url, timeout=20) as response:
            data = response.read(32 * 1024 * 1024 + 1)
        if len(data) > 32 * 1024 * 1024 or not gzip.decompress(data).startswith(b'#tax_id'):
            raise ValueError('Invalid or oversized NCBI gene_info snapshot')
        atomic_bytes(path, data)
        return path, None
    except Exception as exc:
        return None, f'Gene-name source unavailable: {type(exc).__name__}: {exc}'


def _key(name):
    return re.sub(r"[^a-z0-9]", "", name.casefold())


@lru_cache(maxsize=8)
def _load(path, stamp, taxon):
    payload = Path(path).read_bytes()
    text = gzip.decompress(payload).decode() if payload[:2] == b'\x1f\x8b' else payload.decode()
    aliases = defaultdict(set)
    canonical = set()
    for line in io.StringIO(text):
        if line.startswith('#'):
            continue
        cols = line.rstrip('\n').split('\t')
        if len(cols) < 5 or cols[0] != taxon:
            continue
        symbol = cols[2]
        canonical.add(symbol)
        names = [symbol] + cols[4].split('|')
        names += [cols[i] for i in (8, 10, 11) if len(cols) > i]
        names += cols[13].split('|') if len(cols) > 13 else []
        for name in names:
            if name and name != '-':
                aliases[_key(name)].add(symbol)
    if not canonical:
        raise ValueError(f'No gene_info rows for taxonomy {taxon}')
    return aliases, canonical, {'source': 'NCBI gene_info', 'taxon': taxon, 'resolver_version': 2,
                               'path': path, 'sha256': hashlib.sha256(payload).hexdigest()}


def resolve_genes(requested, panel, *, organism='human', alias_path=None,
                  cache_dir=None, download=True):
    download = download and os.environ.get("SCLAMBDA_GENE_ALIAS_DOWNLOAD", "1") != "0"
    panel = set(map(str, panel))
    taxon = TAXA.get(str(organism).lower(), str(organism) if str(organism).isdigit() else None)
    aliases, canonical, version = {}, set(), None
    error = None
    if any(g not in panel for g in requested):
        try:
            path = Path(alias_path) if alias_path else None
            if path is None:
                path, error = _snapshot(taxon, str(cache_root(cache_dir) / 'gene_names'), download)
            if path is not None:
                aliases, canonical, version = _load(str(path.resolve()), path.stat().st_mtime_ns, taxon)
        except Exception as exc:
            error = f'Gene-name source unavailable: {type(exc).__name__}: {exc}'
    rows = []
    for name in requested:
        row = {'requested': name, 'feature': None, 'candidates': []}
        if name in panel:
            row.update(status='exact', feature=name)
        else:
            candidates = {name} if name in canonical else aliases.get(_key(name), set())
            row['candidates'] = sorted(candidates)
            if len(candidates) > 1:
                row['status'] = 'ambiguous'
            elif candidates:
                symbol = next(iter(candidates))
                row.update(status='alias' if symbol in panel else 'not_measured',
                           feature=symbol if symbol in panel else None)
            else:
                row['status'] = 'source_unavailable' if error else 'unrecognized'
        rows.append(row)
    return {'rows': rows, 'version': version, 'source_error': error,
            'genes': sorted({r['feature'] for r in rows if r['feature']})}
