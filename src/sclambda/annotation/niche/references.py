"""Bounded retrieval of attributed literature passages, without an LLM intermediary."""
from datetime import datetime, timezone
import hashlib
import json
import re
from urllib.parse import urlencode
from urllib.request import urlopen
import xml.etree.ElementTree as ET

from sclambda.analysis.gene_names import atomic_bytes, cache_root
from .measurements import identify

API = 'https://www.ebi.ac.uk/europepmc/webservices/rest/'


def _fetch(url, limit=8 * 1024 * 1024):
    with urlopen(url, timeout=15) as response:
        data = response.read(limit + 1)
    if len(data) > limit:
        raise ValueError('Reference response exceeds size limit')
    return data


def _plain(text):
    return re.sub(r'\s+', ' ', re.sub(r'<[^>]+>', ' ', text or '')).strip()


def lookup_reference(cid, query, *, organism, tissue, terms=(), cache_dir=None):
    """Retrieve at most three candidate papers and exact passages, not a verdict.

    Search results exclude review/preprint publication types. The model must
    still check species, study design and whether a passage supports its claim.
    Full-text paragraphs are preferred when the requested terms occur there.
    """
    if not isinstance(query, str) or not query.strip() or len(query) > 600:
        raise ValueError('Reference query must contain 1–600 characters')
    if not isinstance(terms, (list, tuple)) or any(not isinstance(t, str) for t in terms):
        raise ValueError('Reference terms must be strings')
    terms = [t.casefold() for t in terms[:12] if t.strip()]
    search = f'({query}) AND ({organism}) AND ({tissue}) NOT PUB_TYPE:review NOT PUB_TYPE:preprint'
    url = API + 'search?' + urlencode({'query': search, 'format': 'json', 'resultType': 'core', 'pageSize': 6})
    key = hashlib.sha256(json.dumps([2, url, terms], sort_keys=True).encode()).hexdigest()
    path = cache_root(cache_dir) / 'references' / f'{key}.json'
    if path.exists():
        data = json.loads(path.read_text())
    else:
        payload = json.loads(_fetch(url))
        papers = []
        for result in payload.get('resultList', {}).get('result', [])[:6]:
            if len(papers) == 3:
                break
            pubtypes = result.get('pubTypeList', {}).get('pubType', [])
            if result.get('source') == 'PPR' or any(any(word in t.lower() for word in ('review', 'comment', 'editorial', 'preprint')) for t in pubtypes):
                continue
            source, pid = result.get('source'), result.get('id')
            if not source or not pid:
                continue
            abstract = _plain(result.get('abstractText'))
            passages = []
            fulltext_error = None
            pmcid = result.get('pmcid', '')
            if re.fullmatch(r'PMC\d+', pmcid) and result.get('isOpenAccess') == 'Y' and terms:
                try:
                    tree = ET.fromstring(_fetch(API + pmcid + '/fullTextXML'))
                    paragraphs = [_plain(' '.join(p.itertext())) for p in tree.findall('.//body//p')]
                    matches = [(sum(term in p.casefold() for term in terms), i, p) for i, p in enumerate(paragraphs)]
                    matches = sorted((m for m in matches if m[0]), key=lambda m: (-m[0], m[1]))[:2]
                    passages = [{'text': p[:2600], 'section': 'full-text paragraph',
                                 'truncated': len(p) > 2600} for _, _, p in matches]
                except Exception as exc:
                    fulltext_error = f'{type(exc).__name__}: {exc}'
            if not passages and abstract:
                passages = [{'text': abstract[:4000], 'section': 'abstract', 'truncated': len(abstract) > 4000}]
            if not passages:
                continue  # Titles alone are not biological support.
            papers.append({'source_id': f'{source}:{pid}', 'title': result.get('title'),
                'url': f'https://europepmc.org/article/{source}/{pid}', 'doi': result.get('doi'),
                'year': result.get('pubYear'), 'publication_types': result.get('pubTypeList', {}).get('pubType', []),
                'passages': passages, 'fulltext_error': fulltext_error})
        data = {'query': query, 'search': search, 'request_url': url,
                'retrieved_at': datetime.now(timezone.utc).isoformat(), 'papers': papers}
        atomic_bytes(path, json.dumps(data, ensure_ascii=False).encode())
    rows = []
    for paper in data['papers']:
        for i, p in enumerate(paper['passages']):
            rows.append({'item': f"{paper['source_id']}:{i + 1}", 'text': p['text'],
                         **{k: v for k, v in paper.items() if k != 'passages'},
                         'section': p['section'], 'truncated': p['truncated']})
    value = {'kind': 'literature', 'target': cid, 'source': 'Europe PMC retrieval; candidate literature, not a biological verdict',
             'query': query, 'search': data['search'], 'request_url': data['request_url'],
             'retrieved_at': data['retrieved_at'], 'scope': {'organism': organism, 'tissue': tissue},
             'rows': rows}
    return identify(value)
