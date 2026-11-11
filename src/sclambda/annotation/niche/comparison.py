"""Comparative spatial decisions; validates declared logic, not biological truth."""
from copy import deepcopy
import json


def canonicalize(value, measurements=None):
    """Accept an exact inline copy of an assumption as its declared ID."""
    value = deepcopy(value)
    c = value.get('comparison')
    if not isinstance(c, dict) or not isinstance(c.get('assumptions'), list) or not isinstance(c.get('observations'), list):
        return value
    definitions = {a.get('id'): a for a in c['assumptions'] if isinstance(a, dict) and isinstance(a.get('id'), str)}
    for row in c['observations']:
        if isinstance(row, dict) and isinstance(row.get('assumptions'), list):
            row['assumptions'] = [a['id'] if isinstance(a, dict) and isinstance(a.get('id'), str)
                and definitions.get(a['id']) == a else a for a in row['assumptions']]
    by_id = {m['id']: m for m in measurements or []}
    for ref in references(value):
        if not isinstance(ref, dict):
            continue
        # A bare exact passage ID identifies evidence without making another
        # biological claim. Resolve it only when a single recorded source owns
        # that passage; paper IDs, invented IDs and ambiguous matches stay invalid.
        if set(ref) == {'source_id'} and isinstance(ref['source_id'], str):
            matches = [m for m in by_id.values() if m.get('kind') == 'literature'
                       and any(r.get('item') == ref['source_id'] for r in m['rows'])]
            if len(matches) == 1:
                passage = ref['source_id']
                ref.clear()
                ref.update(measurement_id=matches[0]['id'], items=[passage])
        m = by_id.get(ref.get('measurement_id')) if isinstance(ref.get('measurement_id'), str) else None
        if m:
            key = 'genes' if m.get('kind') == 'expression' else 'items'
            if isinstance(ref.get(key), str) and ref[key] in {r.get('gene', r.get('item')) for r in m['rows']}:
                ref[key] = [ref[key]]
        # A known non-expression table uses items. Accept an identical list of
        # actual items under genes without reinterpreting the measurements.
        genes = ref.get('genes')
        if (m and m.get('kind') != 'expression' and 'items' not in ref
                and _strings(genes) and genes
                and set(genes) <= {r.get('item') for r in m['rows']}):
            ref['items'] = ref.pop('genes')

    return value


def references(value):
    c = value.get('comparison') or {}
    rows = c.get('observations')
    assumptions = c.get('assumptions')
    return [ref for row in (rows if isinstance(rows, list) else []) if isinstance(row, dict)
            for ref in (row['refs'] if isinstance(row.get('refs'), list) else [])] + [
        ref for a in (assumptions if isinstance(assumptions, list) else []) if isinstance(a, dict)
        for ref in (a['evidence'] if isinstance(a.get('evidence'), list) else [])]


def _strings(value):
    return isinstance(value, list) and all(isinstance(x, str) and x.strip() for x in value)


def validate(value, previous=None):
    """Reject contradictory declarations without assigning marker identities/weights."""
    c = value.get('comparison')
    if not isinstance(c, dict):
        return ['Comparison: return the comparison object']
    candidates = c.get('candidates')
    if not _strings(candidates) or len(candidates) < 2 or len(set(candidates)) != len(candidates):
        return ['Comparison: candidates must name distinct live alternatives (at least two)']
    broad = c.get('broader_label')
    if not isinstance(broad, str) or not broad.strip() or broad in candidates:
        return ['Comparison: broader_label must omit the disputed distinction']
    rows, assumptions, selection = c.get('observations'), c.get('assumptions'), c.get('selection')
    if not isinstance(rows, list) or not rows or not isinstance(assumptions, list) or not isinstance(selection, dict):
        return ['Comparison: provide observations, assumptions and selection']
    issues, by_assumption, by_row = [], {}, {}
    for a in assumptions:
        if not isinstance(a, dict) or not isinstance(a.get('id'), str) or not a['id'].strip():
            issues.append('Comparison: assumptions need string IDs'); continue
        if a['id'] in by_assumption:
            issues.append('Comparison: assumption IDs must be unique')
        by_assumption[a['id']] = a
        if not isinstance(a.get('claim'), str) or not a['claim'].strip() or not isinstance(a.get('evidence'), list):
            issues.append('Comparison: each assumption needs a claim and evidence list (empty if unverified)')
        if a.get('check') is not None and not isinstance(a['check'], dict):
            issues.append('Comparison: assumption check must be an executable check or null')
    previous_c = (previous or {}).get('comparison') or {}
    def observation_key(row):
        return json.dumps([row.get('refs'), row.get('observation')], sort_keys=True)
    same_candidates = set(previous_c.get('candidates', [])) == set(candidates)
    old_rows = {observation_key(r): r for r in previous_c.get('observations', [])}
    declarations = {}
    for r in rows:
        if not isinstance(r, dict) or not isinstance(r.get('id'), str) or not r['id'].strip():
            issues.append('Comparison: observations need string IDs'); continue
        rid = r['id']
        if rid in by_row:
            issues.append('Comparison: observation IDs must be unique')
        by_row[rid] = r
        if not isinstance(r.get('observation'), str) or not r['observation'].strip():
            issues.append('Comparison: provide a concise observation separate from its interpretation')
        interpretation = r.get('interpretation')
        if r.get('effect') == 'favors' or isinstance(interpretation, dict):
            if not isinstance(interpretation, dict) or any(
                    not isinstance(interpretation.get(k), str) or not interpretation[k].strip()
                    for k in ('expectation', 'applicability', 'limitation')):
                issues.append('Comparison: a favoring interpretation needs expectation, applicability and limitation')
        elif not isinstance(interpretation, str) or not interpretation.strip():
            issues.append('Comparison: provide an interpretation')
        if not isinstance(r.get('refs'), list) or not r['refs']:
            issues.append('Comparison: each observation needs measurement references')
        effect, favors, premises = r.get('effect'), r.get('favors'), r.get('assumptions')
        if not isinstance(effect, str) or effect not in {'shared', 'favors', 'uninformative', 'conflicting'}:
            issues.append('Comparison: invalid observation effect')
        if not _strings(favors) or not set(favors) <= set(candidates):
            issues.append('Comparison: favors must contain candidate names'); continue
        if (effect == 'favors' and (not favors or set(favors) == set(candidates))) or (effect != 'favors' and favors):
            issues.append('Comparison: shared/uninformative/conflicting observations cannot favor a candidate; favoring all candidates is shared')
        if not _strings(premises) or not set(premises) <= set(by_assumption):
            issues.append('Comparison: observations must reference existing assumption IDs'); continue
        # Same recorded observation cannot have two incompatible declared effects.
        identity = observation_key(r)
        declaration = (effect, tuple(sorted(favors)))
        if identity in declarations and declarations[identity] != declaration:
            issues.append('Comparison: the same observation has contradictory effects')
        declarations[identity] = declaration
        old = old_rows.get(identity) if same_candidates else None
        if old and (old.get('effect'), sorted(old.get('favors', []))) != (effect, sorted(favors)):
            if not isinstance(r.get('revision'), str) or not r['revision'].strip():
                issues.append('Comparison: changing a previous observation effect requires an explicit revision')
    if issues:
        return issues
    label, basis, tradeoffs = selection.get('label'), selection.get('basis'), selection.get('tradeoffs')
    if label not in candidates + [broad]:
        issues.append('Comparison: select a candidate or the supported broader_label')
    if not _strings(basis) or not basis or not set(basis) <= set(by_row):
        return issues + ['Comparison: selection basis must reference existing observations']
    review = selection.get('review')
    if not isinstance(review, dict):
        issues.append('Comparison: selection.review must assess the strongest alternative, decisive observations and remaining issue')
    else:
        alternative = review.get('strongest_alternative')
        if alternative not in candidates or alternative == label:
            issues.append('Comparison: strongest_alternative must name a live candidate other than the selected label')
        decisive = review.get('decisive_observations')
        if not _strings(decisive) or not set(decisive) <= set(basis):
            issues.append('Comparison: decisive_observations must reference selected supporting observations')
        elif label != broad and (not decisive or any(
                by_row[rid]['effect'] != 'favors' or label not in by_row[rid]['favors']
                or alternative in by_row[rid]['favors'] for rid in decisive)):
            issues.append('Comparison: decisive observations must distinguish the selected candidate from its strongest alternative')
        elif label == broad and decisive:
            issues.append('Comparison: a shared broader label has no decisive candidate-level observations')
        if 'remaining_issue' not in review or (review['remaining_issue'] is not None and
                (not isinstance(review['remaining_issue'], str) or not review['remaining_issue'].strip())):
            issues.append('Comparison: remaining_issue must describe a potentially choice-changing uncertainty, or be null')
    if label != broad and not any(by_row[rid]['effect'] == 'favors' and label in by_row[rid]['favors'] for rid in basis):
        issues.append('Comparison: a specific winner requires distinguishing support; shared evidence can support the broader label')
    for rid in basis:
        r = by_row[rid]
        if r['effect'] not in {'shared', 'favors'} or (label != broad and r['effect'] == 'favors' and label not in r['favors']):
            issues.append('Comparison: a specific winner requires distinguishing support; shared evidence can support the broader label')
        if any(not by_assumption[aid]['evidence'] for aid in r['assumptions']):
            issues.append('Comparison: an unverified assumption cannot be necessary support; request its check or choose without it')
    contrary = {rid for rid, r in by_row.items() if label != broad and r['effect'] == 'favors' and label not in r['favors']}
    ambiguous = {rid for rid, r in by_row.items() if r['effect'] == 'conflicting'}
    if not isinstance(tradeoffs, dict):
        issues.append('Comparison: selection.tradeoffs must be an object')
    else:
        missing = contrary - set(tradeoffs)
        invalid = set(tradeoffs) - contrary - ambiguous
        if missing:
            issues.append(f'Comparison: selection.tradeoffs is missing opposing observation IDs {sorted(missing)}')
        if invalid:
            issues.append(f'Comparison: selection.tradeoffs has non-opposing/non-conflicting IDs {sorted(invalid)}')
        for against, using in tradeoffs.items():
            if not _strings(using) or not using or not set(using) <= set(basis):
                issues.append('Comparison: a tradeoff must cite selected supporting observations, not a new explanation')
            elif against in contrary and not any(by_row[rid]['effect'] == 'favors' and label in by_row[rid]['favors'] for rid in using):
                issues.append('Comparison: shared context alone cannot outweigh an observation favoring another candidate')
    return issues


def normalize(value, measurements=None):
    """Render legacy output fields from the comparison; no additional LLM narrative."""
    value = canonicalize(value, measurements)
    c = value['comparison']; s = c['selection']; label = s['label']
    by_row = {r['id']: r for r in c['observations']}
    by_measurement = {m['id']: m for m in measurements or []}
    def refs(ids):
        result = []
        for rid in ids:
            for ref in by_row[rid]['refs']:
                if ref not in result: result.append(deepcopy(ref))
        return result
    def describe(r):
        dependency = '; premises: ' + ', '.join(r['assumptions']) if r['assumptions'] else ''
        facts = []
        for ref in r['refs']:
            if not isinstance(ref, dict) or not isinstance(ref.get('measurement_id'), str):
                continue
            m = by_measurement.get(ref.get('measurement_id'), {})
            if m.get('kind') != 'expression' or not isinstance(ref.get('genes'), list):
                continue
            rows = [row for row in m['rows'] if row.get('gene') in ref.get('genes', [])]
            facts.append(json.dumps({
                'measurement_id': m['id'], 'target': m.get('target'),
                'cell_type': m.get('cell_type'), 'reference': m.get('reference'),
                'scope': m.get('scope'), 'n_target': m.get('n_target'),
                'n_reference': m.get('n_reference'), 'method': m.get('method'),
                'significance_kind': m.get('significance_kind'),
                'rows': [{k: row.get(k) for k in ('gene', 'lfc', 'proportion', 'significance')} for row in rows],
            }, ensure_ascii=False))
        # Keep the model's observation in the raw comparison for audit, but do
        # not export its transcription of expression values as measured fact.
        observed = 'Measured (from records): ' + '; '.join(facts) if facts else r['observation']
        interpretation = r['interpretation']
        if isinstance(interpretation, dict):
            interpretation = ' '.join(f'{k.capitalize()}: {interpretation[k]}'
                                      for k in ('expectation', 'applicability', 'limitation'))
        return f"{r['id']}: {observed} Interpretation ({r['effect']}{' ' + ', '.join(r['favors']) if r['favors'] else ''}): {interpretation}{dependency}"
    reason = [f"Selected {label}.", 'Selected support (shared context is not a discriminator):']
    reason.extend(describe(by_row[rid]) for rid in s['basis'])
    if s['tradeoffs']:
        reason.append('Counterevidence and declared tradeoffs:')
        reason.extend(describe(by_row[rid]) + ' Weighed against: ' + ', '.join(using) + '.' for rid, using in s['tradeoffs'].items())
    remaining = [r for r in c['observations'] if r['id'] not in s['basis'] and r['id'] not in s['tradeoffs']]
    if remaining:
        reason.append('Other comparison observations:')
        reason.extend(describe(r) for r in remaining)
    checks = value.get('pending_checks', [])[:]
    unverified = []
    for a in c['assumptions']:
        if not a['evidence']:
            unverified.append(a['claim'])
            if a.get('check') and a['check'] not in checks: checks.append(a['check'])
    if unverified:
        reason.append('Unverified assumptions, not established support: ' + '; '.join(unverified))
    review = s.get('review')
    if isinstance(review, dict):
        reason.append('Selection review: ' + json.dumps(review, ensure_ascii=False))
        if review.get('remaining_issue'):
            value['status'] = 'best_effort'
            uncertainty = value.get('uncertainty', '')
            issue = review['remaining_issue']
            value['uncertainty'] = uncertainty if issue in uncertainty else (uncertainty + '\n' + issue).strip()
    value.update(label=label, reason='\n'.join(reason), support=refs(s['basis']),
                 counterevidence=refs(s['tradeoffs']),
                 alternatives=[name for name in c['candidates'] if name != label],
                 pending_checks=checks)
    return value


def without_measurement_ids(value):
    """Allow a bounded citation-ID correction without changing any assessment."""
    if isinstance(value, dict):
        return {key: without_measurement_ids(item) for key, item in value.items() if key != 'measurement_id'}
    if isinstance(value, list):
        return [without_measurement_ids(item) for item in value]
    return value
