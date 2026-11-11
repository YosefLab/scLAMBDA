"""Cell Ontology terms from an OBO file, and n-gram matching against them."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path


def normalize(text: str) -> str:
    """Normalize text for matching."""
    return text.lower().strip().replace("-", " ").replace("_", " ").replace(".", " ")


def _ngrams(text: str, n: int) -> set[str]:
    text = normalize(text)
    return {text[i : i + n] for i in range(len(text) - n + 1)}


class NgramIndex:
    """IDF-weighted character n-gram overlap, with a bonus for containment.

    Built once per corpus so that many queries do not recount the corpus.
    """

    def __init__(self, corpus: list[str], n: int = 2):
        self.corpus = corpus
        self.n = n
        self._norm = [normalize(t) for t in corpus]
        self._grams = [_ngrams(t, n) for t in corpus]
        freq = Counter(g for grams in self._grams for g in grams)
        total = len(corpus)
        self._idf = {g: math.log(total / (1 + c)) for g, c in freq.items()}
        self._unseen_idf = math.log(total) if total else 0.0

    def search(self, query: str, top: int = 10) -> list[tuple[str, float]]:
        q_norm = normalize(query)
        q_grams = _ngrams(query, self.n)
        idf = lambda g: self._idf.get(g, self._unseen_idf)  # noqa: E731
        max_possible = sum(idf(g) for g in q_grams)
        results = []
        for term, norm, grams in zip(self.corpus, self._norm, self._grams):
            shared = q_grams & grams
            if not shared:
                continue
            ngram_score = sum(idf(g) for g in shared) / max_possible if max_possible > 0 else 0
            containment = 0.0
            if q_norm in norm:
                containment = len(q_norm) / len(norm)
            elif norm in q_norm:
                containment = len(norm) / len(q_norm)
            results.append((term, 0.5 * ngram_score + 0.5 * containment))
        return sorted(results, key=lambda x: x[1], reverse=True)[:top]


def fuzzy_match(target: str, corpus: list[str], top: int = 10, n: int = 2) -> list[tuple[str, float]]:
    """N-gram matching with containment bonus for substring matches."""
    return NgramIndex(corpus, n).search(target, top)


@dataclass
class OntologyTerm:
    id: str
    name: str
    definition: str = ""
    synonyms: list[str] = field(default_factory=list)
    parents: list[str] = field(default_factory=list)


_QUOTED = re.compile(r'^"((?:[^"\\]|\\.)*)"')


def read_obo(path: str | Path, prefix: str = "CL:") -> dict[str, OntologyTerm]:
    """Non-obsolete ``[Term]`` stanzas whose ID starts with ``prefix``."""
    terms: dict[str, OntologyTerm] = {}
    current: OntologyTerm | None = None
    obsolete = False
    in_term = False

    def flush():
        if in_term and current and current.id.startswith(prefix) and current.name and not obsolete:
            terms[current.id] = current

    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if line.startswith("["):
                flush()
                in_term = line.strip() == "[Term]"
                current, obsolete = OntologyTerm(id="", name=""), False
                continue
            if not in_term or ": " not in line:
                continue
            tag, value = line.split(": ", 1)
            if tag == "id":
                current.id = value.strip()
            elif tag == "name":
                current.name = value.strip()
            elif tag == "def" and (m := _QUOTED.match(value)):
                current.definition = m.group(1).replace('\\"', '"')
            elif tag == "synonym" and (m := _QUOTED.match(value)):
                current.synonyms.append(m.group(1).replace('\\"', '"'))
            elif tag == "is_a":
                current.parents.append(value.split()[0])
            elif tag == "is_obsolete" and value.strip() == "true":
                obsolete = True
    flush()
    return terms


def load_ontology(ontology_file: str | Path) -> list[str]:
    """Sorted unique names of the Cell Ontology terms in an OBO file."""
    return sorted({t.name for t in read_obo(ontology_file).values()})


class OntologySearch:
    """Rank ontology terms against a free-text label, by name and synonyms."""

    def __init__(self, terms: dict[str, OntologyTerm]):
        self.terms = terms
        self._owners: dict[str, list[tuple[str, str]]] = {}
        for term in terms.values():
            self._owners.setdefault(term.name, []).append((term.id, "name"))
            for syn in term.synonyms:
                self._owners.setdefault(syn, []).append((term.id, "synonym"))
        self._index = NgramIndex(list(self._owners))

    def search(self, query: str, top: int = 10) -> list[dict]:
        """Best ``top`` terms, each once, with the string that matched."""
        q = normalize(query)
        exact = [s for s in self._owners if normalize(s) == q]
        scored = [(s, 1.0) for s in exact] + self._index.search(query, top=top * 5)
        seen, out = set(), []
        for text, score in scored:
            for term_id, kind in self._owners[text]:
                if term_id in seen:
                    continue
                seen.add(term_id)
                out.append({"id": term_id, "name": self.terms[term_id].name,
                            "matched": text, "via": kind, "score": round(score, 3)})
        return out[:top]
