# Decision format

The comparison is the decision. The CLI derives the label, reason, support,
counterevidence and alternatives from it, and renders expression values from
the cited tables, so do not copy numbers into observations.

```json
{
  "comparison": {
    "candidates": ["Candidate A", "Candidate B"],
    "broader_label": "Identity both candidates share",
    "observations": [
      {
        "id": "o1",
        "refs": [{"measurement_id": "ACTUAL_ID", "genes": ["ACTUAL_SYMBOL"],
                  "cell_type": "exact population", "reference": "copied from the table"}],
        "observation": "The measured or visible pattern, without numbers",
        "interpretation": {
          "expectation": "Why this is more expected under A than B",
          "applicability": "Why that premise applies to this population, reference and assay",
          "limitation": "What this cannot distinguish"
        },
        "effect": "favors", "favors": ["Candidate A"], "assumptions": []
      }
    ],
    "assumptions": [],
    "selection": {
      "label": "Candidate A", "basis": ["o1"], "tradeoffs": {},
      "review": {"strongest_alternative": "Candidate B",
                 "decisive_observations": ["o1"], "remaining_issue": null}
    }
  },
  "status": "best_effort",
  "uncertainty": "Remaining limitations",
  "pending_checks": [],
  "next_question": ""
}
```

## Observations

- `candidates` are distinct live options. `broader_label` is what they share,
  without the disputed refinement.
- `effect` is `shared`, `favors`, `uninformative` or `conflicting`. Only
  `favors` has a non-empty `favors` list, and it cannot list every candidate
  (that is `shared`). `conflicting` means the observation points both ways.
- A `favors` observation needs all three interpretation fields, written before
  deciding the effect. The others can use a short text interpretation.
- A compatible pattern is not necessarily a discriminator. Indirect evidence
  counts; there is no required number of markers or exclusive marker.
- Keep the strongest contrary evidence; an exhaustive list is not needed.
- On later reviews keep each observation's `id`. If its effect or favoured
  candidates change, add a `revision` saying why.

## Selection

- `label` is a candidate or the `broader_label`. `basis` lists the supporting
  observation IDs; a specific candidate needs at least one observation that
  favours it. Shared observations support only the broader label.
- `tradeoffs` maps every observation that favours only other candidates to the
  basis IDs you judged to outweigh it, e.g. `{"o2": ["o1"]}`. Outweighing a
  rival's observation needs distinguishing support, not shared context.
  Tradeoffs can also address `conflicting` observations. Put any new premise in
  the observations first; tradeoffs carry no narrative.
- `review.strongest_alternative` is another live candidate.
  `decisive_observations` come from `basis` and favour the winner but not that
  alternative. For a broader label, use an empty list and a finer candidate as
  the alternative.
- `remaining_issue` is `null` or an uncertainty that could reverse the choice;
  the latter forces `best_effort`. Limitations that could not reverse it go in
  `uncertainty`.
- `major_region` is optional: a broader compartment name at the top level of
  the decision. Leave the key out rather than setting it to `null`.

## Assumptions

An uncertain premise used to dismiss counterevidence or support a key
distinction is an assumption:

```json
{"id": "a1", "claim": "Proposed explanation", "evidence": [],
 "check": {"question": "Can this be verified?",
           "action": {"tool": "lookup_reference", "query": "focused question", "terms": ["term"]}}}
```

Observations list the assumption IDs they depend on. With empty `evidence` the
assumption is unverified, so an observation depending on it cannot be necessary
support in `basis`; its `check` is added to `pending_checks` automatically.
`check` can be `null` when no check is possible.

## Pending checks

Each check is `{"question": "...", "action": {...}}`. Actions:
`test_signatures` (`signatures`: list of `{cell_type, genes}`),
`test_signature` (`genes`, `cell_type`), `get_de` / `get_celltype_de`
(`cell_type`), `lookup_reference` (`query`, `terms`), `consult_image`
(`slides`, `question`). There is no `done` action. Non-empty checks set
`next_question` and force `best_effort`. Saving the decision does not run them;
you do.

## References

- Expression: `measurement_id`, `genes`, `cell_type` (`null` for pooled) and
  `reference`, all matching the cited table.
- Composition: `items` with cell-type names. Gene spatial scores: `items` with
  gene names. Image observations: `items: ["observation"]`. Literature: `items`
  with exact passage IDs.
- Use real IDs. References are checked for every observation and assumption,
  not only the selected ones.

## When a decision is rejected

A formatting repair may change only measurement IDs. A scope mismatch or an
inconsistent comparison is a biological error: give the same critic the
rejected decision, the errors and the evidence in the next round. Do not change
a population field just to pass validation. See [recovery](recovery.md).
