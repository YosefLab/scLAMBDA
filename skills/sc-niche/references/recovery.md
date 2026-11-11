# Recovery

A valid `best_effort` label with unperformed checks is a finished annotation,
not a failure. Recovery is for niches where no valid decision was produced.

## During the main loop

- A failed actor turn or tool still goes to the critic. A tool failure is not a
  negative result.
- A rejected decision gets at most one formatting repair, which may change only
  measurement IDs. Scope mismatches, missing `pending_checks` and inconsistent
  comparisons go back to the same critic for a substantive revision in the next
  round. Do not run checks from a rejected decision.
- A complete response with malformed JSON can be retried as a fresh review
  within the budget. Never repair it by closing braces or filling in fields.

## Final review: at most two attempts

If the main loop ends without a valid decision, ask the same critic for a fresh
flat assessment. Give it the tables, images, candidate names, the last accepted
decision and the errors; earlier conclusions and errors are context, not
evidence. No tools run during these attempts.

```json
{
  "label": "best-supported compartment",
  "status": "best_effort",
  "reason": "Positive evidence, strongest rival and contradiction, and why this choice wins",
  "support": [{"measurement_id": "ACTUAL_COMPOSITION_ID", "items": ["actual cell type"]}],
  "counterevidence": [],
  "alternatives": [{"label": "strongest rival", "reason": "its support and the open distinction"}],
  "uncertainty": "Remaining ambiguity and missing checks that matter",
  "pending_checks": [],
  "next_question": "",
  "output_origin": "final_critic"
}
```

References follow the [decision format](decision.md#references). Keep
`pending_checks` for useful tests that were not run. Save with
`dossier decision --file final-decision.json`; a rejected first attempt gets
the errors and one more try. A valid decision ends recovery; commit it.

## If both attempts fail

```bash
lambdakit dossier retain-failure --run RUN --cluster ID --reason "What failed and where the traces are"
lambdakit dossier commit --run RUN --cluster ID
```

This makes no new inference. It keeps the last accepted label as `best_effort`
(`output_origin=previous_critic`), or, if there is none, writes
`Spatial niche (anatomy unassigned)` (`output_origin=failure_record`). Both
export with `execution_status=inference_failed`. Report these separately and
continue with other niches. A later successful review replaces the record;
commit it again before exporting. Do not hand-edit dossiers or results.
