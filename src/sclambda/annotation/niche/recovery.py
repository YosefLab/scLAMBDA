"""Exportable spatial outputs after the bounded critic review has failed."""

from copy import deepcopy


def retain_failed_output(dossier):
    """Retain an accepted decision or an explicit failure, never an actor guess.

    This is execution recovery, not a new biological decision. Callers record
    the failures and own the review budget before invoking it.
    """
    if dossier.critic_decision.get("output_origin") in {"previous_critic", "failure_record"}:
        return  # Repeated recovery must not relabel a failure as an accepted decision.
    note = "Final review failed; this output is not a newly validated anatomical decision."
    if dossier.critic_decision:
        value = deepcopy(dossier.critic_decision)
        value.update(status="best_effort", output_origin="previous_critic")
        value["uncertainty"] = (value.get("uncertainty", "") + " " + note
            + " The last accepted label is retained and may not incorporate later evidence.").strip()
    else:
        value = dict(label="Spatial niche (anatomy unassigned)", status="best_effort",
            reason="No critic decision passed validation; measurements and failed reviews remain in the trace.",
            uncertainty=note, alternatives=[], support=[], counterevidence=[],
            pending_checks=[], next_question="Anatomical review required", output_origin="failure_record")
    dossier.critic_decision = value
