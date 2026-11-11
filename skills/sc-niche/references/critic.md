# Critic

Choose the best-supported compartment from the tables, gene-resolution records,
images and the actor's plan. The plan's predictions are hypotheses; check which
of them the measurements bore out. Write the decision in the
[decision format](decision.md): the label and explanation are derived from your
comparison.

- Before naming a winner, name the strongest remaining alternative and the
  observation that separates them. A new identity needs its own comparison
  against the strongest broad alternative: excluding a rival does not establish
  it, and excluding a subtype does not exclude its compartment.
- Consider broad identities or mixtures the actor missed.
- Compare the strongest coherent positive evidence with the strongest
  contradiction. One discordant marker need not veto it.
- Match the label's specificity to what was compared. Broad-compartment evidence
  does not establish an untested subregion; move unsupported qualifiers to
  `uncertainty`, also for `best_effort` labels. Adequate fine evidence still
  earns a fine label.
- Image observations are fallible. Look at the images when the label depends on
  position.
- A literature source can support a qualitative association; it does not
  calibrate an LFC as a cutoff. An association in a different population does
  not explain a within-population contrast. Ask for a source check when a
  decisive premise is uncertain.
- When you report a negative expression claim, give the proportion.

## Continue or stop

If a test could change the label, request it in `pending_checks`. If your
reasoning mentions an untested gene or uncertain distinction, either request
the check or say why it cannot change the label; uncertainty prose alone
requests nothing. Leave the checks empty only when none would help. In the
final round, keep unperformed checks and `best_effort`; running out of rounds
never raises confidence.

Before stopping, revisit the plan's consequential questions and earlier checks:
say whether each was answered, unavailable or no longer relevant. A tool call
that succeeded has not necessarily answered the biological question.

If you are given rejected-decision errors, see [recovery](recovery.md).
