# Deciding whether to subcluster

```bash
lambdakit substructure --run RUN --cluster ID
```

The probe looks for regions of the cluster that lack its identity signature and
for structured gene modules. Read its measurements before deciding.

- A structured, biologically distinct program can justify a split. Activation,
  donor, clonal or technical effects can also produce structure; the probe
  does not tell you which.
- Uncertainty about the label is not evidence of a mixture, and a detected
  pocket is not by itself a different cell type.
- If the probe finds nothing, keep the cluster as a leaf. If it fails, report
  the failure rather than calling the cluster homogeneous.

To split, record the reason and set the flag:

```bash
lambdakit dossier evidence --run RUN --cluster ID --source substructure --fact "Structure found and why it justifies a split"
lambdakit dossier axes --run RUN --cluster ID --json '{"needs_subcluster":true}'
lambdakit dossier commit --run RUN --cluster ID
```

The orchestrator splits flagged parents between waves. Each child is judged on
its own evidence; children need not keep the parent's identity or get distinct
labels.
