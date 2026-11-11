# Cell-type catalog (optional)

A catalog lists the cell types expected in the tissue, with candidate markers.
It suggests candidates; it does not limit them and is not evidence. Include
rare populations as well as common ones.

```bash
lambdakit catalog set --run RUN --json '[{"cell_type":"B cell","prior":"high","lineage":"lymphoid","markers":["MS4A1","CD79A"]}]'
```
