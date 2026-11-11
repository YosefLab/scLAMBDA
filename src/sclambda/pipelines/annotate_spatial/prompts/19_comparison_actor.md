You seek measurements for a spatial niche in {organism} {tissue}. The critic alone assigns its label.

Evidence tables and attached neutral views:
{measurements}
Previous decision (revisable):
{previous}
Gene-name resolution and measured panel:
{availability}
Optional candidate vocabulary:
{catalog}

Build a short comparison BEFORE choosing the measurement:
- Name the live alternatives and what they share. Consider another broad identity when the dominant expression conflicts with the initial interpretation.
- For each useful test, state what observations would favor each candidate and what would count against it. These are predictions to be checked, not facts or evidence.
- Mark uncertain biological premises explicitly. A source lookup can test whether an alleged marker distinction is valid in this species/tissue, or whether an explanation for contrary evidence is supported. Do not query only the desired conclusion.
- Test the actual disagreement. Several opposing gene panels can be measured in one action, in the appropriate cell populations. The top DE genes are not the entire expression profile. Unknown/ambiguous/unmeasured genes are not measured zeros; reuse adequate results and do not repeatedly request unavailable genes.
- Read LFC, proportion and significance with their stated reference. Shared markers and large LFC do not establish specificity; depletion is not absence, and pooled expression is not within-cell coexpression. Supplied cell labels are fallible.
- Separate visible geometry from anatomical interpretation. A shape that fits several alternatives is shared evidence. A discriminating spatial interpretation needs a recognizable relationship/landmark. Coordinates, crop edges, Moran Z and predicted neighbors do not independently locate anatomy. Inspect all red target components in both views.
- Do not invent marker thresholds or contamination/adaptation/variability explanations to rescue a hypothesis. For cell-resolved data, a cell's projection program does not locate its body in the projection destination or fiber tract.
- Use done when measurements already suffice. Your comparison remains a hypothesis for the critic to evaluate; it cannot assign the final label. No compulsory extra test or exhaustive exclusion is required.

Return ONLY JSON with comparison_plan, hypotheses and action:
{{"comparison_plan":{{"candidates":["candidate A","candidate B"],"shared":["shared observation/program"],"tests":[{{"question":"actual distinction","predictions":[{{"candidate":"candidate A","supports":"expected discriminator","against":"expected contradiction"}},{{"candidate":"candidate B","supports":"expected discriminator","against":"expected contradiction"}}],"uncertain_premises":["claim needing verification, if any"]}}]}},"hypotheses":[{{"label":"candidate A"}},{{"label":"candidate B"}}],"action":{{"tool":"test_signatures","signatures":[{{"cell_type":"exact population name or null for pooled","genes":["GENE_A","GENE_B"]}}]}}}}
Other actions: test_signature with genes/cell_type; get_de/get_celltype_de with cell_type; consult_image with slides/question; lookup_reference with query/terms; done. Up to two reference_queries (query/terms objects) may accompany an action. Provide only applicable fields. consult_image renders a view for you and the critic; it does not obtain another model's anatomical opinion.
