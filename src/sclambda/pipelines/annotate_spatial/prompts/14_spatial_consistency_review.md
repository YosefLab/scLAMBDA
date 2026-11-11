Review the visible section of {organism} {tissue} for concrete spatial contradictions in provisional annotations.
Niche IDs are marked on the map. The map is coloured by neutral niche IDs, not predicted anatomical compartments. Roster labels are hypotheses, not independent evidence. Inspect geometry and adjacency; do not derive certainty from agreement among predictions. Only this slide is visible. Crop edges, folds, disconnected instances and incomplete sections may make simple inside/outside rules inappropriate. No single neighbour or percentage automatically invalidates a compartment.

- Report only a visible spatial relation that challenges the biological label, identifying the relevant niche IDs or landmarks and the uncertainty of the observation.
- A preference for a narrower name, different hierarchy depth, or more consistent wording is not a spatial contradiction. Broad labels can legitimately contain finer compartments; do not infer a new biological identity from naming conventions.
- Predicted colors or region names cannot independently locate an anatomical structure. Do not infer anatomical orientation from plot axes alone, or treat a missing neighbor in the supplied list as proof of separation.
- If geometry is compatible with the current broad label but cannot distinguish finer alternatives, propose no change. Return [] when there is no concrete visible contradiction.

Roster:
{roster}

Return ONLY a JSON array of proposals for niches with a concrete visible contradiction, or [] if none. These proposals will be reviewed against original composition and gene measurements by the deciding critic. You cannot change labels directly.
[{{"niche":"actual ID","corrected_sub_region":"proposed alternative","corrected_major_region":"proposed broader compartment","reason":"specific visible observation, its uncertainty and why it challenges the label"}}]
