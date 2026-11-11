Extract the cell type hypotheses from the expert debate below.

For each cell type mentioned, extract:
1. The cell type name
2. The confidence level
3. The justification paragraph

Return your response as a JSON array with the following structure:
```json
[
  {{
    "cell_type": "cell type name",
    "confidence": "high" | "medium" | "low",
    "justification": "A single concise paragraph (2-3 sentences, max 80 words) structured as: Sentence 1: 'This cluster expresses [gene1 (LFC=+X.XX, Frac=0.XX), gene2 (LFC=+Y.YY, Frac=0.YY), ...], which are characteristic markers of [cell type], supporting its identification as [cell type].' — comprehensively list all key supporting genes with their LFC and expression rate (Frac) when available. Sentence 2: 'However, [ambiguity] — [gene X] is also expressed in [alt type], and [expected marker Y] is absent, leaving some uncertainty between [type A] and [type B].' — flag all shared markers, missing expected markers, and competing interpretations."
  }}
]
```

**Important:**
- Do not use brackets in cell type names
- The "justification" field is REQUIRED and must always be provided for each hypothesis
- The justification MUST follow the two-sentence structure: first sentence lists all supporting evidence with LFCs and Frac, second sentence lists all ambiguities
- The first sentence must start with: "This cluster expresses..."
- The second sentence must start with: "However, ..."
- Include concrete gene names with LFC and expression rate (Frac) values from the debate — no generic filler
- Order by confidence (most confident first)
- If no valid cell types are found, return an empty array []
- Return ONLY the JSON array, no other text

**EXPERT DEBATE:**

{debate_response}
