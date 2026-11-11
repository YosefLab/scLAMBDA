Extract the updated cell type hypotheses and confidence decision from the reevaluation below.

Return your response as a JSON object with the following structure:
```json
{{
  "confidence_decision": "confident" | "needs_refinement",
  "hypotheses": [
    {{
      "cell_type": "cell type name",
      "confidence": "high" | "medium" | "low",
      "justification": "A single concise paragraph (2-3 sentences, max 80 words). Sentence 1: 'This cluster expresses [gene1 (LFC=+X.XX, Frac=0.XX), gene2 (LFC=+Y.YY, Frac=0.YY), ...], which are characteristic markers of [cell type], supporting its identification as [cell type].' — comprehensively list all supporting genes with LFC and expression rate (Frac) when available. Sentence 2: 'However, [ambiguity/limitation]...' — flag all shared markers, missing expected markers, and competing interpretations."
    }}
  ]
}}
```

**Important:**
- `confidence_decision` must be either "confident" or "needs_refinement"
  - Use "confident" if the reevaluation has converged on a clear top cell type with high or medium confidence and additional marker testing is unlikely to change the conclusion
  - Use "needs_refinement" if the top hypotheses remain ambiguous, competing, or if the reevaluation explicitly suggests additional markers should be tested
- The "justification" field is REQUIRED and must always be provided for each hypothesis
- The justification MUST follow the two-sentence structure: first sentence lists all supporting evidence with LFC and expression rate (Frac) values, second sentence lists all ambiguities
- The first sentence must start with: "This cluster expresses..."
- The second sentence must start with: "However, ..."
- Update justifications to incorporate new evidence from the signature DE results
- Include both updated original hypotheses and any new cell type candidates
- Order by confidence (most confident first)
- If a hypothesis was ruled out, you may omit it or include it with confidence "low"
- Reformulate justification to include only objective facts
- Return ONLY the JSON object, no other text

**REEVALUATION RESPONSE:**

{reevaluation_response}
