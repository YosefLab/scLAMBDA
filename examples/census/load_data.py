"""Download a primary-data-only Census slice (requires cellxgene-census).

This is the maintained copy of the study's external
sc_llm/exp/whole_census/load_data.py loader. Existing downloaded files are not
changed by this correction and must be filtered or regenerated separately.
"""
import json


def load_census_chunk(ontology_ids, census_version="2025-01-30", organism="Homo sapiens"):
    ontology_ids = list(ontology_ids)
    if not ontology_ids:
        raise ValueError("At least one cell-type ontology ID is required")
    import cellxgene_census

    ct_string = " or ".join(
        f"cell_type_ontology_term_id == {json.dumps(term)}" for term in ontology_ids
    )
    with cellxgene_census.open_soma(census_version=census_version) as census:
        return cellxgene_census.get_anndata(
            census=census,
            organism=organism,
            obs_value_filter=f"({ct_string}) and is_primary_data == True",
            obs_embeddings=["scvi"],
            var_value_filter='feature_type=="protein_coding"',
        )
