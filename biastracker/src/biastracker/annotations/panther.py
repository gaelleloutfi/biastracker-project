import os
from pathlib import Path
from typing import Optional

import pandas as pd

from biastracker.dataset import AnnotationSet
from biastracker.annotations.custom import _read_delimited


def load_panther_annotation(
    path: str | os.PathLike,
    name: str,
    panther_type: str,
    id_col: Optional[str] = None,
    term_id_col: Optional[str] = None,
    term_name_col: Optional[str] = None,
    category: Optional[str] = None,
) -> AnnotationSet:
    """Load a PANTHER annotation table into an AnnotationSet.
    
    Parameters
    ----------
    path:
        Path to the CSV or TSV file.
    name:
        Human-readable name for the resulting AnnotationSet.
    panther_type:
        Type of PANTHER annotation (e.g., 'family', 'pathway', 'go_slim').
        This is used to automatically infer the category.
    id_col:
        Column containing protein IDs. If None, it will be inferred.
    term_id_col:
        Column containing term IDs. If None, it will be inferred.
    term_name_col:
        Column containing term names. If None, it will be inferred.
    category:
        Explicit category override. If None, inferred from panther_type.
    """
    df = _read_delimited(path)

    # Infer id_col
    if not id_col:
        for col in ["primary_id", "uniprot", "uniprot_id", "accession", "protein_id", "mapped_id"]:
            if col in df.columns:
                id_col = col
                break

    # Infer term_id_col
    if not term_id_col:
        for col in ["term_id", "go_id", "pathway_id", "family_id", "panther_id", "id"]:
            if col in df.columns:
                term_id_col = col
                break

    # Infer term_name_col
    if not term_name_col:
        for col in ["term_name", "name", "label", "description", "go_term", "pathway_name", "family_name"]:
            if col in df.columns:
                term_name_col = col
                break

    # Validate
    missing = []
    if not id_col or id_col not in df.columns:
        missing.append("id_col")
    if not term_id_col or term_id_col not in df.columns:
        missing.append("term_id_col")
    if not term_name_col or term_name_col not in df.columns:
        missing.append("term_name_col")

    if missing:
        raise ValueError(f"Could not infer or find required columns for PANTHER file: {', '.join(missing)}")

    # Default category mapping
    if category is None:
        type_lower = panther_type.lower()
        if type_lower == "family":
            category = "family"
        elif type_lower == "subfamily":
            category = "subfamily"
        elif type_lower in ("pathway", "reactome"):
            category = "pathway"
        elif type_lower in ("go_slim", "gene_ontology"):
            category = "ontology"
        elif type_lower == "protein_class":
            category = "protein_class"
        else:
            category = "unknown"

    out = pd.DataFrame()
    out["primary_id"] = df[id_col].astype(str)
    out["term_id"] = df[term_id_col].astype(str)
    out["term_name"] = df[term_name_col].astype(str)
    out["source"] = "PANTHER"
    out["category"] = category

    _nan_mask = (
        out["primary_id"].str.strip().str.lower().isin({"", "nan", "none"})
        | out["term_name"].str.strip().str.lower().isin({"", "nan", "none"})
    )
    out = out[~_nan_mask].copy()

    return AnnotationSet(
        name=name,
        source="PANTHER",
        table=out,
        id_col="primary_id",
        term_col="term_name",
        term_id_col="term_id",
        category_col="category",
    )
