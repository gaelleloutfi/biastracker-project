import os
from typing import Optional

import pandas as pd

from biastracker.dataset import AnnotationSet
from biastracker.annotations.custom import _read_delimited


def load_hpa_subcellular(
    path: str | os.PathLike,
    name: str = "hpa_subcellular",
    id_col: Optional[str] = None,
    location_col: Optional[str] = None,
) -> AnnotationSet:
    """Load an HPA subcellular localization table into an AnnotationSet.
    
    Parameters
    ----------
    path:
        Path to the CSV or TSV file.
    name:
        Human-readable name for the resulting AnnotationSet.
    id_col:
        Column containing protein IDs. If None, it will be inferred.
    location_col:
        Column containing subcellular locations. If None, it will be inferred.
    """
    df = _read_delimited(path)

    # Infer id_col
    if not id_col:
        for col in ["primary_id", "uniprot", "uniprot_id", "Uniprot", "Gene", "Ensembl", "Protein"]:
            if col in df.columns:
                id_col = col
                break

    # Infer location_col
    if not location_col:
        for col in ["location", "main_location", "subcellular_location", "Subcellular location", "Main location"]:
            if col in df.columns:
                location_col = col
                break

    # Validate
    missing = []
    if not id_col or id_col not in df.columns:
        missing.append("id_col")
    if not location_col or location_col not in df.columns:
        missing.append("location_col")

    if missing:
        raise ValueError(f"Could not infer or find required columns for HPA file: {', '.join(missing)}")

    # We only need id and location
    df = df[[id_col, location_col]].copy()
    
    # Split multiple locations separated by ';' or ','
    df[location_col] = df[location_col].astype(str)
    
    # Use pandas str.split and explode
    # First replace ';' with ',' to unify separators, then split by ','
    df[location_col] = df[location_col].str.replace(';', ',')
    df = df.assign(**{location_col: df[location_col].str.split(',')})
    df = df.explode(location_col)
    
    # Clean up whitespace
    df[location_col] = df[location_col].str.strip()
    
    out = pd.DataFrame()
    out["primary_id"] = df[id_col].astype(str).str.strip()
    out["term_name"] = df[location_col]
    out["term_id"] = "HPA:" + out["term_name"].str.replace(' ', '_')
    out["source"] = "HPA"
    out["category"] = "subcellular_location"

    _nan_mask = (
        out["primary_id"].str.lower().isin({"", "nan", "none"})
        | out["term_name"].str.lower().isin({"", "nan", "none"})
    )
    out = out[~_nan_mask].copy()

    return AnnotationSet(
        name=name,
        source="HPA",
        table=out,
        id_col="primary_id",
        term_col="term_name",
        term_id_col="term_id",
        category_col="category",
    )
