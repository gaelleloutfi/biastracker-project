"""Loader for the BiasTracker contaminants annotation database.

The contaminants workbook is a *wide membership matrix*: one row per UniProt
accession and one column per contaminant / abundance category (c-RAP, MQ, CCP,
keratin variants, top-1000 proteome, ...), with ``"+"`` marking membership.
This module melts it into a long-format :class:`AnnotationSet` — one
``primary_id`` ↔ ``term_name`` pair per row — so it can be used directly for
ORA and fgsea. Accessions are UniProt, matching BiasTracker ``primary_id``.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

from biastracker.dataset import AnnotationSet

ID_COL = "UNIPROT_ACCESSION"

# Membership column -> category grouping. Also defines the default set of
# columns that are melted into terms (metadata columns are ignored).
MEMBERSHIP_CATEGORY = {
    "CCP": "contaminant_db",
    "c-RAP": "contaminant_db",
    "MQ": "contaminant_db",
    "all keratin": "keratin",
    "human keratin": "keratin",
    "mouse keratin": "keratin",
    "rat keratin": "keratin",
    "sheep keratin": "keratin",
    "all keratinization": "keratinization",
    "human keratinization": "keratinization",
    "mouse keratinization": "keratinization",
    "rat keratinization": "keratinization",
    "top 1000 Human": "abundant_proteome",
    "top 1000 Mus": "abundant_proteome",
}

OUTPUT_COLUMNS = ["primary_id", "term_id", "term_name", "source", "category", "symbol", "organism"]


def read_contaminants(path: str | Path) -> pd.DataFrame:
    """Read the contaminants database from an Excel workbook or a CSV/TSV file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path, sheet_name=0)
    else:
        sep = "\t" if path.suffix.lower() in {".tsv", ".txt", ".tab"} else ","
        df = pd.read_csv(path, sep=sep)
    if ID_COL not in df.columns:
        raise ValueError(f"Contaminants file is missing the '{ID_COL}' column")
    return df


def _term_id(term: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", term.lower()).strip("_")
    return f"CONTAM:{slug}"


def _clean(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def normalize_contaminants(
    df: pd.DataFrame,
    membership_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Melt the wide membership matrix into long annotation rows.

    Each selected membership column becomes a term; a protein is annotated with
    that term when its cell holds ``"+"``.
    """
    cols = membership_cols or [c for c in MEMBERSHIP_CATEGORY if c in df.columns]
    if not cols:
        raise ValueError("No known membership columns found in contaminants file")

    has_symbol = "SYMBOL" in df.columns
    has_org = "Organism" in df.columns

    rows: list[dict[str, str]] = []
    for col in cols:
        mask = df[col].astype(str).str.strip() == "+"
        sub = df.loc[mask]
        category = MEMBERSHIP_CATEGORY.get(col, "contaminant")
        term_id = _term_id(col)
        for idx, acc in sub[ID_COL].items():
            acc = _clean(acc)
            if not acc:
                continue
            rows.append({
                "primary_id": acc,
                "term_id": term_id,
                "term_name": col,
                "source": "contaminants",
                "category": category,
                "symbol": _clean(df.at[idx, "SYMBOL"]) if has_symbol else "",
                "organism": _clean(df.at[idx, "Organism"]) if has_org else "",
            })

    out = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    if out.empty:
        return out
    return out.drop_duplicates(subset=["primary_id", "term_name"]).reset_index(drop=True)


def load_contaminants(
    path: str | Path,
    name: str = "contaminants",
    membership_cols: list[str] | None = None,
) -> AnnotationSet:
    """Load the contaminants database (xlsx/csv) into an :class:`AnnotationSet`."""
    normalized = normalize_contaminants(read_contaminants(path), membership_cols=membership_cols)
    return AnnotationSet(
        name=name,
        source="contaminants",
        table=normalized,
        id_col="primary_id",
        term_col="term_name",
        term_id_col="term_id",
        category_col="category",
        metadata={
            "n_terms": normalized["term_name"].nunique() if not normalized.empty else 0,
            "n_ids": normalized["primary_id"].nunique() if not normalized.empty else 0,
            "input_path": str(Path(path)),
        },
    )
