"""Loaders for custom annotation files (long-format tables and GMT files)."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pandas as pd

from biastracker.dataset import AnnotationSet

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_CANONICAL_COLS = ["primary_id", "term_id", "term_name", "source", "category"]


def _read_delimited(path: str | os.PathLike) -> pd.DataFrame:
    """Read a CSV or TSV file, auto-detecting the separator.

    Detection order:
    1. ``.tsv`` / ``.tab`` extension  → TAB
    2. Otherwise try comma; if the result has only one column, retry with TAB.
    """
    path = Path(path)
    ext = path.suffix.lower()

    if ext in {".tsv", ".tab", ".txt"}:
        return pd.read_csv(path, sep="\t", dtype=str)

    # Try comma first
    df = pd.read_csv(path, sep=",", dtype=str)
    if df.shape[1] == 1:
        # Probably tab-separated with a .csv extension – retry
        df = pd.read_csv(path, sep="\t", dtype=str)
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_long_annotation_table(
    path: str | os.PathLike,
    name: str,
    source: str = "custom",
    id_col: str = "primary_id",
    term_col: str = "term_name",
    term_id_col: Optional[str] = None,
    category_col: Optional[str] = None,
) -> AnnotationSet:
    """Load a long-format annotation table (CSV or TSV) into an :class:`AnnotationSet`.

    Each row maps one identifier to one annotation term.

    Parameters
    ----------
    path:
        Path to the CSV or TSV file.  The separator is detected automatically.
    name:
        Human-readable name for the resulting :class:`AnnotationSet`.
    source:
        Provenance tag written to the ``source`` column (default ``"custom"``).
    id_col:
        Column in the file that contains the entity identifiers.
    term_col:
        Column in the file that contains the annotation term names.
    term_id_col:
        Column containing term IDs.  When ``None`` or absent from the file,
        ``term_id`` is set equal to ``term_name``.
    category_col:
        Column containing category labels.  When ``None`` or absent, ``category``
        defaults to ``"unknown"``.

    Returns
    -------
    AnnotationSet
    """
    df = _read_delimited(path)

    # Validate required columns
    missing = [c for c in (id_col, term_col) if c not in df.columns]
    if missing:
        raise ValueError(
            f"Required column(s) {missing} not found in '{path}'. "
            f"Available columns: {list(df.columns)}"
        )

    out = pd.DataFrame()
    out["primary_id"] = df[id_col].astype(str)
    out["term_name"] = df[term_col].astype(str)

    # term_id: use provided column if present, else copy term_name
    if term_id_col and term_id_col in df.columns:
        out["term_id"] = df[term_id_col].astype(str)
    else:
        out["term_id"] = out["term_name"]

    # category: use provided column if present, else "unknown"
    if category_col and category_col in df.columns:
        out["category"] = df[category_col].astype(str)
    else:
        out["category"] = "unknown"

    out["source"] = source

    # Drop rows with missing primary_id or term_name (literal "nan" strings too)
    _nan_mask = (
        out["primary_id"].str.strip().str.lower().isin({"", "nan", "none"})
        | out["term_name"].str.strip().str.lower().isin({"", "nan", "none"})
    )
    out = out[~_nan_mask].copy()

    return AnnotationSet(
        name=name,
        source=source,
        table=out,
        id_col="primary_id",
        term_col="term_name",
        term_id_col="term_id",
        category_col="category",
    )


def load_gmt(
    path: str | os.PathLike,
    name: str,
    source: str = "custom",
    category: str = "unknown",
) -> AnnotationSet:
    """Load a GMT (Gene Matrix Transposed) file into an :class:`AnnotationSet`.

    Each line of a GMT file has the format::

        term_name <TAB> description <TAB> id1 <TAB> id2 <TAB> ...

    The description field is stored in the returned :class:`AnnotationSet`'s
    metadata under the key ``"descriptions"``.

    Parameters
    ----------
    path:
        Path to the ``.gmt`` file.
    name:
        Human-readable name for the resulting :class:`AnnotationSet`.
    source:
        Provenance tag written to the ``source`` column (default ``"custom"``).
    category:
        Category label applied to all terms (default ``"unknown"``).

    Returns
    -------
    AnnotationSet
    """
    path = Path(path)
    rows: list[dict] = []
    descriptions: dict[str, str] = {}

    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue  # malformed line — skip

            term_name = parts[0].strip()
            description = parts[1].strip() if len(parts) > 1 else ""
            ids = [p.strip() for p in parts[2:] if p.strip()]

            if term_name:
                descriptions[term_name] = description

            for pid in ids:
                if pid:
                    rows.append({
                        "primary_id": pid,
                        "term_id": term_name,
                        "term_name": term_name,
                        "source": source,
                        "category": category,
                    })

    if not rows:
        df = pd.DataFrame(columns=_CANONICAL_COLS)
    else:
        df = pd.DataFrame(rows)

    return AnnotationSet(
        name=name,
        source=source,
        table=df,
        id_col="primary_id",
        term_col="term_name",
        term_id_col="term_id",
        category_col="category",
        metadata={"descriptions": descriptions},
    )
