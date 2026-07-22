"""UniProt functional (Gene Ontology) annotation.

Fetches GO term annotations for a set of UniProt accessions directly from the
UniProt REST API and returns them as a long-format
:class:`~biastracker.dataset.AnnotationSet` (one ``primary_id`` ↔ ``term_name``
pair per row) that can be fed straight into ``run_ora`` / ``run_fgsea``.

This complements :mod:`biastracker.annotations.panther` (PANTHER/GO via the
PANTHER service) and :mod:`biastracker.annotations.uniprot_mapping` (ID mapping
only — that module does *not* fetch functional terms).

The network layer (:func:`fetch_uniprot_go_tsv`) is kept separate from the pure
parser (:func:`parse_uniprot_go_tsv`) so the parsing logic is unit-testable
without any HTTP calls.
"""
from __future__ import annotations

import re
from io import StringIO
from pathlib import Path
from typing import Iterable

import pandas as pd

from biastracker.annotations._http import (
    DEFAULT_ANNOTATION_TTL_DAYS,
    cache_is_fresh,
    ensure_cache_dir,
    request_with_retries,
    safe_cache_key,
)
from biastracker.dataset import AnnotationSet

UNIPROT_ACCESSIONS_URL = "https://rest.uniprot.org/uniprotkb/accessions"

# GO aspect code -> (UniProt `fields` name, TSV header substring, category label).
GO_ASPECTS: dict[str, tuple[str, str, str]] = {
    "P": ("go_p", "biological process", "biological_process"),
    "C": ("go_c", "cellular component", "cellular_component"),
    "F": ("go_f", "molecular function", "molecular_function"),
}
DEFAULT_ASPECTS: tuple[str, ...] = ("P", "C", "F")

OUTPUT_COLUMNS = ["primary_id", "term_id", "term_name", "source", "category"]

# Matches "apoptotic process [GO:0006915]" -> ("apoptotic process", "GO:0006915").
_TERM_RE = re.compile(r"^(?P<name>.*?)\s*\[(?P<id>GO:\d+)\]$")


def _unique_nonempty(ids: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in ids:
        text = str(value).strip()
        if not text or text.lower() in {"nan", "none"} or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _batched(items: list[str], batch_size: int) -> list[list[str]]:
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def _aspect_fields(aspects: Iterable[str]) -> list[str]:
    fields = []
    for code in aspects:
        code = code.upper()
        if code not in GO_ASPECTS:
            raise ValueError(f"Unknown GO aspect '{code}'. Valid: {sorted(GO_ASPECTS)}")
        fields.append(GO_ASPECTS[code][0])
    if not fields:
        raise ValueError("At least one GO aspect must be requested.")
    return fields


def fetch_uniprot_go_tsv(
    accessions: list[str],
    aspects: Iterable[str] = DEFAULT_ASPECTS,
    session=None,
    timeout: float = 90,
) -> str:
    """Fetch a TSV of GO annotations for *accessions* from the UniProt API.

    Uses the ``/uniprotkb/accessions`` endpoint (accession lookup). Batch the
    accessions yourself — this issues a single request. Returns the raw TSV text.
    """
    clean = _unique_nonempty(accessions)
    if not clean:
        raise ValueError("No accessions provided for UniProt GO fetch")
    fields = ["accession", *_aspect_fields(aspects)]
    response = request_with_retries(
        "GET",
        UNIPROT_ACCESSIONS_URL,
        session=session,
        timeout=timeout,
        params={
            "accessions": ",".join(clean),
            "fields": ",".join(fields),
            "format": "tsv",
        },
    )
    return response.text


def parse_uniprot_go_tsv(tsv_text: str) -> pd.DataFrame:
    """Parse UniProt GO TSV into long-format annotation rows.

    Expects an ``Entry`` column plus one or more ``Gene Ontology (...)`` columns
    whose cells hold ``"name [GO:id]; name [GO:id]"`` lists. Returns a DataFrame
    with :data:`OUTPUT_COLUMNS`; empty (but correctly typed) when there are no
    GO terms. Malformed / empty cells are skipped.
    """
    if not tsv_text or not tsv_text.strip():
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    df = pd.read_csv(StringIO(tsv_text), sep="\t", dtype=str, keep_default_na=False)
    if "Entry" not in df.columns:
        raise ValueError(
            f"UniProt GO TSV is missing the 'Entry' column; got {list(df.columns)}"
        )

    # Map each GO aspect column present in the header to its category label.
    aspect_cols: dict[str, str] = {}
    for col in df.columns:
        low = col.lower()
        if "gene ontology" not in low:
            continue
        for _code, (_field, header_sub, category) in GO_ASPECTS.items():
            if header_sub in low:
                aspect_cols[col] = category
                break

    rows: list[dict[str, str]] = []
    for record_d in df.to_dict(orient="records"):
        acc = str(record_d.get("Entry", "")).strip()
        if not acc:
            continue
        for col, category in aspect_cols.items():
            cell = str(record_d.get(col, "")).strip()
            if not cell:
                continue
            for token in cell.split(";"):
                token = token.strip()
                if not token:
                    continue
                match = _TERM_RE.match(token)
                if match:
                    term_name = match.group("name").strip()
                    term_id = match.group("id")
                else:
                    # Fall back to the raw token as the term name (no GO id).
                    term_name = token
                    term_id = token
                if not term_name:
                    continue
                rows.append({
                    "primary_id": acc,
                    "term_id": term_id,
                    "term_name": term_name,
                    "source": "UniProt",
                    "category": category,
                })

    out = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    if out.empty:
        return out
    return out.drop_duplicates().reset_index(drop=True)


def load_uniprot_go_annotations(
    accessions: list[str],
    name: str = "uniprot_go",
    aspects: Iterable[str] = DEFAULT_ASPECTS,
    batch_size: int = 500,
    cache_dir: str | Path = ".cache/biastracker/uniprot_go",
    use_cache: bool = True,
    max_age_days: float | None = DEFAULT_ANNOTATION_TTL_DAYS,
    session=None,
) -> AnnotationSet:
    """Fetch UniProt GO annotations for *accessions* and return an AnnotationSet.

    Accessions are de-duplicated and fetched in batches. Parsed batches are
    cached on disk (keyed by accession batch + aspects) so repeat runs are cheap.
    Cached batches older than *max_age_days* are re-fetched (pass ``0`` to force a
    refresh, ``None`` to never expire). Fresh results are always written back so
    the cache stays current.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")
    aspects = tuple(a.upper() for a in aspects)
    _aspect_fields(aspects)  # validate early

    clean = _unique_nonempty(accessions)
    if not clean:
        raise ValueError("No accessions provided for UniProt GO annotation")

    cache_path = ensure_cache_dir(cache_dir) if use_cache else None
    frames: list[pd.DataFrame] = []

    for batch in _batched(clean, batch_size):
        parsed: pd.DataFrame | None = None
        cache_file = None
        if cache_path is not None:
            key = safe_cache_key(",".join(sorted(batch)) + "|" + ",".join(aspects))
            cache_file = cache_path / f"go_{key}.csv"
            if cache_is_fresh(cache_file, max_age_days):
                parsed = pd.read_csv(cache_file, dtype=str, keep_default_na=False)

        if parsed is None:
            tsv = fetch_uniprot_go_tsv(batch, aspects=aspects, session=session)
            parsed = parse_uniprot_go_tsv(tsv)
            if cache_file is not None:
                parsed.to_csv(cache_file, index=False)

        frames.append(parsed)

    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=OUTPUT_COLUMNS)
    if not out.empty:
        out = out.drop_duplicates().reset_index(drop=True)

    return AnnotationSet(
        name=name,
        source="UniProt",
        table=out,
        id_col="primary_id",
        term_col="term_name",
        term_id_col="term_id",
        category_col="category",
        metadata={
            "aspects": list(aspects),
            "n_input_ids": len(clean),
            "n_annotation_rows": len(out),
            "source": "UniProt_GO_API",
        },
    )
