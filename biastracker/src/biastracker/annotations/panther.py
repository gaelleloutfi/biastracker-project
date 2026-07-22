from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from biastracker.annotations._http import (
    DEFAULT_ANNOTATION_TTL_DAYS,
    cache_is_fresh,
    ensure_cache_dir,
    read_json_cache,
    request_with_retries,
    safe_cache_key,
    write_json_cache,
)
from biastracker.dataset import AnnotationSet
from biastracker.annotations.custom import _read_delimited


PANTHER_BASE_URL = "https://pantherdb.org/services/oai/pantherdb"

PANTHER_DATASETS = {
    "go_bp": "GO:0008150",
    "go_cc": "GO:0005575",
    "go_mf": "GO:0003674",
    "panther_go_slim_bp": "ANNOT_TYPE_ID_PANTHER_GO_SLIM_BP",
    "panther_go_slim_cc": "ANNOT_TYPE_ID_PANTHER_GO_SLIM_CC",
    "panther_go_slim_mf": "ANNOT_TYPE_ID_PANTHER_GO_SLIM_MF",
    "protein_class": "ANNOT_TYPE_ID_PANTHER_PC",
    "panther_pathway": "ANNOT_TYPE_ID_PANTHER_PATHWAY",
    "reactome_pathway": "ANNOT_TYPE_ID_REACTOME_PATHWAY",
}

PANTHER_DEFAULT_CATEGORIES = list(PANTHER_DATASETS)
PANTHER_OUTPUT_COLUMNS = [
    "primary_id",
    "term_id",
    "term_name",
    "source",
    "category",
    "panther_accession",
    "panther_dataset_id",
]
PANTHER_OVERREP_COLUMNS = [
    "term_id",
    "term_name",
    "source",
    "category",
    "expected",
    "fold_enrichment",
    "p_value",
    "fdr",
    "direction",
]


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

    Annotation IDs must match the dataset ID namespace. UniProt accessions are
    preferred for BiasTracker protein-level datasets.
    
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
    out["primary_id"] = df[id_col].astype(str).str.strip()
    out["term_id"] = df[term_id_col].astype(str).str.strip()
    out["term_name"] = df[term_name_col].astype(str).str.strip()
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


def _as_list(value) -> list:
    """Normalize possibly-singular API values to a list."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def fetch_supported_panther_datasets(
    session=None,
    cache_dir: str | Path = ".cache/biastracker/panther",
    use_cache: bool = True,
) -> dict:
    """Fetch PANTHER's supported annotation dataset metadata."""
    return _fetch_supported_panther_resource(
        "supportedannotdatasets",
        session=session,
        cache_dir=cache_dir,
        use_cache=use_cache,
    )


def fetch_supported_panther_genomes(
    session=None,
    cache_dir: str | Path = ".cache/biastracker/panther",
    use_cache: bool = True,
) -> dict:
    """Fetch PANTHER's supported organism/genome metadata."""
    return _fetch_supported_panther_resource(
        "supportedgenomes",
        session=session,
        cache_dir=cache_dir,
        use_cache=use_cache,
    )


def fetch_panther_geneinfo_batch(
    ids: list[str],
    organism: int | str = 9606,
    session=None,
    timeout=90,
) -> dict:
    """Fetch one PANTHER geneinfo batch for the given IDs."""
    clean_ids = _unique_nonempty(ids)
    if not clean_ids:
        raise ValueError("No IDs provided for PANTHER geneinfo")

    response = request_with_retries(
        "POST",
        f"{PANTHER_BASE_URL}/geneinfo",
        session=session,
        timeout=timeout,
        data={
            "organism": str(organism),
            "geneInputList": ",".join(clean_ids),
        },
        headers={"Accept": "application/json"},
    )
    payload = response.json()
    error = payload.get("search", {}).get("error") if isinstance(payload, dict) else None
    if error:
        raise RuntimeError(f"PANTHER geneinfo error: {error}")
    return payload


def _extract_uniprot_from_panther_accession(accession: str) -> str:
    """Extract a UniProt accession from a PANTHER accession string.

    PANTHER accession strings use the format::

        HUMAN|HGNC=1100|UniProtKB=P38398

    This helper searches for the ``UniProtKB=<ACCESSION>`` pattern and returns
    the accession (e.g. ``P38398``). Returns an empty string when the pattern
    is not present.
    """
    match = re.search(r"UniProtKB=([A-Z0-9]+)", accession)
    if match:
        return match.group(1)
    return ""


def parse_panther_geneinfo(
    payload: dict,
    requested_ids: list[str] | None = None,
    categories: list[str] | None = None,
) -> pd.DataFrame:
    """Convert PANTHER geneinfo JSON to BiasTracker long-form rows.

    ``primary_id`` is resolved in priority order:

    1. UniProt accession extracted from the PANTHER accession string
       (e.g. ``HUMAN|HGNC=1100|UniProtKB=P38398`` → ``P38398``).
    2. The ``input_id`` / ``mapped_id`` field returned by the API.
    3. The raw PANTHER accession string as a last resort.

    Positional index-based fallback (``requested[index]``) is intentionally
    omitted because PANTHER may return genes in a different order than the
    request, which causes silent ID misassignment.

    When *requested_ids* is provided, only rows whose ``primary_id`` appears in
    that set are kept.
    """
    requested = _unique_nonempty(requested_ids or [])
    requested_set = set(requested)
    selected_categories = set(categories) if categories is not None else None
    dataset_to_category = {value: key for key, value in PANTHER_DATASETS.items()}
    genes = _as_list(_dig(payload, "search", "mapped_genes", "gene"))
    rows: list[dict[str, str]] = []

    for gene in genes:
        if not isinstance(gene, dict):
            continue

        panther_accession = _extract_panther_accession(gene)

        # Priority 1: UniProt accession embedded in the PANTHER accession string.
        primary_id = _extract_uniprot_from_panther_accession(panther_accession)
        # Priority 2: explicit input_id / mapped_id field from the API response.
        if not primary_id:
            primary_id = _extract_input_id(gene)
        # Priority 3: fall back to the raw PANTHER accession (no index guess).
        if not primary_id:
            primary_id = panther_accession
        if not primary_id:
            continue

        # Filter to requested IDs when a set was supplied.
        if requested_set and primary_id not in requested_set:
            continue

        ann_types = _as_list(
            _dig(gene, "annotation_type_list", "annotation_data_type")
        )
        for ann_type in ann_types:
            if not isinstance(ann_type, dict):
                continue

            dataset_id = _extract_dataset_id(ann_type)
            category = dataset_to_category.get(dataset_id, dataset_id or "unknown")
            if selected_categories is not None and category not in selected_categories:
                continue

            annotations = _as_list(_dig(ann_type, "annotation_list", "annotation"))
            for annotation in annotations:
                if not isinstance(annotation, dict):
                    continue

                term_id = _clean_optional_value(
                    _first_present(annotation, ("id", "term_id", "accession"))
                )
                term_name = _clean_optional_value(
                    _first_present(
                        annotation,
                        ("name", "label", "term", "description", "content"),
                    )
                )
                if not term_id and not term_name:
                    continue

                rows.append(
                    {
                        "primary_id": primary_id,
                        "term_id": term_id,
                        "term_name": term_name,
                        "source": "PANTHER",
                        "category": category,
                        "panther_accession": panther_accession,
                        "panther_dataset_id": dataset_id,
                    }
                )

    out = pd.DataFrame(rows, columns=PANTHER_OUTPUT_COLUMNS)
    if out.empty:
        return out
    return out.drop_duplicates().reset_index(drop=True)


def fetch_panther_annotations(
    ids: list[str],
    organism: int | str = 9606,
    categories: list[str] | None = None,
    batch_size: int = 500,
    cache_dir: str | Path = ".cache/biastracker/panther",
    use_cache: bool = True,
    max_age_days: float | None = DEFAULT_ANNOTATION_TTL_DAYS,
    session=None,
) -> pd.DataFrame:
    """Fetch and normalize PANTHER annotations for IDs.

    Cached batches older than *max_age_days* are re-fetched (pass ``0`` to force a
    refresh, ``None`` to never expire); fresh results are always written back.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")

    clean_ids = _unique_nonempty(ids)
    if not clean_ids:
        raise ValueError("No IDs provided for PANTHER annotations")

    selected_categories = categories or PANTHER_DEFAULT_CATEGORIES
    cache_path = ensure_cache_dir(cache_dir) if use_cache else None
    frames: list[pd.DataFrame] = []

    for batch in _batched(clean_ids, batch_size):
        raw_payload = None
        parsed = None
        raw_cache_file = None
        parsed_cache_file = None

        if cache_path is not None:
            key = _batch_cache_key(batch, organism, selected_categories)
            raw_cache_file = cache_path / f"geneinfo_{key}.json"
            parsed_cache_file = cache_path / f"geneinfo_{key}.csv"
            if cache_is_fresh(parsed_cache_file, max_age_days):
                parsed = pd.read_csv(
                    parsed_cache_file,
                    dtype=str,
                    keep_default_na=False,
                )
            else:
                raw_payload = read_json_cache(raw_cache_file, max_age_days=max_age_days)

        if parsed is None:
            if raw_payload is None:
                raw_payload = fetch_panther_geneinfo_batch(
                    batch,
                    organism=organism,
                    session=session,
                )
                if raw_cache_file is not None:
                    write_json_cache(raw_cache_file, raw_payload)

            parsed = parse_panther_geneinfo(
                raw_payload,
                requested_ids=batch,
                categories=selected_categories,
            )
            if parsed_cache_file is not None:
                parsed.to_csv(parsed_cache_file, index=False)

        frames.append(parsed)

    if not frames:
        return pd.DataFrame(columns=PANTHER_OUTPUT_COLUMNS)

    out = pd.concat(frames, ignore_index=True)
    if out.empty:
        return pd.DataFrame(columns=PANTHER_OUTPUT_COLUMNS)
    return out.drop_duplicates().reset_index(drop=True)


def load_panther_api_annotations(
    ids: list[str],
    name: str = "panther_api",
    organism: int | str = 9606,
    categories: list[str] | None = None,
    batch_size: int = 500,
    cache_dir: str | Path = ".cache/biastracker/panther",
    use_cache: bool = True,
    max_age_days: float | None = DEFAULT_ANNOTATION_TTL_DAYS,
) -> AnnotationSet:
    """Fetch PANTHER API annotations and return an AnnotationSet."""
    selected_categories = categories or PANTHER_DEFAULT_CATEGORIES
    normalized = fetch_panther_annotations(
        ids,
        organism=organism,
        categories=selected_categories,
        batch_size=batch_size,
        cache_dir=cache_dir,
        use_cache=use_cache,
        max_age_days=max_age_days,
    )

    return AnnotationSet(
        name=name,
        source="PANTHER",
        table=normalized,
        id_col="primary_id",
        term_col="term_name",
        term_id_col="term_id",
        category_col="category",
        metadata={
            "organism": organism,
            "categories": selected_categories,
            "n_input_ids": len(_unique_nonempty(ids)),
            "n_annotation_rows": len(normalized),
            "source": "PANTHER_API",
        },
    )


def run_panther_overrep(
    query_ids: list[str],
    organism: int | str = 9606,
    ref_ids: list[str] | None = None,
    ref_organism: int | str = 9606,
    annot_dataset: str = "GO:0008150",
    enrichment_test_type: str = "FISHER",
    correction: str = "FDR",
    session=None,
    timeout=120,
) -> pd.DataFrame:
    """Run PANTHER's external overrepresentation test for validation.

    This wrapper calls PANTHER's official enrichment service and is intended as
    an optional comparison mode. It does not replace BiasTracker's internal ORA
    implementation in ``analysis.enrichment.run_ora``.
    """
    clean_query_ids = _unique_nonempty(query_ids)
    if not clean_query_ids:
        raise ValueError("No query IDs provided for PANTHER overrepresentation")

    data = {
        "geneInputList": ",".join(clean_query_ids),
        "organism": str(organism),
        "annotDataSet": annot_dataset,
        "enrichmentTestType": enrichment_test_type,
        "correction": correction,
    }

    if ref_ids is not None:
        clean_ref_ids = _unique_nonempty(ref_ids)
        if not clean_ref_ids:
            raise ValueError("ref_ids was provided but did not contain any IDs")
        data["refInputList"] = ",".join(clean_ref_ids)
        data["refOrganism"] = str(ref_organism)

    response = request_with_retries(
        "POST",
        f"{PANTHER_BASE_URL}/enrich/overrep",
        session=session,
        timeout=timeout,
        data=data,
        headers={"Accept": "application/json"},
    )
    payload = response.json()
    _raise_panther_payload_error(payload, context="PANTHER overrepresentation")

    category = {value: key for key, value in PANTHER_DATASETS.items()}.get(
        annot_dataset,
        annot_dataset,
    )
    rows: list[dict[str, Any]] = []

    for result in _as_list(_dig(payload, "results", "result")):
        if not isinstance(result, dict):
            continue

        term = result.get("term") if isinstance(result.get("term"), dict) else {}
        term_id = _clean_optional_value(
            _first_present(term, ("id", "term_id", "accession"))
            or _first_present(result, ("term_id", "termId", "id"))
        )
        term_name = _clean_optional_value(
            _first_present(term, ("label", "name", "term", "description"))
            or _first_present(result, ("term_name", "termName", "label", "name"))
        )
        fold_enrichment = _to_float(
            _first_present(
                result,
                ("fold_enrichment", "foldEnrichment", "fold_enrichment_value"),
            )
        )

        rows.append(
            {
                "term_id": term_id or "PANTHER:UNCLASSIFIED",
                "term_name": term_name,
                "source": "PANTHER",
                "category": category,
                "expected": _to_float(_first_present(result, ("expected",))),
                "fold_enrichment": fold_enrichment,
                "p_value": _to_float(
                    _first_present(result, ("p_value", "pValue", "pvalue"))
                ),
                "fdr": _to_float(
                    _first_present(result, ("fdr", "FDR", "false_discovery_rate"))
                ),
                "direction": _overrep_direction(fold_enrichment),
            }
        )

    return pd.DataFrame(rows, columns=PANTHER_OVERREP_COLUMNS)


def _fetch_supported_panther_resource(
    resource: str,
    session=None,
    cache_dir: str | Path = ".cache/biastracker/panther",
    use_cache: bool = True,
) -> dict:
    cache_file = None
    if use_cache:
        cache_file = ensure_cache_dir(cache_dir) / f"{resource}.json"
        cached = read_json_cache(cache_file)
        if cached is not None:
            return cached

    response = request_with_retries(
        "GET",
        f"{PANTHER_BASE_URL}/{resource}",
        session=session,
        headers={"Accept": "application/json"},
    )
    payload = response.json()
    if cache_file is not None:
        write_json_cache(cache_file, payload)
    return payload


def _raise_panther_payload_error(payload: Any, context: str) -> None:
    if not isinstance(payload, dict):
        return

    for path in (
        ("search", "error"),
        ("results", "error"),
        ("error",),
    ):
        error = _dig(payload, *path)
        if error:
            raise RuntimeError(f"{context} error: {error}")


def _unique_nonempty(ids: list[str]) -> list[str]:
    seen: set[str] = set()
    clean_ids: list[str] = []
    for value in ids:
        text = str(value).strip()
        if not text or text.lower() in {"nan", "none"} or text in seen:
            continue
        seen.add(text)
        clean_ids.append(text)
    return clean_ids


def _batched(ids: list[str], batch_size: int) -> list[list[str]]:
    return [ids[index : index + batch_size] for index in range(0, len(ids), batch_size)]


def _batch_cache_key(
    batch: list[str],
    organism: int | str,
    categories: list[str],
) -> str:
    value = json.dumps(
        {"ids": batch, "organism": str(organism), "categories": categories},
        sort_keys=True,
        separators=(",", ":"),
    )
    return safe_cache_key(value)


def _dig(value: Any, *keys: str) -> Any:
    current = value
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _first_present(data: dict, keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = data.get(key)
        if value is not None and str(value).strip():
            return value
    return None


def _clean_optional_value(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _to_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _overrep_direction(fold_enrichment: float | None) -> str:
    if fold_enrichment is None or fold_enrichment == 1:
        return "neutral"
    if fold_enrichment > 1:
        return "enriched"
    return "depleted"


def _extract_panther_accession(gene: dict) -> str:
    for key in ("accession", "panther_accession", "panther_id", "id"):
        value = gene.get(key)
        if value:
            return str(value).strip()

    for nested_key in ("gene", "gene_info", "mapped_gene"):
        nested = gene.get(nested_key)
        if isinstance(nested, dict):
            for key in ("accession", "panther_accession", "id"):
                value = nested.get(key)
                if value:
                    return str(value).strip()
    return ""


def _extract_input_id(gene: dict) -> str:
    for key in (
        "input_id",
        "inputId",
        "input",
        "search_id",
        "searchId",
        "geneInput",
        "mapped_id",
        "primary_id",
    ):
        value = gene.get(key)
        if value:
            return str(value).strip()

    for nested_key in ("gene", "gene_info", "mapped_gene"):
        nested = gene.get(nested_key)
        if isinstance(nested, dict):
            value = _first_present(
                nested,
                ("input_id", "inputId", "search_id", "mapped_id", "primary_id"),
            )
            if value:
                return str(value).strip()
    return ""


def _extract_dataset_id(ann_type: dict) -> str:
    for key in ("content", "id", "annotation_data_type", "type", "name"):
        value = ann_type.get(key)
        if value:
            return str(value).strip()
    return ""
