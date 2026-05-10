import re
import warnings
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd

from biastracker.dataset import AnnotationSet


REQUIRED_HPA_COLUMNS = ("Gene", "Gene name")
LOCATION_COLUMNS = (
    "Main location",
    "Additional location",
    "Extracellular location",
)
LOCATION_CATEGORY = {
    "Main location": "hpa_main_location",
    "Additional location": "hpa_additional_location",
    "Extracellular location": "hpa_extracellular_location",
}
OUTPUT_COLUMNS = [
    "primary_id",
    "term_id",
    "term_name",
    "source",
    "category",
    "gene_ensembl_id",
    "gene_name",
    "reliability",
    "location_type",
]


def read_hpa_subcellular_tsv(path: str | Path) -> pd.DataFrame:
    """Read a local HPA subcellular TSV or ZIP archive."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() == ".zip":
        df = _read_first_tsv_from_zip(path)
    else:
        df = pd.read_csv(path, sep="\t")

    _validate_hpa_columns(df)
    return df


def _split_hpa_locations(value: Any) -> list[str]:
    """Split semicolon-separated HPA location labels."""
    if value is None or pd.isna(value):
        return []

    return [part.strip() for part in str(value).split(";") if part.strip()]


def _parse_hpa_go_pairs(go_raw: Any) -> dict[str, str]:
    """Parse HPA GO id values into location label to GO ID mappings."""
    if go_raw is None or pd.isna(go_raw):
        return {}

    pairs: dict[str, str] = {}
    pattern = re.compile(r"^\s*(.*?)\s*\((GO:\d+)\)\s*$")
    for entry in str(go_raw).split(";"):
        match = pattern.match(entry)
        if match:
            pairs[match.group(1).strip()] = match.group(2)
    return pairs


def _normalize_hpa_location_id(location: str) -> str:
    """Create a stable synthetic HPA location ID."""
    normalized = re.sub(r"[^a-z0-9]+", "_", location.lower())
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return f"HPA_LOC:{normalized}"


def _make_hpa_term_id(location: str, go_pairs: dict[str, str]) -> str:
    """Return the GO ID for a location, or a synthetic HPA ID fallback."""
    return go_pairs.get(location, _normalize_hpa_location_id(location))


def normalize_hpa_subcellular(
    df: pd.DataFrame,
    map_to_uniprot: bool = False,
    ensembl_to_uniprot: dict[str, list[str]] | None = None,
    include_location_types: list[str] | None = None,
) -> pd.DataFrame:
    """Normalize raw HPA subcellular rows into BiasTracker annotation rows."""
    _validate_hpa_columns(df)

    if map_to_uniprot and ensembl_to_uniprot is None:
        raise ValueError("ensembl_to_uniprot must be provided when map_to_uniprot=True")

    selected_location_types = include_location_types or list(LOCATION_COLUMNS)
    unsupported = [
        column for column in selected_location_types if column not in LOCATION_CATEGORY
    ]
    if unsupported:
        raise ValueError(f"Unsupported HPA location columns: {', '.join(unsupported)}")

    rows: list[dict[str, str]] = []
    unmapped_genes: set[str] = set()

    for _, row in df.iterrows():
        gene_id = str(row["Gene"]).strip()
        gene_name = _clean_optional_value(row.get("Gene name"))
        reliability = _clean_optional_value(row.get("Reliability"))
        go_pairs = _parse_hpa_go_pairs(row.get("GO id"))

        primary_ids = [gene_id]
        if map_to_uniprot:
            primary_ids = [str(value) for value in ensembl_to_uniprot.get(gene_id, [])]
            if not primary_ids:
                unmapped_genes.add(gene_id)
                continue

        for location_type in selected_location_types:
            if location_type not in df.columns:
                continue

            for location in _split_hpa_locations(row.get(location_type)):
                term_id = _make_hpa_term_id(location, go_pairs)
                for primary_id in primary_ids:
                    rows.append(
                        {
                            "primary_id": primary_id,
                            "term_id": term_id,
                            "term_name": location,
                            "source": "HPA",
                            "category": LOCATION_CATEGORY[location_type],
                            "gene_ensembl_id": gene_id,
                            "gene_name": gene_name,
                            "reliability": reliability,
                            "location_type": location_type,
                        }
                    )

    if unmapped_genes:
        warnings.warn(
            f"{len(unmapped_genes)} HPA genes were not mapped to UniProt accessions",
            RuntimeWarning,
            stacklevel=2,
        )

    out = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    if out.empty:
        return out
    return out.drop_duplicates().reset_index(drop=True)


def load_hpa_subcellular(
    path: str | Path,
    name: str = "hpa_subcellular",
    map_to_uniprot: bool = False,
    ensembl_to_uniprot: dict[str, list[str]] | None = None,
    cache_dir: str | Path | None = None,
    use_uniprot_api: bool = False,
    from_ns: str = "Ensembl",
    to_ns: str = "UniProtKB",
) -> AnnotationSet:
    """Load a local HPA subcellular location TSV into an AnnotationSet."""
    raw_df = read_hpa_subcellular_tsv(path)

    mapping = ensembl_to_uniprot
    if map_to_uniprot:
        if mapping is None:
            if not use_uniprot_api:
                raise ValueError(
                    "map_to_uniprot=True requires ensembl_to_uniprot to be provided "
                    "or use_uniprot_api=True"
                )
            from biastracker.annotations.uniprot_mapping import map_ids_to_uniprot

            ids = raw_df["Gene"].dropna().astype(str).str.strip().unique().tolist()
            kwargs = {
                "ids": ids,
                "from_ns": from_ns,
                "to_ns": to_ns,
            }
            if cache_dir is not None:
                kwargs["cache_dir"] = cache_dir
            mapping = map_ids_to_uniprot(**kwargs)

    normalized = normalize_hpa_subcellular(
        raw_df,
        map_to_uniprot=map_to_uniprot,
        ensembl_to_uniprot=mapping,
    )

    return AnnotationSet(
        name=name,
        source="HPA",
        table=normalized,
        id_col="primary_id",
        term_col="term_name",
        term_id_col="term_id",
        category_col="category",
        metadata={
            "n_raw_rows": len(raw_df),
            "n_annotation_rows": len(normalized),
            "map_to_uniprot": map_to_uniprot,
            "id_namespace": "UniProtKB" if map_to_uniprot else "Ensembl",
            "from_ns": from_ns,
            "to_ns": to_ns,
            "input_path": str(Path(path)),
        },
    )


def _read_first_tsv_from_zip(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as archive:
        candidates = [
            name
            for name in archive.namelist()
            if not name.endswith("/")
            and Path(name).suffix.lower() in {".tsv", ".txt", ".tab"}
        ]
        if not candidates:
            raise ValueError(f"No TSV/TXT file found inside HPA ZIP archive: {path}")

        with archive.open(candidates[0]) as handle:
            return pd.read_csv(handle, sep="\t")


def _validate_hpa_columns(df: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_HPA_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"HPA subcellular file is missing columns: {', '.join(missing)}")


def _clean_optional_value(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()
