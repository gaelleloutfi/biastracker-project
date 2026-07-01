import copy
from dataclasses import dataclass, field
from typing import Literal
import pandas as pd
from pathlib import Path

from biastracker.preprocessing import select_numeric_features

def check_protperties_available():
    """
    Check if protperties and its core functions are available.
    """
    try:
        from protperties import (
            from_diann_parquet, # noqa: F401
            from_maxquant_evidence, # noqa: F401
            from_maxquant_proteingroups, # noqa: F401
            from_diann_pg_matrix, # noqa: F401
            from_manual_table, # noqa: F401
            build_protein_table, # noqa: F401
            add_ptr_annotation, # noqa: F401
        ) # noqa: F401
    except ImportError as e:
        raise ImportError(
            "Failed to import required functions from protperties. "
            "Please ensure protperties is installed in editable mode:\n"
            "python -m pip install -e ../protperties\n"
            "Original error: " + str(e)
        ) from e


@dataclass
class BiasDataset:
    name: str
    table: pd.DataFrame
    level: Literal["protein", "peptide"]
    source_type: str = "unknown"
    id_col: str = "primary_id"
    group_col: str | None = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.table, pd.DataFrame):
            raise TypeError("table must be a DataFrame")
        if self.level not in ("protein", "peptide"):
            raise ValueError('level must be "protein" or "peptide"')
        if "level" not in self.table.columns:
            raise ValueError('table must contain a "level" column')
        if "sequence" not in self.table.columns:
            raise ValueError('table must contain a "sequence" column')
        if self.group_col is not None and self.group_col not in self.table.columns:
            raise ValueError(f'group_col "{self.group_col}" must exist in table')

    def copy(self) -> "BiasDataset":
        return BiasDataset(
            name=self.name,
            table=self.table.copy(),
            level=self.level,
            source_type=self.source_type,
            id_col=self.id_col,
            group_col=self.group_col,
            metadata=copy.deepcopy(self.metadata)
        )

    def available_features(self, features: list[str] | None = None) -> list[str]:
        return select_numeric_features(self.table, features=features)

    def ids(self) -> set[str]:
        if self.id_col is not None and self.id_col in self.table.columns:
            return set(self.table[self.id_col].dropna().astype(str).unique())
        return set()

    def has_groups(self) -> bool:
        return self.group_col is not None and self.group_col in self.table.columns


@dataclass
class AnnotationSet:
    name: str
    source: str
    table: pd.DataFrame
    id_col: str = "primary_id"
    term_col: str = "term_name"
    term_id_col: str = "term_id"
    category_col: str = "category"
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.table, pd.DataFrame):
            raise TypeError("table must be a DataFrame")
        
        if self.id_col not in self.table.columns:
            raise ValueError(f'Required column "{self.id_col}" is missing')
        if self.term_col not in self.table.columns:
            raise ValueError(f'Required column "{self.term_col}" is missing')

        # Drop rows with missing primary_id or term_name
        self.table = self.table.dropna(subset=[self.id_col, self.term_col]).copy()
        
        # Cast IDs and terms to strings
        self.table[self.id_col] = self.table[self.id_col].astype(str)
        self.table[self.term_col] = self.table[self.term_col].astype(str)
        
        # if term_id is missing, create it from term_name
        if self.term_id_col not in self.table.columns:
            self.table[self.term_id_col] = self.table[self.term_col]
        else:
            self.table[self.term_id_col] = self.table[self.term_id_col].fillna(self.table[self.term_col])
            
        # if category is missing, create it with value "unknown"
        if self.category_col not in self.table.columns:
            self.table[self.category_col] = "unknown"
        else:
            self.table[self.category_col] = self.table[self.category_col].fillna("unknown")

    def terms(self) -> list[str]:
        return self.table[self.term_col].unique().tolist()

    def ids_for_term(self, term_name: str) -> set[str]:
        subset = self.table[self.table[self.term_col] == term_name]
        return set(subset[self.id_col].unique())

    def subset_to_ids(self, ids: set[str]) -> "AnnotationSet":
        sub_table = self.table[self.table[self.id_col].isin(ids)].copy()
        return AnnotationSet(
            name=self.name,
            source=self.source,
            table=sub_table,
            id_col=self.id_col,
            term_col=self.term_col,
            term_id_col=self.term_id_col,
            category_col=self.category_col,
            metadata=copy.deepcopy(self.metadata)
        )


def load_standard_table(
    path: str,
    name: str,
    level: Literal["protein", "peptide"],
    source_type: str = "standard_csv",
    group_col: str | None = None,
    id_col: str = "primary_id"
) -> BiasDataset:
    """Load a standard delimited table directly using pandas.

    CSV files are read comma-separated. TSV, TAB, and TXT files are read
    tab-separated. Other extensions try comma first and retry tab-separated if
    comma parsing produces a single column.
    """
    path_obj = Path(path)
    ext = path_obj.suffix.lower()

    try:
        sep = "\t" if ext in {".tsv", ".tab", ".txt"} else ","
        df = pd.read_csv(path, sep=sep)
        if sep == "," and df.shape[1] == 1:
            df = pd.read_csv(path, sep="\t")
    except Exception as e:
        raise RuntimeError(f"Failed to load standard table from {path}: {e}") from e

    if "sequence" not in df.columns:
        raise ValueError('Standard table must contain a "sequence" column')

    if "level" not in df.columns:
        df["level"] = level
    else:
        existing_levels = df["level"].dropna().unique()
        if len(existing_levels) > 1 or (len(existing_levels) == 1 and existing_levels[0] != level):
            raise ValueError(f"Table contains levels {existing_levels} but expected only '{level}'")

    return BiasDataset(
        name=name,
        table=df,
        level=level,
        source_type=source_type,
        id_col=id_col,
        group_col=group_col
    )


def load_manual_table_via_protperties(path: str, name: str, level: Literal["protein", "peptide"], **kwargs) -> BiasDataset:
    """Load a table using protperties.from_manual_table."""
    try:
        from protperties import from_manual_table
    except ImportError as e:
        raise ImportError("Failed to import 'from_manual_table' from protperties.") from e

    group_col = kwargs.pop("group_col", None)
    id_col = kwargs.pop("id_col", "primary_id")
    df = from_manual_table(path, level=level, **kwargs)
    return BiasDataset(
        name=name,
        table=df,
        level=level,
        source_type="manual",
        id_col=id_col,
        group_col=group_col
    )


def load_diann_report(path: str, name: str, **kwargs) -> BiasDataset:
    """Load a DIA-NN report using protperties."""
    try:
        from protperties import from_diann_parquet
    except ImportError as e:
        raise ImportError("Failed to import 'from_diann_parquet' from protperties.") from e

    group_col = kwargs.pop("group_col", None)
    id_col = kwargs.pop("id_col", "primary_id")
    df = from_diann_parquet(path, **kwargs)
    return BiasDataset(
        name=name,
        table=df,
        level="peptide",
        source_type="diann",
        id_col=id_col,
        group_col=group_col
    )


def load_maxquant_evidence(path: str, name: str, **kwargs) -> BiasDataset:
    """Load a MaxQuant evidence table using protperties."""
    try:
        from protperties import from_maxquant_evidence
    except ImportError as e:
        raise ImportError("Failed to import 'from_maxquant_evidence' from protperties.") from e

    group_col = kwargs.pop("group_col", None)
    id_col = kwargs.pop("id_col", "primary_id")
    df = from_maxquant_evidence(path, **kwargs)
    return BiasDataset(
        name=name,
        table=df,
        level="peptide",
        source_type="maxquant_evidence",
        id_col=id_col,
        group_col=group_col
    )


def load_maxquant_proteingroups(path: str, name: str, **kwargs) -> BiasDataset:
    """Load a MaxQuant proteinGroups table using protperties."""
    try:
        from protperties import from_maxquant_proteingroups
    except ImportError as e:
        raise ImportError("Failed to import 'from_maxquant_proteingroups' from protperties.") from e

    group_col = kwargs.pop("group_col", None)
    id_col = kwargs.pop("id_col", "primary_id")
    kwargs.setdefault("fetch_sequences", True)
    df = from_maxquant_proteingroups(path, **kwargs)
    return BiasDataset(
        name=name,
        table=df,
        level="protein",
        source_type="maxquant_proteingroups",
        id_col=id_col,
        group_col=group_col
    )


def load_diann_pg_matrix(path: str, name: str, **kwargs) -> BiasDataset:
    """Load a DIA-NN pg_matrix using protperties."""
    try:
        from protperties import from_diann_pg_matrix
    except ImportError as e:
        raise ImportError("Failed to import 'from_diann_pg_matrix' from protperties.") from e

    group_col = kwargs.pop("group_col", None)
    id_col = kwargs.pop("id_col", "primary_id")
    kwargs.setdefault("fetch_sequences", True)
    df = from_diann_pg_matrix(path, **kwargs)
    return BiasDataset(
        name=name,
        table=df,
        level="protein",
        source_type="diann_pg_matrix",
        id_col=id_col,
        group_col=group_col
    )
