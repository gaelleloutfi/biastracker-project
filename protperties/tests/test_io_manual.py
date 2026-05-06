import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from protperties.io_manual import from_manual_table


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write(tmp_path: Path, filename: str, content: str) -> Path:
    p = tmp_path / filename
    p.write_text(content)
    return p


# ---------------------------------------------------------------------------
# Basic TSV / CSV loading
# ---------------------------------------------------------------------------

def test_peptide_tsv(tmp_path):
    """Load a TSV with peptide sequences; check basic structure and properties."""
    path = _write(tmp_path, "peptides.tsv",
                  "sequence\tintensity\n"
                  "PEPTIDEK\t1000.0\n"
                  "ACDEFGHIK\t2000.0\n")

    result = from_manual_table(path, level="peptide")

    assert len(result) == 2
    assert (result["level"] == "peptide").all()

    # physicochemical properties
    for col in ("length", "mw", "pi", "gravy", "instability",
                "aromaticity", "aliphatic_index", "ext_reduced",
                "ext_cystine", "charge_at_pH"):
        assert col in result.columns, f"Missing column: {col}"

    # missed cleavages present for peptides
    assert "missed_cleavages" in result.columns

    # expression column taken from 'intensity'
    assert "expression" in result.columns
    assert result["expression"].iloc[0] == 1000.0
    assert result["expression"].iloc[1] == 2000.0


def test_protein_csv(tmp_path):
    """Load a CSV with protein sequences; missed_cleavages must NOT be added."""
    path = _write(tmp_path, "proteins.csv",
                  "Sequence,Expression\n"
                  "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAAVRESVPSLL,5000.0\n")

    result = from_manual_table(path, level="protein")

    assert len(result) == 1
    assert (result["level"] == "protein").all()
    assert "missed_cleavages" not in result.columns
    assert result["expression"].iloc[0] == 5000.0


# ---------------------------------------------------------------------------
# Separator inference
# ---------------------------------------------------------------------------

def test_sep_inference_tab(tmp_path):
    path = _write(tmp_path, "data.tsv",
                  "sequence\tintensity\nPEPTIDEK\t42.0\n")
    result = from_manual_table(path, level="peptide", sep=None)
    assert len(result) == 1


def test_sep_inference_comma(tmp_path):
    path = _write(tmp_path, "data.csv",
                  "sequence,intensity\nPEPTIDEK,42.0\n")
    result = from_manual_table(path, level="peptide", sep=None)
    assert len(result) == 1


def test_sep_explicit_tab(tmp_path):
    path = _write(tmp_path, "data.tsv",
                  "sequence\tintensity\nPEPTIDEK\t42.0\n")
    result = from_manual_table(path, level="peptide", sep="\t")
    assert len(result) == 1


def test_sep_explicit_comma(tmp_path):
    path = _write(tmp_path, "data.csv",
                  "sequence,intensity\nPEPTIDEK,42.0\n")
    result = from_manual_table(path, level="peptide", sep=",")
    assert len(result) == 1


def test_sep_semicolon(tmp_path):
    path = _write(tmp_path, "data.csv",
                  "sequence;intensity\nPEPTIDEK;42.0\n")
    result = from_manual_table(path, level="peptide", sep=None)
    assert len(result) == 1


# ---------------------------------------------------------------------------
# Sequence column detection
# ---------------------------------------------------------------------------

def test_seq_col_stripped_sequence(tmp_path):
    path = _write(tmp_path, "data.tsv",
                  "Stripped.Sequence\tintensity\nPEPTIDEK\t1.0\n")
    result = from_manual_table(path, level="peptide")
    assert "sequence" in result.columns
    assert result["sequence"].iloc[0] == "PEPTIDEK"


def test_seq_col_Sequence_capitalised(tmp_path):
    path = _write(tmp_path, "data.tsv",
                  "Sequence\tintensity\nPEPTIDEK\t1.0\n")
    result = from_manual_table(path, level="peptide")
    assert result["sequence"].iloc[0] == "PEPTIDEK"


def test_seq_col_not_found_raises(tmp_path):
    path = _write(tmp_path, "data.tsv",
                  "peptide_seq\tintensity\nPEPTIDEK\t1.0\n")
    with pytest.raises(ValueError, match="sequence column"):
        from_manual_table(path, level="peptide")


# ---------------------------------------------------------------------------
# Expression column detection and fill
# ---------------------------------------------------------------------------

def test_expr_col_not_found_uses_fill(tmp_path):
    """When no expression column exists the column is created with fill_expression."""
    path = _write(tmp_path, "data.tsv",
                  "sequence\nPEPTIDEK\n")
    result = from_manual_table(path, level="peptide", fill_expression=0.0)
    assert "expression" in result.columns
    assert result["expression"].iloc[0] == 0.0


def test_expr_col_custom_fill(tmp_path):
    path = _write(tmp_path, "data.tsv",
                  "sequence\nPEPTIDEK\n")
    result = from_manual_table(path, level="peptide", fill_expression=99.9)
    assert result["expression"].iloc[0] == pytest.approx(99.9)


def test_expr_col_precursor_normalised(tmp_path):
    path = _write(tmp_path, "data.tsv",
                  "sequence\tPrecursor.Normalised\nPEPTIDEK\t777.0\n")
    result = from_manual_table(path, level="peptide")
    assert result["expression"].iloc[0] == 777.0


# ---------------------------------------------------------------------------
# Physicochemical properties
# ---------------------------------------------------------------------------

def test_properties_computed(tmp_path):
    path = _write(tmp_path, "data.tsv",
                  "sequence\tintensity\nPEPTIDEK\t1.0\n")
    result = from_manual_table(path, level="peptide")
    row = result.iloc[0]
    assert row["length"] == len("PEPTIDEK")
    assert row["mw"] > 0
    assert isinstance(row["pi"], float)
    assert isinstance(row["gravy"], float)


# ---------------------------------------------------------------------------
# Missed cleavages
# ---------------------------------------------------------------------------

def test_missed_cleavages_for_peptides(tmp_path):
    # PEPTIDEKR has one internal K before the terminal R → 1 MC
    path = _write(tmp_path, "data.tsv",
                  "sequence\tintensity\nPEPTIDEKR\t1.0\nACDEFGHIK\t2.0\n")
    result = from_manual_table(path, level="peptide")
    assert "missed_cleavages" in result.columns
    # PEPTIDEKR: internal K before R counts → MC == 1
    assert result.loc[result["sequence"] == "PEPTIDEKR", "missed_cleavages"].iloc[0] == 1
    # ACDEFGHIK: no internal K/R → MC == 0
    assert result.loc[result["sequence"] == "ACDEFGHIK", "missed_cleavages"].iloc[0] == 0


def test_missed_cleavages_absent_for_proteins(tmp_path):
    path = _write(tmp_path, "data.csv",
                  "Sequence,intensity\nPEPTIDEK,1.0\n")
    result = from_manual_table(path, level="protein")
    assert "missed_cleavages" not in result.columns


# ---------------------------------------------------------------------------
# File-not-found
# ---------------------------------------------------------------------------

def test_file_not_found():
    with pytest.raises(FileNotFoundError):
        from_manual_table("/nonexistent/path/file.tsv", level="peptide")


# ---------------------------------------------------------------------------
# Numeric expression coercion
# ---------------------------------------------------------------------------

def test_expression_numeric_coercion(tmp_path):
    """Non-numeric expression values are coerced; bad values become fill_expression."""
    path = _write(tmp_path, "data.tsv",
                  "sequence\tintensity\nPEPTIDEK\t1000\nACDEFGHIK\tn/a\n")
    result = from_manual_table(path, level="peptide", fill_expression=0.0)
    assert result["expression"].dtype.kind == "f"
    assert result["expression"].iloc[0] == pytest.approx(1000.0)
    # "n/a" cannot be coerced → filled with fill_expression
    assert result["expression"].iloc[1] == pytest.approx(0.0)


def test_expression_is_numeric_dtype(tmp_path):
    """expression column must be float even when all values are valid numbers."""
    path = _write(tmp_path, "data.tsv",
                  "sequence\tintensity\nPEPTIDEK\t42\n")
    result = from_manual_table(path, level="peptide")
    assert np.issubdtype(result["expression"].dtype, np.number)


# ---------------------------------------------------------------------------
# Empty dataframe: sequence + expression columns must exist before returning
# ---------------------------------------------------------------------------

def test_empty_df_has_sequence_column(tmp_path):
    """An empty input file must still produce a DataFrame with a 'sequence' column."""
    path = _write(tmp_path, "data.tsv",
                  "Stripped.Sequence\tIntensity\n")
    result = from_manual_table(path, level="peptide")
    assert "sequence" in result.columns
    assert "expression" in result.columns
    assert "level" in result.columns
    assert len(result) == 0


# ---------------------------------------------------------------------------
# Properties computed on canonical 'sequence' column
# ---------------------------------------------------------------------------

def test_props_use_canonical_sequence_column(tmp_path):
    """Properties must be computed on the renamed 'sequence' column, not the raw one."""
    # File uses 'Stripped.Sequence' header; after rename the props are on 'sequence'
    path = _write(tmp_path, "data.tsv",
                  "Stripped.Sequence\tintensity\nPEPTIDEK\t1.0\n")
    result = from_manual_table(path, level="peptide")
    # 'Stripped.Sequence' should have been renamed; no residual column
    assert "Stripped.Sequence" not in result.columns
    assert result["sequence"].iloc[0] == "PEPTIDEK"
    assert result["length"].iloc[0] == len("PEPTIDEK")

def test_level_column_set(tmp_path):
    path = _write(tmp_path, "data.tsv",
                  "sequence\tintensity\nPEPTIDEK\t1.0\n")
    result = from_manual_table(path, level="peptide")
    assert "level" in result.columns
    assert result["level"].iloc[0] == "peptide"

    path2 = _write(tmp_path, "data2.tsv",
                   "sequence\tintensity\nPEPTIDEK\t1.0\n")
    result2 = from_manual_table(path2, level="protein")
    assert result2["level"].iloc[0] == "protein"
