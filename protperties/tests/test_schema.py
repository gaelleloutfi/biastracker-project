import pytest
import pandas as pd

from protperties.schema import ensure_sequence_table


def test_ensure_sequence_table_basic_peptide():
    df = pd.DataFrame({
        "Stripped.Sequence": ["PEPTIDEK", "ACDEFGHIK"],
        "Run": ["R1", "R1"],
        "mw": [900.0, 1000.0],
    })

    result = ensure_sequence_table(df, level="peptide")

    assert "level" in result.columns
    assert (result["level"] == "peptide").all()
    assert "sequence" in result.columns
    assert list(result["sequence"]) == ["PEPTIDEK", "ACDEFGHIK"]
    # Original columns must be preserved
    assert "Run" in result.columns
    assert "mw" in result.columns
    assert "Stripped.Sequence" in result.columns


def test_ensure_sequence_table_with_id_col():
    df = pd.DataFrame({
        "Stripped.Sequence": ["PEPTIDEK", "ACDEFGHIK"],
        "my_id": ["id1", "id2"],
    })

    result = ensure_sequence_table(df, level="peptide", id_col="my_id")

    assert "primary_id" in result.columns
    assert list(result["primary_id"]) == ["id1", "id2"]


def test_ensure_sequence_table_missing_sequence_raises():
    df = pd.DataFrame({
        "Run": ["R1", "R2"],
        "intensity": [100.0, 200.0],
    })

    with pytest.raises(ValueError, match="sequence"):
        ensure_sequence_table(df, level="peptide")


def test_ensure_sequence_table_missing_id_col_raises():
    df = pd.DataFrame({
        "sequence": ["PEPTIDEK"],
    })

    with pytest.raises(ValueError, match="nonexistent_col"):
        ensure_sequence_table(df, level="peptide", id_col="nonexistent_col")


def test_ensure_sequence_table_protein_level():
    df = pd.DataFrame({
        "sequence": ["PEPTIDEK", "ACDEFGHIK"],
    })

    result = ensure_sequence_table(df, level="protein")

    assert "level" in result.columns
    assert (result["level"] == "protein").all()


def test_ensure_sequence_table_preserves_existing():
    df = pd.DataFrame({
        "sequence": ["MYPEPTIDE", "ANOTHER"],
        "Run": ["R1", "R2"],
    })

    result = ensure_sequence_table(df, level="peptide")

    assert list(result["sequence"]) == ["MYPEPTIDE", "ANOTHER"]
    assert "Stripped.Sequence" not in result.columns


def test_ensure_sequence_table_no_id_col():
    df = pd.DataFrame({
        "sequence": ["PEPTIDEK"],
        "Run": ["R1"],
    })

    result = ensure_sequence_table(df, level="peptide", id_col=None)

    assert "primary_id" not in result.columns


def test_ensure_sequence_table_does_not_modify_input():
    df = pd.DataFrame({
        "Stripped.Sequence": ["PEPTIDEK"],
        "Run": ["R1"],
    })

    ensure_sequence_table(df, level="peptide")

    assert "level" not in df.columns
    assert "sequence" not in df.columns
