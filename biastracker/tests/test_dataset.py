import pytest
import pandas as pd
import numpy as np
from biastracker.dataset import BiasDataset, AnnotationSet, load_standard_table

# BiasDataset tests

def test_biasdataset_valid(toy_protein_df):
    ds = BiasDataset(name="test", table=toy_protein_df, level="protein", metadata={"key": "val"})
    assert ds.name == "test"
    assert ds.level == "protein"
    assert ds.metadata["key"] == "val"

def test_biasdataset_missing_sequence(toy_protein_df):
    df = toy_protein_df.drop(columns=["sequence"])
    with pytest.raises(ValueError, match="sequence"):
        BiasDataset(name="test", table=df, level="protein")

def test_biasdataset_invalid_level(toy_protein_df):
    with pytest.raises(ValueError, match="level must be"):
        BiasDataset(name="test", table=toy_protein_df, level="invalid")

def test_biasdataset_missing_group_col(toy_protein_df):
    with pytest.raises(ValueError, match="must exist in table"):
        BiasDataset(name="test", table=toy_protein_df, level="protein", group_col="missing_group")

def test_biasdataset_ids(toy_protein_df):
    ds = BiasDataset(name="test", table=toy_protein_df, level="protein")
    ids = ds.ids()
    assert set(ids) == {"P1", "P2", "P3", "P4", "P5", "P6"}

def test_biasdataset_available_features(toy_protein_df):
    ds = BiasDataset(name="test", table=toy_protein_df, level="protein")
    feats = ds.available_features()
    assert "length" in feats
    assert "mw" in feats
    assert "pi" in feats
    assert "sequence" not in feats # not numeric
    
    feats_subset = ds.available_features(["length", "sequence", "missing"])
    assert feats_subset == ["length"]

def test_biasdataset_metadata_preserved(toy_protein_df):
    ds = BiasDataset(name="test", table=toy_protein_df, level="protein", metadata={"custom": "info"})
    ds_copy = ds.copy()
    assert ds_copy.metadata["custom"] == "info"

# AnnotationSet tests

def test_annotationset_valid(toy_annotation_df):
    ann = AnnotationSet(name="test", source="GO", table=toy_annotation_df)
    assert ann.name == "test"

def test_annotationset_fills_missing(toy_annotation_df):
    df = toy_annotation_df.drop(columns=["term_id", "category"])
    ann = AnnotationSet(name="test", source="test", table=df)
    assert "term_id" in ann.table.columns
    assert "category" in ann.table.columns
    assert ann.table["term_id"].iloc[0] == ann.table["term_name"].iloc[0]
    assert ann.table["category"].iloc[0] == "unknown"

def test_annotationset_drops_invalid():
    df = pd.DataFrame({
        "primary_id": ["A", None, "C", "D"],
        "term_name": ["Term1", "Term2", None, np.nan]
    })
    ann = AnnotationSet(name="test", source="test", table=df)
    assert len(ann.table) == 1
    assert ann.table.iloc[0]["primary_id"] == "A"
    assert ann.table.iloc[0]["term_name"] == "Term1"

def test_annotationset_terms(toy_annotation_set):
    terms = toy_annotation_set.terms()
    assert set(terms) == {"membrane", "nucleus"}

def test_annotationset_ids_for_term(toy_annotation_set):
    ids = toy_annotation_set.ids_for_term("membrane")
    assert set(ids) == {"P1", "P2", "P3"}

def test_annotationset_subset_to_ids(toy_annotation_set):
    sub_ann = toy_annotation_set.subset_to_ids({"P1", "P4"})
    assert set(sub_ann.table["primary_id"]) == {"P1", "P4"}

# Standard table loading tests

def test_load_standard_table(tmp_standard_csv):
    ds = load_standard_table(str(tmp_standard_csv), name="test", level="protein")
    assert ds.name == "test"
    assert ds.level == "protein"
    assert "level" in ds.table.columns

def test_load_standard_table_adds_level(tmp_path, toy_protein_df):
    df = toy_protein_df.drop(columns=["level"])
    csv_path = tmp_path / "test.csv"
    df.to_csv(csv_path, index=False)
    ds = load_standard_table(str(csv_path), name="test", level="peptide")
    assert ds.level == "peptide"
    assert all(ds.table["level"] == "peptide")

def test_load_standard_table_wrong_level(tmp_standard_csv):
    with pytest.raises(ValueError, match="expected only 'peptide'"):
        load_standard_table(str(tmp_standard_csv), name="test", level="peptide")

def test_load_standard_table_missing_sequence(tmp_path, toy_protein_df):
    df = toy_protein_df.drop(columns=["sequence"])
    csv_path = tmp_path / "test.csv"
    df.to_csv(csv_path, index=False)
    with pytest.raises(ValueError, match="sequence"):
        load_standard_table(str(csv_path), name="test", level="protein")

def test_load_standard_table_group_col_preserved(tmp_standard_csv):
    ds = load_standard_table(str(tmp_standard_csv), name="test", level="protein", group_col="group")
    assert ds.group_col == "group"
