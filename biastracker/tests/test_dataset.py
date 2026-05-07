import pytest
import sys
from biastracker.dataset import check_protperties_available, BiasDataset, AnnotationSet, AnnotationSet
import pandas as pd
import numpy as np

def test_check_protperties_available_success():
    """Verify that the function imports successfully if protperties is installed."""
    try:
        check_protperties_available()
    except ImportError:
        pytest.skip("protperties is not currently installed in the environment.")

def test_check_protperties_available_failure(monkeypatch):
    """Monkeypatch sys.modules to simulate missing protperties."""
    with monkeypatch.context() as m:
        m.setitem(sys.modules, 'protperties', None)
        with pytest.raises(ImportError, match="Failed to import required functions from protperties"):
            check_protperties_available()


def test_biasdataset_valid():
    df = pd.DataFrame({
        "primary_id": ["A", "B", "C"],
        "level": ["protein", "protein", "protein"],
        "sequence": ["SEQ1", "SEQ2", "SEQ3"],
        "val1": [1.0, 2.0, 3.0]
    })
    ds = BiasDataset(name="test", table=df, level="protein")
    assert ds.name == "test"
    assert ds.level == "protein"
    assert "A" in ds.ids()

def test_biasdataset_missing_sequence():
    df = pd.DataFrame({
        "primary_id": ["A", "B", "C"],
        "level": ["protein", "protein", "protein"]
    })
    with pytest.raises(ValueError, match="sequence"):
        BiasDataset(name="test", table=df, level="protein")

def test_biasdataset_missing_group_col():
    df = pd.DataFrame({
        "primary_id": ["A", "B", "C"],
        "level": ["protein", "protein", "protein"],
        "sequence": ["SEQ1", "SEQ2", "SEQ3"]
    })
    with pytest.raises(ValueError, match="must exist"):
        BiasDataset(name="test", table=df, level="protein", group_col="missing_group")

def test_biasdataset_available_features():
    df = pd.DataFrame({
        "primary_id": ["A", "B", "C"],
        "level": ["protein", "protein", "protein"],
        "sequence": ["SEQ1", "SEQ2", "SEQ3"],
        "num1": [1.0, 2.0, 3.0],
        "num2": [4.0, 5.0, 6.0],
        "str1": ["x", "y", "z"]
    })
    ds = BiasDataset(name="test", table=df, level="protein")
    feats = ds.available_features()
    assert "num1" in feats
    assert "num2" in feats
    assert "str1" not in feats
    
    feats_subset = ds.available_features(["num1", "str1", "missing"])
    assert feats_subset == ["num1"]

def test_annotationset_fills_missing():
    df = pd.DataFrame({
        "primary_id": ["A", "B"],
        "term_name": ["Term1", "Term2"]
    })
    ann = AnnotationSet(name="test", source="test", table=df)
    assert "term_id" in ann.table.columns
    assert list(ann.table["term_id"]) == ["Term1", "Term2"]
    assert "category" in ann.table.columns
    assert list(ann.table["category"]) == ["unknown", "unknown"]

def test_annotationset_drops_invalid():
    df = pd.DataFrame({
        "primary_id": ["A", None, "C", "D"],
        "term_name": ["Term1", "Term2", None, np.nan]
    })
    ann = AnnotationSet(name="test", source="test", table=df)
    assert len(ann.table) == 1
    assert ann.table.iloc[0]["primary_id"] == "A"
    assert ann.table.iloc[0]["term_name"] == "Term1"

from biastracker.dataset import load_standard_table, load_manual_table_via_protperties

def test_load_standard_table(tmp_path):
    csv_path = tmp_path / "test.csv"
    csv_path.write_text("primary_id,sequence,val1\nA,SEQ1,1.0\nB,SEQ2,2.0")
    
    ds = load_standard_table(str(csv_path), name="test", level="protein")
    assert ds.name == "test"
    assert ds.level == "protein"
    assert "level" in ds.table.columns
    assert ds.table["level"].iloc[0] == "protein"
    assert "A" in ds.ids()

def test_load_standard_table_wrong_level(tmp_path):
    csv_path = tmp_path / "test.csv"
    csv_path.write_text("primary_id,sequence,level\nA,SEQ1,peptide\nB,SEQ2,peptide")
    
    with pytest.raises(ValueError, match="expected only 'protein'"):
        load_standard_table(str(csv_path), name="test", level="protein")

def test_load_manual_table_via_protperties(tmp_path):
    try:
        import protperties
    except ImportError:
        pytest.skip("protperties not installed")
        
    csv_path = tmp_path / "test.csv"
    csv_path.write_text("primary_id,sequence,val1\nA,SEQ1,1.0\nB,SEQ2,2.0")
    
    ds = load_manual_table_via_protperties(str(csv_path), name="test_manual", level="protein", group_col=None)
    assert ds.name == "test_manual"
    assert ds.level == "protein"
    assert ds.source_type == "manual"
