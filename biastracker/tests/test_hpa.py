import pandas as pd
import pytest
from biastracker.annotations.hpa import load_hpa_subcellular

def test_load_hpa_subcellular_inference(tmp_path):
    df = pd.DataFrame({
        "Uniprot": ["P12345", "Q67890"],
        "Subcellular location": ["Nucleus", "Cytosol; Mitochondrion"]
    })
    path = tmp_path / "hpa.csv"
    df.to_csv(path, index=False)
    
    ann = load_hpa_subcellular(path)
    assert ann.name == "hpa_subcellular"
    assert ann.source == "HPA"
    
    table = ann.table
    assert len(table) == 3
    
    p1 = table[table["primary_id"] == "P12345"]
    assert len(p1) == 1
    assert p1.iloc[0]["term_name"] == "Nucleus"
    assert p1.iloc[0]["term_id"] == "HPA:Nucleus"
    assert p1.iloc[0]["category"] == "subcellular_location"
    
    p2 = table[table["primary_id"] == "Q67890"]
    assert len(p2) == 2
    assert set(p2["term_name"]) == {"Cytosol", "Mitochondrion"}
    assert set(p2["term_id"]) == {"HPA:Cytosol", "HPA:Mitochondrion"}

def test_load_hpa_subcellular_explicit(tmp_path):
    df = pd.DataFrame({
        "my_id": ["P1", "P2"],
        "my_loc": ["Cytoplasm, Nucleus", "Golgi apparatus"]
    })
    path = tmp_path / "hpa.tsv"
    df.to_csv(path, sep="\t", index=False)
    
    ann = load_hpa_subcellular(path, id_col="my_id", location_col="my_loc")
    table = ann.table
    assert len(table) == 3
    assert "HPA:Golgi_apparatus" in table["term_id"].values
    assert "Golgi apparatus" in table["term_name"].values

def test_load_hpa_subcellular_missing_cols(tmp_path):
    df = pd.DataFrame({
        "foo": ["A", "B"],
        "bar": ["C", "D"]
    })
    path = tmp_path / "hpa.csv"
    df.to_csv(path, index=False)
    
    with pytest.raises(ValueError, match="Could not infer or find required columns"):
        load_hpa_subcellular(path)
