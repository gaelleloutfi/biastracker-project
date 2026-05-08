import pandas as pd
import pytest
from biastracker.annotations.umap_subcellular import load_umap_subcellular

def test_load_umap_subcellular_inference(tmp_path):
    df = pd.DataFrame({
        "gene": ["GENE1", "GENE2"],
        "compartment": ["Mitochondria", "ER; Golgi"]
    })
    path = tmp_path / "umap.csv"
    df.to_csv(path, index=False)
    
    ann = load_umap_subcellular(path)
    assert ann.name == "czbiohub_subcellular_umap"
    assert ann.source == "CZBIOHUB_SUBCELLULAR_UMAP"
    
    table = ann.table
    assert len(table) == 3
    
    p1 = table[table["primary_id"] == "GENE1"]
    assert len(p1) == 1
    assert p1.iloc[0]["term_name"] == "Mitochondria"
    assert p1.iloc[0]["term_id"] == "UMAP:Mitochondria"
    assert p1.iloc[0]["category"] == "subcellular_location"
    
    p2 = table[table["primary_id"] == "GENE2"]
    assert len(p2) == 2
    assert set(p2["term_name"]) == {"ER", "Golgi"}
    assert set(p2["term_id"]) == {"UMAP:ER", "UMAP:Golgi"}

def test_load_umap_subcellular_explicit(tmp_path):
    df = pd.DataFrame({
        "my_id": ["P1", "P2"],
        "my_loc": ["Cytosol, Nucleus", "Lysosome"]
    })
    path = tmp_path / "umap.tsv"
    df.to_csv(path, sep="\t", index=False)
    
    ann = load_umap_subcellular(path, id_col="my_id", location_col="my_loc")
    table = ann.table
    assert len(table) == 3
    assert "UMAP:Cytosol" in table["term_id"].values
    assert "Cytosol" in table["term_name"].values

def test_load_umap_subcellular_missing_cols(tmp_path):
    df = pd.DataFrame({
        "foo": ["A", "B"],
        "bar": ["C", "D"]
    })
    path = tmp_path / "umap.csv"
    df.to_csv(path, index=False)
    
    with pytest.raises(ValueError, match="Could not infer or find required columns"):
        load_umap_subcellular(path)
