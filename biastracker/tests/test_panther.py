import pytest
from biastracker.annotations.panther import load_panther_annotation

def test_panther_loader_inference(tmp_path):
    p = tmp_path / "fake_panther.csv"
    p.write_text("uniprot,go_id,name\nP12345,GO:0001,Some process\nP67890,GO:0002,Other process\n")
    
    ann = load_panther_annotation(p, name="test_panther", panther_type="go_slim")
    assert ann.name == "test_panther"
    assert ann.source == "PANTHER"
    assert "P12345" in ann.ids_for_term("Some process")
    assert len(ann.table) == 2
    assert ann.table["category"].iloc[0] == "ontology"
    assert ann.table["primary_id"].iloc[0] == "P12345"
    assert ann.table["term_id"].iloc[0] == "GO:0001"
    assert ann.table["term_name"].iloc[0] == "Some process"

def test_panther_loader_missing_cols(tmp_path):
    p = tmp_path / "fake_panther2.csv"
    p.write_text("random_id,random_term,random_name\nP12345,GO:0001,Some process\n")
    
    with pytest.raises(ValueError, match="Could not infer or find required columns"):
        load_panther_annotation(p, name="test2", panther_type="family")
        
def test_panther_categories(tmp_path):
    p = tmp_path / "fake.csv"
    p.write_text("uniprot,go_id,name\nP1,G1,N1\n")
    
    ann1 = load_panther_annotation(p, name="t1", panther_type="family")
    assert ann1.table["category"].iloc[0] == "family"
    
    ann2 = load_panther_annotation(p, name="t2", panther_type="reactome")
    assert ann2.table["category"].iloc[0] == "pathway"
    
    ann3 = load_panther_annotation(p, name="t3", panther_type="protein_class")
    assert ann3.table["category"].iloc[0] == "protein_class"
    
    ann4 = load_panther_annotation(p, name="t4", panther_type="unknown_type")
    assert ann4.table["category"].iloc[0] == "unknown"

def test_panther_explicit_cols_and_category(tmp_path):
    p = tmp_path / "fake_panther3.csv"
    p.write_text("random_id,random_term,random_name\nP12345,GO:0001,Some process\n")
    
    ann = load_panther_annotation(
        p, 
        name="test3", 
        panther_type="family",
        id_col="random_id",
        term_id_col="random_term",
        term_name_col="random_name",
        category="custom_category"
    )
    
    assert ann.table["category"].iloc[0] == "custom_category"
    assert ann.table["primary_id"].iloc[0] == "P12345"
    assert ann.table["term_id"].iloc[0] == "GO:0001"
    assert ann.table["term_name"].iloc[0] == "Some process"
