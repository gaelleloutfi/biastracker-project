import pytest
import pandas as pd
from biastracker.dataset import BiasDataset, AnnotationSet
import yaml

@pytest.fixture
def toy_protein_df():
    return pd.DataFrame({
        "primary_id": ["P1", "P2", "P3", "P4", "P5", "P6"],
        "sequence": ["MAG", "MTS", "MKR", "MLA", "MVR", "MQT"],
        "level": ["protein"] * 6,
        "group": ["A", "A", "A", "B", "B", "B"],
        "length": [10, 15, 20, 12, 18, 22],
        "mw": [1000, 1500, 2000, 1200, 1800, 2200],
        "pi": [5.5, 6.0, 5.8, 8.5, 9.0, 8.8]
    })

@pytest.fixture
def toy_peptide_df():
    return pd.DataFrame({
        "primary_id": ["P1", "P1", "P2"],
        "sequence": ["MAG", "AGR", "MTS"],
        "level": ["peptide"] * 3,
        "group": ["A", "A", "B"],
        "intensity": [100, 150, 200]
    })

@pytest.fixture
def toy_bias_dataset(toy_protein_df):
    return BiasDataset(
        name="toy_proteins",
        table=toy_protein_df,
        level="protein",
        source_type="standard",
        id_col="primary_id",
        group_col="group"
    )

@pytest.fixture
def toy_annotation_df():
    return pd.DataFrame({
        "primary_id": ["P1", "P2", "P3", "P4", "P5", "P6"],
        "term_id": ["GO:1", "GO:1", "GO:1", "GO:2", "GO:2", "GO:2"],
        "term_name": ["membrane", "membrane", "membrane", "nucleus", "nucleus", "nucleus"],
        "category": ["subcellular"] * 6
    })

@pytest.fixture
def toy_annotation_set(toy_annotation_df):
    return AnnotationSet(
        name="toy_annotations",
        source="GO",
        table=toy_annotation_df,
        id_col="primary_id",
        term_col="term_name",
        term_id_col="term_id",
        category_col="category"
    )

@pytest.fixture
def tmp_standard_csv(tmp_path, toy_protein_df):
    path = tmp_path / "toy_proteins.csv"
    toy_protein_df.to_csv(path, index=False)
    return path

@pytest.fixture
def tmp_annotation_csv(tmp_path, toy_annotation_df):
    path = tmp_path / "toy_annotations.csv"
    toy_annotation_df.to_csv(path, index=False)
    return path

@pytest.fixture
def tmp_yaml_config(tmp_path, tmp_standard_csv, tmp_annotation_csv):
    config = {
        "project_name": "test_project",
        "output": {"directory": str(tmp_path / "results")},
        "datasets": [
            {
                "name": "toy_data",
                "type": "standard_csv",
                "path": str(tmp_standard_csv),
                "level": "protein",
                "group_col": "group"
            }
        ],
        "annotations": [
            {
                "name": "toy_go",
                "type": "long",
                "path": str(tmp_annotation_csv),
                "source": "GO"
            }
        ],
        "analysis": {
            "summary": True,
            "comparisons": [
                {
                    "dataset": "toy_data",
                    "group_col": "group",
                    "group_a": "A",
                    "group_b": "B",
                    "features": ["length", "mw"]
                }
            ],
            "enrichment": [
                {
                    "dataset": "toy_data",
                    "group_col": "group",
                    "query_group": "A",
                    "annotation": "toy_go",
                    "min_term_size": 2
                }
            ]
        }
    }
    path = tmp_path / "config.yaml"
    with open(path, "w") as f:
        yaml.dump(config, f)
    return path
