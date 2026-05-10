import pandas as pd
import pytest

from biastracker.annotations.hpa import load_hpa_subcellular


def test_load_hpa_subcellular_local_hpa_format(tmp_path):
    df = pd.DataFrame(
        {
            "Gene": ["ENSG00000000003"],
            "Gene name": ["TSPAN6"],
            "Reliability": ["Approved"],
            "Main location": ["Cell Junctions;Cytosol"],
            "Additional location": ["Nucleoli fibrillar center"],
            "GO id": [
                "Cell Junctions (GO:0030054);Cytosol (GO:0005829);"
                "Nucleoli fibrillar center (GO:0001650)"
            ],
        }
    )
    path = tmp_path / "hpa.tsv"
    df.to_csv(path, sep="\t", index=False)

    ann = load_hpa_subcellular(path)

    assert ann.name == "hpa_subcellular"
    assert ann.source == "HPA"
    assert set(ann.table["primary_id"]) == {"ENSG00000000003"}
    assert set(ann.table["term_name"]) == {
        "Cell Junctions",
        "Cytosol",
        "Nucleoli fibrillar center",
    }
    assert "GO:0030054" in set(ann.table["term_id"])


def test_load_hpa_subcellular_maps_to_uniprot_with_provided_mapping(tmp_path):
    df = pd.DataFrame(
        {
            "Gene": ["ENSG00000000003"],
            "Gene name": ["TSPAN6"],
            "Main location": ["Cytosol"],
            "GO id": ["Cytosol (GO:0005829)"],
        }
    )
    path = tmp_path / "hpa.tsv"
    df.to_csv(path, sep="\t", index=False)

    ann = load_hpa_subcellular(
        path,
        map_to_uniprot=True,
        ensembl_to_uniprot={"ENSG00000000003": ["P12345"]},
    )

    assert set(ann.table["primary_id"]) == {"P12345"}
    assert ann.metadata["id_namespace"] == "UniProtKB"


def test_load_hpa_subcellular_missing_cols(tmp_path):
    df = pd.DataFrame({"foo": ["A"], "bar": ["B"]})
    path = tmp_path / "hpa.tsv"
    df.to_csv(path, sep="\t", index=False)

    with pytest.raises(ValueError, match="missing columns"):
        load_hpa_subcellular(path)
