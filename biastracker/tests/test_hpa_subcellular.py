import zipfile

import pandas as pd
import pytest

from biastracker.annotations.hpa import (
    _make_hpa_term_id,
    _normalize_hpa_location_id,
    _parse_hpa_go_pairs,
    _split_hpa_locations,
    load_hpa_subcellular,
    normalize_hpa_subcellular,
    read_hpa_subcellular_tsv,
)


HPA_HEADER = [
    "Gene",
    "Gene name",
    "Reliability",
    "Main location",
    "Additional location",
    "Extracellular location",
    "Enhanced",
    "Supported",
    "Approved",
    "Uncertain",
    "Single-cell variation intensity",
    "Single-cell variation spatial",
    "Cell cycle dependency",
    "GO id",
]


def tiny_hpa_df():
    return pd.DataFrame(
        [
            [
                "ENSG00000000003",
                "TSPAN6",
                "Approved",
                "Cell Junctions;Cytosol",
                "Nucleoli fibrillar center",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "Cell Junctions (GO:0030054);Cytosol (GO:0005829);"
                "Nucleoli fibrillar center (GO:0001650)",
            ],
            [
                "ENSG00000000457",
                "SCYL3",
                "Supported",
                "Cytosol;Golgi apparatus",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "Cytosol (GO:0005829);Golgi apparatus (GO:0005794)",
            ],
        ],
        columns=HPA_HEADER,
    )


def write_tiny_hpa(path):
    tiny_hpa_df().to_csv(path, sep="\t", index=False)


def test_read_hpa_subcellular_tsv_with_exact_header(tmp_path):
    path = tmp_path / "subcellular_location.tsv"
    write_tiny_hpa(path)

    df = read_hpa_subcellular_tsv(path)

    assert list(df.columns) == HPA_HEADER
    assert df.loc[0, "Gene"] == "ENSG00000000003"


def test_split_hpa_locations():
    assert _split_hpa_locations("Cell Junctions;Cytosol") == [
        "Cell Junctions",
        "Cytosol",
    ]
    assert _split_hpa_locations("") == []
    assert _split_hpa_locations(None) == []
    assert _split_hpa_locations(float("nan")) == []


def test_parse_hpa_go_pairs():
    assert _parse_hpa_go_pairs(
        "Cell Junctions (GO:0030054);Cytosol (GO:0005829)"
    ) == {
        "Cell Junctions": "GO:0030054",
        "Cytosol": "GO:0005829",
    }


def test_normalize_hpa_subcellular_uses_ensembl_ids_and_location_categories():
    normalized = normalize_hpa_subcellular(tiny_hpa_df())

    first_gene = normalized[normalized["gene_ensembl_id"] == "ENSG00000000003"]
    assert set(first_gene["primary_id"]) == {"ENSG00000000003"}
    assert set(first_gene["term_name"]) == {
        "Cell Junctions",
        "Cytosol",
        "Nucleoli fibrillar center",
    }
    assert "GO:0030054" in set(first_gene["term_id"])

    category_by_location = dict(zip(first_gene["term_name"], first_gene["category"]))
    assert category_by_location["Cell Junctions"] == "hpa_main_location"
    assert (
        category_by_location["Nucleoli fibrillar center"]
        == "hpa_additional_location"
    )


def test_synthetic_fallback_for_missing_go_pair():
    assert _normalize_hpa_location_id("Rods & Rings") == "HPA_LOC:rods_rings"
    assert (
        _make_hpa_term_id("Nucleoli fibrillar center", {})
        == "HPA_LOC:nucleoli_fibrillar_center"
    )


def test_normalize_hpa_subcellular_maps_one_ensembl_to_many_uniprot():
    normalized = normalize_hpa_subcellular(
        tiny_hpa_df().iloc[[0]],
        map_to_uniprot=True,
        ensembl_to_uniprot={"ENSG00000000003": ["P12345", "Q99999"]},
    )

    assert set(normalized["primary_id"]) == {"P12345", "Q99999"}
    assert len(normalized[normalized["term_name"] == "Cell Junctions"]) == 2


def test_load_hpa_subcellular_requires_mapping_when_offline(tmp_path):
    path = tmp_path / "subcellular_location.tsv"
    write_tiny_hpa(path)

    with pytest.raises(ValueError, match="ensembl_to_uniprot"):
        load_hpa_subcellular(path, map_to_uniprot=True, use_uniprot_api=False)


def test_read_hpa_subcellular_tsv_from_zip(tmp_path):
    tsv_path = tmp_path / "subcellular_location.tsv"
    zip_path = tmp_path / "subcellular_location.tsv.zip"
    write_tiny_hpa(tsv_path)

    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.write(tsv_path, arcname="subcellular_location.tsv")

    df = read_hpa_subcellular_tsv(zip_path)

    assert list(df.columns) == HPA_HEADER
    assert len(df) == 2


def test_load_hpa_subcellular_returns_annotation_set(tmp_path):
    path = tmp_path / "subcellular_location.tsv"
    write_tiny_hpa(path)

    ann = load_hpa_subcellular(path)

    assert ann.name == "hpa_subcellular"
    assert ann.source == "HPA"
    for column in ["primary_id", "term_id", "term_name", "source", "category"]:
        assert column in ann.table.columns
    assert ann.metadata["n_raw_rows"] == 2
    assert ann.metadata["id_namespace"] == "Ensembl"
