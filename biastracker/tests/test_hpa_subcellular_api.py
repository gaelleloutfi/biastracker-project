import pandas as pd

from biastracker.annotations.hpa import (
    load_hpa_subcellular,
    normalize_hpa_subcellular,
    read_hpa_subcellular_tsv,
)


HPA_COLUMNS = [
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


def tiny_hpa_subcellular_df() -> pd.DataFrame:
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
                "Secreted",
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
        columns=HPA_COLUMNS,
    )


def write_tiny_hpa_subcellular_tsv(path):
    tiny_hpa_subcellular_df().to_csv(path, sep="\t", index=False)


def test_hpa_subcellular_reads_tiny_tsv_with_exact_header(tmp_path):
    path = tmp_path / "subcellular_location.tsv"
    write_tiny_hpa_subcellular_tsv(path)

    df = read_hpa_subcellular_tsv(path)

    assert list(df.columns) == HPA_COLUMNS
    assert list(df["Gene"]) == ["ENSG00000000003", "ENSG00000000457"]


def test_hpa_subcellular_splits_locations_and_parses_go_ids():
    normalized = normalize_hpa_subcellular(tiny_hpa_subcellular_df())

    tspan6 = normalized[normalized["gene_ensembl_id"] == "ENSG00000000003"]
    assert set(tspan6["term_name"]) == {
        "Cell Junctions",
        "Cytosol",
        "Nucleoli fibrillar center",
    }
    assert dict(zip(tspan6["term_name"], tspan6["term_id"])) == {
        "Cell Junctions": "GO:0030054",
        "Cytosol": "GO:0005829",
        "Nucleoli fibrillar center": "GO:0001650",
    }


def test_hpa_subcellular_location_categories_and_fallback_ids():
    normalized = normalize_hpa_subcellular(tiny_hpa_subcellular_df())

    category_by_location = dict(zip(normalized["term_name"], normalized["category"]))
    assert category_by_location["Cell Junctions"] == "hpa_main_location"
    assert category_by_location["Nucleoli fibrillar center"] == "hpa_additional_location"
    assert category_by_location["Secreted"] == "hpa_extracellular_location"

    secreted = normalized[normalized["term_name"] == "Secreted"].iloc[0]
    assert secreted["term_id"] == "HPA_LOC:secreted"


def test_hpa_subcellular_without_uniprot_mapping_uses_ensembl_ids(tmp_path):
    path = tmp_path / "subcellular_location.tsv"
    write_tiny_hpa_subcellular_tsv(path)

    ann = load_hpa_subcellular(path, map_to_uniprot=False)

    assert set(ann.table["primary_id"]) == {"ENSG00000000003", "ENSG00000000457"}
    assert ann.metadata["id_namespace"] == "Ensembl"


def test_hpa_subcellular_with_mocked_uniprot_mapping_expands_ids(tmp_path, monkeypatch):
    path = tmp_path / "subcellular_location.tsv"
    write_tiny_hpa_subcellular_tsv(path)

    def fake_map_ids_to_uniprot(ids, from_ns, to_ns, **kwargs):
        assert ids == ["ENSG00000000003", "ENSG00000000457"]
        assert from_ns == "Ensembl"
        assert to_ns == "UniProtKB"
        return {
            "ENSG00000000003": ["P12345", "Q99999"],
            "ENSG00000000457": ["O12345"],
        }

    monkeypatch.setattr(
        "biastracker.annotations.uniprot_mapping.map_ids_to_uniprot",
        fake_map_ids_to_uniprot,
    )

    ann = load_hpa_subcellular(
        path,
        map_to_uniprot=True,
        use_uniprot_api=True,
        cache_dir=tmp_path / "cache",
    )

    assert set(ann.table["primary_id"]) == {"P12345", "Q99999", "O12345"}
    assert ann.metadata["id_namespace"] == "UniProtKB"
