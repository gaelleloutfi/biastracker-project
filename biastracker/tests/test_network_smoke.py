import os

import pytest


pytestmark = pytest.mark.skipif(
    os.environ.get("BIASTRACKER_RUN_NETWORK_TESTS") != "1",
    reason="set BIASTRACKER_RUN_NETWORK_TESTS=1 to run live network smoke tests",
)


def test_panther_live_smoke_returns_expected_columns(tmp_path):
    from biastracker.annotations.panther import PANTHER_OUTPUT_COLUMNS, fetch_panther_annotations

    df = fetch_panther_annotations(
        ["P04637", "P00533", "Q00987"],
        organism=9606,
        cache_dir=tmp_path / "panther",
        use_cache=False,
    )

    assert list(df.columns) == PANTHER_OUTPUT_COLUMNS


def test_hpa_live_smoke_downloads_official_zip_and_emits_rows(tmp_path):
    from biastracker.annotations._http import request_with_retries
    from biastracker.workflow import HPA_SUBCELLULAR_DEFAULT_URL
    from biastracker.annotations.hpa import load_hpa_subcellular

    response = request_with_retries(
        "GET",
        HPA_SUBCELLULAR_DEFAULT_URL,
        timeout=120,
    )
    archive_path = tmp_path / "subcellular_location.tsv.zip"
    archive_path.write_bytes(response.content)

    ann = load_hpa_subcellular(archive_path)

    assert not ann.table.empty
    assert {"primary_id", "term_id", "term_name", "category"}.issubset(ann.table.columns)


def test_uniprot_live_smoke_maps_ensembl_to_uniprot(tmp_path):
    from biastracker.annotations.uniprot_mapping import map_ids_to_uniprot

    mapping = map_ids_to_uniprot(
        ["ENSG00000141510"],
        from_ns="Ensembl",
        to_ns="UniProtKB",
        cache_dir=tmp_path / "uniprot",
        use_cache=False,
    )

    assert mapping["ENSG00000141510"]
