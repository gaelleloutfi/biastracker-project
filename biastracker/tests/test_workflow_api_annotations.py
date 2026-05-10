import json
from pathlib import Path

import pandas as pd
import pytest

from biastracker.dataset import AnnotationSet
from biastracker.workflow import run_workflow


def base_config(tmp_path):
    data = tmp_path / "dataset.csv"
    pd.DataFrame(
        {
            "primary_id": ["P04637", "P00533", "Q00987"],
            "sequence": ["AAA", "BBB", "CCC"],
            "level": ["protein", "protein", "protein"],
        }
    ).to_csv(data, index=False)

    return {
        "project_name": "api_annotations",
        "output": {"directory": str(tmp_path / "results")},
        "datasets": [
            {
                "name": "proteins",
                "type": "standard_csv",
                "path": str(data),
                "level": "protein",
            }
        ],
        "annotations": [],
        "analysis": {},
    }


def write_hpa_tsv(path):
    columns = [
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
    pd.DataFrame(
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
            ]
        ],
        columns=columns,
    ).to_csv(path, sep="\t", index=False)


def test_workflow_panther_api_annotation_uses_dataset_ids_and_exports(tmp_path, monkeypatch):
    config = base_config(tmp_path)
    config["annotations"] = [
        {
            "name": "panther_api",
            "type": "panther_api",
            "dataset": "proteins",
            "categories": ["go_bp"],
            "cache_dir": str(tmp_path / "panther_cache"),
        }
    ]

    def fake_load_panther_api_annotations(
        ids,
        name,
        organism,
        categories,
        batch_size,
        cache_dir,
        use_cache,
    ):
        assert ids == ["P04637", "P00533", "Q00987"]
        assert name == "panther_api"
        assert categories == ["go_bp"]
        return AnnotationSet(
            name=name,
            source="PANTHER",
            table=pd.DataFrame(
                {
                    "primary_id": ["P04637"],
                    "term_id": ["GO:0006915"],
                    "term_name": ["apoptotic process"],
                    "category": ["go_bp"],
                }
            ),
            metadata={"source": "PANTHER_API"},
        )

    monkeypatch.setattr(
        "biastracker.annotations.panther.load_panther_api_annotations",
        fake_load_panther_api_annotations,
    )

    run_workflow(config)

    annotations_dir = Path(config["output"]["directory"]) / "annotations"
    exported = pd.read_csv(annotations_dir / "panther_api.csv")
    metadata = json.loads((annotations_dir / "panther_api.metadata.json").read_text())

    assert exported.to_dict("records") == [
        {
            "primary_id": "P04637",
            "term_id": "GO:0006915",
            "term_name": "apoptotic process",
            "category": "go_bp",
        }
    ]
    assert metadata["name"] == "panther_api"
    assert metadata["source"] == "PANTHER"
    assert metadata["n_rows"] == 1
    assert metadata["n_unique_ids"] == 1
    assert metadata["n_terms"] == 1
    assert metadata["metadata"] == {"source": "PANTHER_API"}


def test_workflow_hpa_subcellular_annotation_loads_local_tsv_and_exports(tmp_path):
    config = base_config(tmp_path)
    hpa_path = tmp_path / "hpa.tsv"
    write_hpa_tsv(hpa_path)
    config["annotations"] = [
        {
            "name": "hpa",
            "type": "hpa_subcellular",
            "path": str(hpa_path),
        }
    ]

    run_workflow(config)

    annotations_dir = Path(config["output"]["directory"]) / "annotations"
    exported = pd.read_csv(annotations_dir / "hpa.csv")
    metadata = json.loads((annotations_dir / "hpa.metadata.json").read_text())

    assert set(exported["term_name"]) == {
        "Cell Junctions",
        "Cytosol",
        "Nucleoli fibrillar center",
    }
    assert set(exported["category"]) == {
        "hpa_main_location",
        "hpa_additional_location",
    }
    assert metadata["n_rows"] == 3
    assert metadata["n_unique_ids"] == 1
    assert metadata["n_terms"] == 3


def test_workflow_panther_api_missing_dataset_raises_value_error(tmp_path):
    config = base_config(tmp_path)
    config["annotations"] = [
        {
            "name": "panther_api",
            "type": "panther_api",
            "dataset": "missing",
        }
    ]

    with pytest.raises(ValueError, match="references dataset 'missing'"):
        run_workflow(config)
