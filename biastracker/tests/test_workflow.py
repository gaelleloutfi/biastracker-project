import pytest
import json
from pathlib import Path
from typer.testing import CliRunner
from biastracker.cli import app
from biastracker.dataset import BiasDataset
from biastracker.workflow import run_workflow
import yaml
import matplotlib
import pandas as pd
matplotlib.use('Agg')

runner = CliRunner()

def test_full_workflow(tmp_yaml_config):
    # tmp_yaml_config sets up the environment and returns the path to the config
    result = runner.invoke(app, ["run", str(tmp_yaml_config)])
    
    assert result.exit_code == 0
    
    # Check outputs
    out_dir = tmp_yaml_config.parent / "results"
    
    tables_dir = out_dir / "tables"
    figures_dir = out_dir / "figures"
    
    assert tables_dir.exists()
    assert figures_dir.exists()
    
    # At least one dataset summary
    assert list(tables_dir.glob("dataset_summary*.csv"))
    
    # At least one feature statistics
    assert list(tables_dir.glob("feature_statistics*.csv"))
    
    # At least one enrichment CSV
    assert list(tables_dir.glob("enrichment*.csv"))
    
    # At least one violin plot
    assert list(figures_dir.glob("violin*.png"))
    
    # At least one CDF plot
    assert list(figures_dir.glob("cdf*.png"))
    
    # Enrichment dotplot
    assert list(figures_dir.glob("enrichment_dotplot*.png"))
    assert (out_dir / "config_used.yaml").exists()


def test_example_config_exists_and_valid():
    config_path = Path("examples/ghost_proteome_config.yaml")
    if not config_path.exists():
        pytest.skip("Example config not found")
        
    # Check if valid YAML
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    assert isinstance(config, dict)
    assert "project_name" in config

def test_example_config_run():
    config_path = Path("examples/ghost_proteome_config.yaml")
    data_path = Path("examples/data/toy_proteins.csv")
    ann_path = Path("examples/annotations/toy_terms.csv")
    
    if not (config_path.exists() and data_path.exists() and ann_path.exists()):
        pytest.skip("Example data files not found")
        
    result = runner.invoke(app, ["run", str(config_path)])
    assert result.exit_code == 0

def test_missing_dataset_in_config_raises(tmp_yaml_config):
    with open(tmp_yaml_config, "r") as f:
        config = yaml.safe_load(f)
    config["analysis"]["comparisons"][0]["dataset"] = "missing_ds"

    with pytest.raises(ValueError, match="unknown dataset 'missing_ds'"):
        run_workflow(config, config_path=tmp_yaml_config)

def test_missing_annotation_in_config_raises(tmp_yaml_config):
    with open(tmp_yaml_config, "r") as f:
        config = yaml.safe_load(f)
    config["analysis"]["enrichment"][0]["annotation"] = "missing_ann"

    with pytest.raises(ValueError, match="unknown annotation 'missing_ann'"):
        run_workflow(config, config_path=tmp_yaml_config)

def _base_config(tmp_path):
    data = tmp_path / "data.csv"
    pd.DataFrame({
        "primary_id": ["P1", "P2"],
        "sequence": ["AA", "BB"],
        "level": ["protein", "protein"],
    }).to_csv(data, index=False)
    return {
        "project_name": "test",
        "output": {"directory": str(tmp_path / "out")},
        "datasets": [{"name": "ds", "type": "standard_csv", "path": str(data), "level": "protein"}],
        "annotations": [],
        "analysis": {},
    }

def test_hpa_umap_and_gmt_config_loading(tmp_path):
    config = _base_config(tmp_path)
    hpa = tmp_path / "hpa.tsv"
    hpa.write_text(
        "Gene\tGene name\tReliability\tMain location\tAdditional location\t"
        "Extracellular location\tGO id\n"
        "ENSG1\tGENE1\tApproved\tNucleus\t\t\tNucleus (GO:0005634)\n"
    )
    umap = tmp_path / "umap.tsv"
    umap.write_text("primary_id\torganelle\nP1\tMitochondria\n")
    gmt = tmp_path / "terms.gmt"
    gmt.write_text("TermA\tdescription\tP1\tP2\n")
    config["annotations"] = [
        {"name": "hpa", "type": "hpa_subcellular", "path": str(hpa)},
        {"name": "umap", "type": "umap_subcellular", "path": str(umap)},
        {"name": "gmt", "type": "gmt", "path": str(gmt)},
    ]

    run_workflow(config)

    annotations_dir = Path(config["output"]["directory"]) / "annotations"
    assert (annotations_dir / "hpa.csv").exists()
    assert (annotations_dir / "hpa.metadata.json").exists()
    assert (annotations_dir / "umap.csv").exists()
    assert (annotations_dir / "gmt.csv").exists()

    hpa_metadata = json.loads((annotations_dir / "hpa.metadata.json").read_text())
    assert hpa_metadata["name"] == "hpa"
    assert hpa_metadata["n_rows"] == 1
    assert hpa_metadata["n_unique_ids"] == 1
    assert hpa_metadata["n_terms"] == 1

def test_workflow_loads_tiny_local_hpa_subcellular_tsv(tmp_path):
    config = _base_config(tmp_path)
    hpa = tmp_path / "hpa.tsv"
    hpa.write_text(
        "Gene\tGene name\tReliability\tMain location\tAdditional location\t"
        "Extracellular location\tGO id\n"
        "ENSG00000000003\tTSPAN6\tApproved\tCell Junctions;Cytosol\t"
        "Nucleoli fibrillar center\t\t"
        "Cell Junctions (GO:0030054);Cytosol (GO:0005829);"
        "Nucleoli fibrillar center (GO:0001650)\n"
    )
    config["annotations"] = [
        {"name": "hpa", "type": "hpa_subcellular", "path": str(hpa)}
    ]

    run_workflow(config)

def test_workflow_hpa_subcellular_defaults_to_download_url(tmp_path):
    config = _base_config(tmp_path)
    config["annotations"] = [{"name": "hpa", "type": "hpa_subcellular"}]

    import io
    import zipfile

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(
            "subcellular_location.tsv",
            "Gene\tGene name\tReliability\tMain location\tAdditional location\t"
            "Extracellular location\tGO id\n"
            "ENSG1\tGENE1\tApproved\tNucleus\t\t\tNucleus (GO:0005634)\n",
        )

    class DummyResponse:
        content = buffer.getvalue()

    from biastracker import workflow

    def fake_request(method, url, **kwargs):
        assert method == "GET"
        assert url == workflow.HPA_SUBCELLULAR_DEFAULT_URL
        return DummyResponse()

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr("biastracker.workflow.request_with_retries", fake_request)
    try:
        run_workflow(config)
    finally:
        monkeypatch.undo()


def test_workflow_hpa_subcellular_downloads_url(tmp_path, monkeypatch):
    config = _base_config(tmp_path)
    url = "https://example.test/hpa.tsv"
    config["annotations"] = [
        {
            "name": "hpa",
            "type": "hpa_subcellular",
            "url": url,
        }
    ]

    class DummyResponse:
        text = (
            "Gene\tGene name\tReliability\tMain location\tAdditional location\t"
            "Extracellular location\tGO id\n"
            "ENSG1\tGENE1\tApproved\tNucleus\t\t\tNucleus (GO:0005634)\n"
        )

    def fake_request(method, request_url, **kwargs):
        assert method == "GET"
        assert request_url == url
        return DummyResponse()

    monkeypatch.setattr("biastracker.workflow.request_with_retries", fake_request)
    run_workflow(config)


def test_workflow_hpa_subcellular_with_mocked_uniprot_mapping(tmp_path, monkeypatch):
    config = _base_config(tmp_path)
    hpa = tmp_path / "hpa.tsv"
    hpa.write_text(
        "Gene\tGene name\tReliability\tMain location\tAdditional location\t"
        "Extracellular location\tGO id\n"
        "ENSG1\tGENE1\tApproved\tNucleus\t\t\tNucleus (GO:0005634)\n"
    )
    config["annotations"] = [
        {
            "name": "hpa",
            "type": "hpa_subcellular",
            "path": str(hpa),
            "map_to_uniprot": True,
            "from_ns": "Ensembl",
            "to_ns": "UniProtKB",
            "cache_dir": str(tmp_path / "cache"),
        }
    ]

    def fake_mapping(ids, from_ns, to_ns, **kwargs):
        assert ids == ["ENSG1"]
        assert from_ns == "Ensembl"
        assert to_ns == "UniProtKB"
        return {"ENSG1": ["P12345"]}

    monkeypatch.setattr(
        "biastracker.annotations.uniprot_mapping.map_ids_to_uniprot",
        fake_mapping,
    )
    run_workflow(config)


def test_workflow_panther_api_uses_loaded_dataset_ids(tmp_path, monkeypatch):
    config = _base_config(tmp_path)
    config["annotations"] = [
        {
            "name": "panther_api",
            "type": "panther_api",
            "dataset": "ds",
            "id_col": "primary_id",
            "organism": 9606,
            "categories": ["go_bp"],
            "batch_size": 500,
            "cache_dir": str(tmp_path / "panther_cache"),
        }
    ]

    def fake_loader(ids, name, organism, categories, batch_size, cache_dir, use_cache):
        from biastracker.dataset import AnnotationSet

        assert ids == ["P1", "P2"]
        assert name == "panther_api"
        assert organism == 9606
        assert categories == ["go_bp"]
        assert batch_size == 500
        df = pd.DataFrame(
            {
                "primary_id": ["P1"],
                "term_id": ["GO:1"],
                "term_name": ["process"],
                "category": ["go_bp"],
            }
        )
        return AnnotationSet(name=name, source="PANTHER", table=df)

    monkeypatch.setattr(
        "biastracker.annotations.panther.load_panther_api_annotations",
        fake_loader,
    )
    run_workflow(config)


def test_workflow_panther_api_missing_dataset_raises(tmp_path):
    config = _base_config(tmp_path)
    config["annotations"] = [
        {"name": "panther_api", "type": "panther_api", "dataset": "missing"}
    ]

    with pytest.raises(ValueError, match="references dataset 'missing'"):
        run_workflow(config)


def test_workflow_panther_api_missing_id_col_raises(tmp_path):
    config = _base_config(tmp_path)
    config["annotations"] = [
        {
            "name": "panther_api",
            "type": "panther_api",
            "dataset": "ds",
            "id_col": "missing_col",
        }
    ]

    with pytest.raises(ValueError, match="requires id_col 'missing_col'"):
        run_workflow(config)

def test_protproperties_dataset_types_are_dispatched(tmp_path, monkeypatch):
    calls = []

    def fake_loader(path, name, **kwargs):
        calls.append(name)
        df = pd.DataFrame({
            "primary_id": ["P1"],
            "sequence": ["AA"],
            "level": [kwargs.pop("level", "protein")],
        })
        return BiasDataset(name=name, table=df, level=df["level"].iloc[0])

    monkeypatch.setattr("biastracker.workflow.load_manual_table_via_protperties", lambda path, name, level, **kwargs: fake_loader(path, name, level=level))
    monkeypatch.setattr("biastracker.workflow.load_diann_report", fake_loader)
    monkeypatch.setattr("biastracker.workflow.load_maxquant_evidence", fake_loader)
    monkeypatch.setattr("biastracker.workflow.load_maxquant_proteingroups", fake_loader)
    monkeypatch.setattr("biastracker.workflow.load_diann_pg_matrix", fake_loader)

    config = {
        "project_name": "test",
        "output": {"directory": str(tmp_path / "out")},
        "datasets": [
            {"name": "manual", "type": "manual", "path": "unused", "level": "protein"},
            {"name": "diann", "type": "diann_report", "path": "unused"},
            {"name": "mq_ev", "type": "maxquant_evidence", "path": "unused"},
            {"name": "mq_pg", "type": "maxquant_proteingroups", "path": "unused"},
            {"name": "pg", "type": "diann_pg_matrix", "path": "unused"},
        ],
        "annotations": [],
        "analysis": {},
    }

    run_workflow(config)
    assert calls == ["manual", "diann", "mq_ev", "mq_pg", "pg"]
