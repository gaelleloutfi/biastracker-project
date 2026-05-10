from typer.testing import CliRunner
from biastracker.cli import app
import pandas as pd

runner = CliRunner()

def _combined_output(result):
    stderr = result.stderr if result.stderr_bytes is not None else ""
    return (result.stdout or "") + (stderr or "") + (result.output or "")

def test_version():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "BiasTracker version:" in result.stdout

def test_check():
    result = runner.invoke(app, ["check"])
    assert result.exit_code == 0
    assert "Success" in result.stdout

def test_analyze(tmp_path, toy_protein_df):
    input_csv = tmp_path / "input.csv"
    toy_protein_df.to_csv(input_csv, index=False)
    
    out_dir = tmp_path / "out"
    result = runner.invoke(app, [
        "analyze",
        "--input", str(input_csv),
        "--out", str(out_dir)
    ])
    assert result.exit_code == 0
    assert (out_dir / "tables" / "dataset_summary.csv").exists()

def test_compare_groups(tmp_path, toy_protein_df):
    input_csv = tmp_path / "input.csv"
    toy_protein_df.to_csv(input_csv, index=False)
    
    out_dir = tmp_path / "out"
    result = runner.invoke(app, [
        "compare-groups",
        "--input", str(input_csv),
        "--group-col", "group",
        "--group-a", "A",
        "--group-b", "B",
        "--out", str(out_dir)
    ])
    assert result.exit_code == 0
    assert (out_dir / "tables" / "feature_statistics.csv").exists()
    assert (out_dir / "figures" / "violin_length.png").exists()
    assert (out_dir / "figures" / "cdf_length.png").exists()

def test_enrich(tmp_path, toy_protein_df, toy_annotation_df):
    input_csv = tmp_path / "input.csv"
    toy_protein_df.to_csv(input_csv, index=False)
    
    ann_csv = tmp_path / "ann.csv"
    toy_annotation_df.to_csv(ann_csv, index=False)
    
    out_dir = tmp_path / "out"
    result = runner.invoke(app, [
        "enrich",
        "--input", str(input_csv),
        "--group-col", "group",
        "--query-group", "A",
        "--annotations", str(ann_csv),
        "--min-term-size", "1",
        "--out", str(out_dir)
    ])
    assert result.exit_code == 0
    assert (out_dir / "tables" / "enrichment_results.csv").exists()

def test_enrich_calls_run_group_ora_with_keywords(tmp_path, toy_protein_df, toy_annotation_df, monkeypatch):
    input_csv = tmp_path / "input.csv"
    toy_protein_df.to_csv(input_csv, index=False)
    ann_csv = tmp_path / "ann.csv"
    toy_annotation_df.to_csv(ann_csv, index=False)
    out_dir = tmp_path / "out"

    def keyword_only_run_group_ora(*, dataset, group_col, query_group, annotations, min_term_size, **kwargs):
        return pd.DataFrame({
            "term_name": ["term"],
            "p_value": [1.0],
            "fdr": [1.0],
            "odds_ratio": [1.0],
        })

    monkeypatch.setattr("biastracker.cli.run_group_ora", keyword_only_run_group_ora)

    result = runner.invoke(app, [
        "enrich",
        "--input", str(input_csv),
        "--group-col", "group",
        "--query-group", "A",
        "--annotations", str(ann_csv),
        "--min-term-size", "1",
        "--out", str(out_dir),
    ])
    assert result.exit_code == 0

def test_run_valid_config(tmp_yaml_config):
    result = runner.invoke(app, ["run", str(tmp_yaml_config)])
    assert result.exit_code == 0
    assert "Workflow completed successfully" in result.stdout

def test_run_invalid_config(tmp_path):
    config_yaml = tmp_path / "config.yaml"
    config_yaml.write_text("invalid_yaml: [}")
    
    result = runner.invoke(app, ["run", str(config_yaml)])
    assert result.exit_code == 1
    output = _combined_output(result)
    assert "Error loading or validating config" in output

def test_run_missing_keys(tmp_path):
    config_yaml = tmp_path / "config.yaml"
    config_yaml.write_text("datasets: []\n")
    
    result = runner.invoke(app, ["run", str(config_yaml)])
    assert result.exit_code == 1
    output = _combined_output(result)
    assert "missing required field: 'project_name'" in output


def test_fetch_hpa_subcellular_saves_csv(tmp_path, monkeypatch):
    from biastracker.dataset import AnnotationSet

    def fake_download(cache_dir):
        path = tmp_path / "hpa.tsv"
        path.write_text("unused")
        return path

    def fake_loader(path, name, map_to_uniprot, cache_dir, use_uniprot_api):
        assert path == tmp_path / "hpa.tsv"
        assert name == "hpa_subcellular"
        assert map_to_uniprot is True
        assert use_uniprot_api is True
        df = pd.DataFrame(
            {
                "primary_id": ["P12345"],
                "term_id": ["GO:0005634"],
                "term_name": ["Nucleus"],
                "category": ["hpa_main_location"],
            }
        )
        return AnnotationSet(name=name, source="HPA", table=df)

    monkeypatch.setattr("biastracker.cli._download_hpa_subcellular", fake_download)
    monkeypatch.setattr("biastracker.cli.load_hpa_subcellular", fake_loader)

    out = tmp_path / "hpa.csv"
    result = runner.invoke(
        app,
        [
            "fetch-hpa-subcellular",
            "--map-to-uniprot",
            "--out",
            str(out),
            "--cache-dir",
            str(tmp_path / "cache"),
        ],
    )

    assert result.exit_code == 0
    saved = pd.read_csv(out)
    assert saved.loc[0, "primary_id"] == "P12345"
    assert saved.loc[0, "term_name"] == "Nucleus"


def test_fetch_panther_saves_csv(tmp_path, monkeypatch):
    from biastracker.dataset import AnnotationSet

    input_csv = tmp_path / "ids.csv"
    pd.DataFrame({"primary_id": ["P1", "P2"]}).to_csv(input_csv, index=False)

    def fake_loader(ids, name, organism, categories, cache_dir):
        assert ids == ["P1", "P2"]
        assert name == "panther_api"
        assert organism == "9606"
        assert categories == ["go_bp", "go_cc"]
        df = pd.DataFrame(
            {
                "primary_id": ["P1"],
                "term_id": ["GO:1"],
                "term_name": ["process"],
                "category": ["go_bp"],
            }
        )
        return AnnotationSet(name=name, source="PANTHER", table=df)

    monkeypatch.setattr("biastracker.cli.load_panther_api_annotations", fake_loader)

    out = tmp_path / "panther.csv"
    result = runner.invoke(
        app,
        [
            "fetch-panther",
            "--input",
            str(input_csv),
            "--id-col",
            "primary_id",
            "--organism",
            "9606",
            "--categories",
            "go_bp,go_cc",
            "--out",
            str(out),
            "--cache-dir",
            str(tmp_path / "cache"),
        ],
    )

    assert result.exit_code == 0
    saved = pd.read_csv(out)
    assert saved.loc[0, "primary_id"] == "P1"
    assert saved.loc[0, "term_name"] == "process"
