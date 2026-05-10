import typer
from pathlib import Path
from typing import Optional

import pandas as pd

from biastracker import __version__
from biastracker.annotations._http import request_with_retries
from biastracker.dataset import check_protperties_available, load_standard_table
from biastracker.annotations.hpa import load_hpa_subcellular
from biastracker.annotations.panther import load_panther_api_annotations
from biastracker.reports import (
    save_dataset_summary,
    save_group_comparison_results,
    save_enrichment_results,
    prepare_output_dirs,
)
from biastracker.analysis.compare import compare_groups
from biastracker.analysis.enrichment import run_group_ora
from biastracker.annotations.custom import load_long_annotation_table
from biastracker.plots import plot_violin, plot_cdf, plot_enrichment_dotplot

app = typer.Typer(help="BiasTracker CLI")

HPA_SUBCELLULAR_DEFAULT_URL = (
    "https://www.proteinatlas.org/download/tsv/subcellular_location.tsv.zip"
)

@app.command()
def version():
    """Print the installed BiasTracker version."""
    typer.echo(f"BiasTracker version: {__version__}")

@app.command()
def check():
    """Check if the protperties package is correctly installed and accessible."""
    try:
        check_protperties_available()
        typer.echo("Success: protperties is correctly installed and accessible.")
    except ImportError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)

@app.command()
def analyze(
    input: str = typer.Option(..., "--input", help="Path to input data CSV"),
    name: str = typer.Option("dataset", "--name", help="Dataset name"),
    level: str = typer.Option("protein", "--level", help="Dataset level (protein or peptide)"),
    out: str = typer.Option(..., "--out", help="Output directory"),
):
    """Runs a basic single-dataset summary."""
    dataset = load_standard_table(input, name=name, level=level)
    save_dataset_summary(dataset, output_dir=out)
    typer.echo(f"Dataset summary saved to {out}")

@app.command("compare-groups")
def compare_groups_cmd(
    input: str = typer.Option(..., "--input", help="Path to input data CSV"),
    group_col: str = typer.Option(..., "--group-col", help="Column name containing the groups"),
    group_a: str = typer.Option(..., "--group-a", help="First group name"),
    group_b: str = typer.Option(..., "--group-b", help="Second group name"),
    out: str = typer.Option(..., "--out", help="Output directory"),
    name: str = typer.Option("dataset", "--name", help="Dataset name"),
    level: str = typer.Option("protein", "--level", help="Dataset level (protein or peptide)"),
    features: Optional[str] = typer.Option(None, "--features", help="Comma-separated list of features to compare"),
):
    """Compares two groups inside one dataset, for example PE1 vs ghost."""
    dataset = load_standard_table(input, name=name, level=level)
    
    feature_list = None
    if features:
        feature_list = [f.strip() for f in features.split(",")]
        
    res = compare_groups(dataset, group_col, group_a, group_b, features=feature_list)
    save_group_comparison_results(res, out)
    
    dirs = prepare_output_dirs(out)
    
    for feat in res["feature"].unique():
        safe_feat = str(feat).replace("/", "_").replace("\\", "_")
        v_path = dirs["figures"] / f"violin_{safe_feat}.png"
        c_path = dirs["figures"] / f"cdf_{safe_feat}.png"
        
        try:
            plot_violin(dataset, feature=feat, group_col=group_col, output_path=v_path)
        except Exception as e:
            typer.echo(f"Warning: skipped violin plot for '{feat}': {e}", err=True)
            
        try:
            plot_cdf(dataset, feature=feat, group_col=group_col, output_path=c_path)
        except Exception as e:
            typer.echo(f"Warning: skipped CDF plot for '{feat}': {e}", err=True)
            
    typer.echo(f"Group comparison completed. Results saved to {out}")

@app.command()
def enrich(
    input: str = typer.Option(..., "--input", help="Path to input data CSV"),
    group_col: str = typer.Option(..., "--group-col", help="Column name containing the groups"),
    query_group: str = typer.Option(..., "--query-group", help="The query group name"),
    annotations: str = typer.Option(..., "--annotations", help="Path to annotations CSV"),
    out: str = typer.Option(..., "--out", help="Output directory"),
    name: str = typer.Option("dataset", "--name", help="Dataset name"),
    level: str = typer.Option("protein", "--level", help="Dataset level (protein or peptide)"),
    min_term_size: int = typer.Option(3, "--min-term-size", help="Minimum term size to include"),
):
    """Runs ORA enrichment for one group against the dataset background."""
    dataset = load_standard_table(input, name=name, level=level)
    ann_set = load_long_annotation_table(annotations, name="custom")
    
    res = run_group_ora(
        dataset=dataset,
        group_col=group_col,
        query_group=query_group,
        annotations=ann_set,
        min_term_size=min_term_size,
    )
    save_enrichment_results(res, out)
    
    if not res.empty:
        dirs = prepare_output_dirs(out)
        try:
            plot_enrichment_dotplot(res, output_path=dirs["figures"] / "enrichment_dotplot.png")
        except Exception as e:
            typer.echo(f"Warning: skipped enrichment dotplot: {e}", err=True)
            
    typer.echo(f"Enrichment analysis completed. Results saved to {out}")

@app.command("fetch-hpa-subcellular")
def fetch_hpa_subcellular_cmd(
    out: str = typer.Option(..., "--out", help="Path to output normalized CSV"),
    map_to_uniprot: bool = typer.Option(
        False,
        "--map-to-uniprot/--no-map-to-uniprot",
        help="Map HPA Ensembl gene IDs to UniProt accessions",
    ),
    cache_dir: str = typer.Option(
        ".cache/biastracker/hpa",
        "--cache-dir",
        help="Directory for downloaded HPA and mapping cache files",
    ),
):
    """Download/parse HPA subcellular annotations and save normalized CSV."""
    try:
        hpa_path = _download_hpa_subcellular(cache_dir)
        ann_set = load_hpa_subcellular(
            path=hpa_path,
            name="hpa_subcellular",
            map_to_uniprot=map_to_uniprot,
            cache_dir=cache_dir,
            use_uniprot_api=map_to_uniprot,
        )
        _write_annotation_csv(ann_set.table, out)
    except Exception as e:
        typer.echo(f"Failed to fetch HPA subcellular annotations: {e}", err=True)
        raise typer.Exit(1)

    typer.echo(f"HPA subcellular annotations saved to {out}")


@app.command("fetch-panther")
def fetch_panther_cmd(
    input: str = typer.Option(..., "--input", help="Path to input CSV containing IDs"),
    id_col: str = typer.Option("primary_id", "--id-col", help="ID column name"),
    organism: str = typer.Option("9606", "--organism", help="PANTHER organism ID"),
    categories: Optional[str] = typer.Option(
        None,
        "--categories",
        help="Comma-separated PANTHER annotation categories",
    ),
    out: str = typer.Option(..., "--out", help="Path to output normalized CSV"),
    cache_dir: str = typer.Option(
        ".cache/biastracker/panther",
        "--cache-dir",
        help="Directory for PANTHER API cache files",
    ),
):
    """Fetch PANTHER API annotations for IDs in a CSV and save normalized CSV."""
    try:
        df = pd.read_csv(input)
        if id_col not in df.columns:
            raise ValueError(f"Column '{id_col}' not found in input CSV")
        category_list = _comma_list(categories)
        ids = df[id_col].dropna().astype(str).tolist()
        ann_set = load_panther_api_annotations(
            ids=ids,
            name="panther_api",
            organism=organism,
            categories=category_list,
            cache_dir=cache_dir,
        )
        _write_annotation_csv(ann_set.table, out)
    except Exception as e:
        typer.echo(f"Failed to fetch PANTHER annotations: {e}", err=True)
        raise typer.Exit(1)

    typer.echo(f"PANTHER annotations saved to {out}")

@app.command()
def run(
    config: str = typer.Argument(..., help="Path to config YAML file")
):
    from biastracker.config import load_config, validate_minimal_config
    from biastracker.workflow import run_workflow
    
    try:
        config_data = load_config(config)
        validate_minimal_config(config_data)
    except Exception as e:
        typer.echo(f"Error loading or validating config: {e}", err=True)
        raise typer.Exit(1)
        
    try:
        run_workflow(config_data, config_path=config)
        typer.echo("Workflow completed successfully.")
    except Exception as e:
        typer.echo(f"Workflow failed: {e}", err=True)
        raise typer.Exit(1)

def _download_hpa_subcellular(cache_dir: str) -> Path:
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    out_path = cache_path / "subcellular_location.tsv.zip"
    if out_path.exists():
        return out_path

    response = request_with_retries(
        "GET",
        HPA_SUBCELLULAR_DEFAULT_URL,
        timeout=120,
    )
    content = getattr(response, "content", None)
    if content is None:
        content = response.text.encode("utf-8")
    out_path.write_bytes(content)
    return out_path


def _comma_list(value: str | None) -> list[str] | None:
    if value is None:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    return items or None


def _write_annotation_csv(df: pd.DataFrame, out: str) -> None:
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)


if __name__ == "__main__":
    app()
