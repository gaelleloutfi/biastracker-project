import typer
import yaml
from typing import Optional
from pathlib import Path

from biastracker import __version__
from biastracker.dataset import check_protperties_available, load_standard_table
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
        except ValueError:
            pass  # May not have valid data to plot
            
        try:
            plot_cdf(dataset, feature=feat, group_col=group_col, output_path=c_path)
        except ValueError:
            pass
            
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
    
    res = run_group_ora(dataset, group_col, query_group, ann_set, min_term_size=min_term_size)
    save_enrichment_results(res, out)
    
    if not res.empty:
        dirs = prepare_output_dirs(out)
        try:
            plot_enrichment_dotplot(res, output_path=dirs["figures"] / "enrichment_dotplot.png")
        except ValueError:
            pass
            
    typer.echo(f"Enrichment analysis completed. Results saved to {out}")

@app.command()
def run(
    config: str = typer.Argument(..., help="Path to config YAML file")
):
    """Recognizes a YAML config file for full workflow execution."""
    try:
        with open(config, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except Exception as e:
        typer.echo(f"Failed to load config: {e}", err=True)
        raise typer.Exit(1)
        
    if not isinstance(data, dict) or "datasets" not in data or "analyses" not in data:
        typer.echo("Error: Config must contain 'datasets' and 'analyses'", err=True)
        raise typer.Exit(1)
        
    typer.echo("config-driven execution is recognized but not fully implemented yet")

if __name__ == "__main__":
    app()
