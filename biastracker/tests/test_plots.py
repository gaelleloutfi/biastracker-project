import pandas as pd
import numpy as np
import pytest
import matplotlib.pyplot as plt
from pathlib import Path

from biastracker.dataset import BiasDataset
from biastracker.plots import plot_violin, plot_cdf, plot_enrichment_dotplot


@pytest.fixture
def dummy_dataset():
    df = pd.DataFrame({
        "primary_id": ["P1", "P2", "P3", "P4", "P5"],
        "level": ["protein"] * 5,
        "sequence": ["A", "B", "C", "D", "E"],
        "intensity": [10.0, 15.0, np.nan, 20.0, 25.0],
        "group": ["A", "A", "B", "B", "B"]
    })
    return BiasDataset(name="test", table=df, level="protein", group_col="group")


@pytest.fixture
def dummy_enrichment():
    return pd.DataFrame({
        "term_id": ["T1", "T2", "T3"],
        "term_name": ["Term A", "Term B", "Term C"],
        "p_value": [0.01, 0.05, 0.001],
        "fdr": [0.05, 0.1, 0.01],
        "odds_ratio": [2.0, 1.5, 3.0]
    })


def test_plot_violin_no_group(dummy_dataset, tmp_path):
    out_path = tmp_path / "violin.png"
    fig = plot_violin(dummy_dataset, feature="intensity", output_path=out_path)
    assert isinstance(fig, plt.Figure)
    assert out_path.exists()
    plt.close(fig)


def test_plot_violin_with_group(dummy_dataset, tmp_path):
    out_path = tmp_path / "violin_group.png"
    fig = plot_violin(dummy_dataset, feature="intensity", group_col="group", output_path=out_path)
    assert isinstance(fig, plt.Figure)
    assert out_path.exists()
    plt.close(fig)


def test_plot_cdf_no_group(dummy_dataset, tmp_path):
    out_path = tmp_path / "cdf.png"
    fig = plot_cdf(dummy_dataset, feature="intensity", output_path=out_path)
    assert isinstance(fig, plt.Figure)
    assert out_path.exists()
    plt.close(fig)


def test_plot_cdf_with_group(dummy_dataset, tmp_path):
    out_path = tmp_path / "cdf_group.png"
    fig = plot_cdf(dummy_dataset, feature="intensity", group_col="group", output_path=out_path)
    assert isinstance(fig, plt.Figure)
    assert out_path.exists()
    plt.close(fig)


def test_plot_enrichment_dotplot(dummy_enrichment, tmp_path):
    out_path = tmp_path / "dotplot.png"
    fig = plot_enrichment_dotplot(dummy_enrichment, output_path=out_path)
    assert isinstance(fig, plt.Figure)
    assert out_path.exists()
    plt.close(fig)


def test_plot_enrichment_no_fdr(dummy_enrichment, tmp_path):
    df = dummy_enrichment.drop(columns=["fdr"])
    out_path = tmp_path / "dotplot_pval.png"
    fig = plot_enrichment_dotplot(df, output_path=out_path)
    assert isinstance(fig, plt.Figure)
    assert out_path.exists()
    plt.close(fig)
