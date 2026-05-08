import pandas as pd
import pytest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from biastracker.plots import plot_violin, plot_cdf, plot_enrichment_dotplot


@pytest.fixture
def dummy_enrichment():
    return pd.DataFrame({
        "term_id": ["T1", "T2", "T3"],
        "term_name": ["Term A", "Term B", "Term C"],
        "p_value": [0.01, 0.05, 0.001],
        "fdr": [0.05, 0.1, 0.01],
        "odds_ratio": [2.0, 1.5, 3.0]
    })


def test_plot_violin_with_group_saves_png(toy_bias_dataset, tmp_path):
    out_path = tmp_path / "violin.png"
    fig = plot_violin(toy_bias_dataset, feature="length", group_col="group", output_path=out_path)
    assert isinstance(fig, plt.Figure)
    assert out_path.exists()
    plt.close(fig)

def test_plot_violin_returns_figure_when_not_saving(toy_bias_dataset):
    fig = plot_violin(toy_bias_dataset, feature="length", group_col="group")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

def test_plot_cdf_saves_png(toy_bias_dataset, tmp_path):
    out_path = tmp_path / "cdf.png"
    fig = plot_cdf(toy_bias_dataset, feature="length", group_col="group", output_path=out_path)
    assert isinstance(fig, plt.Figure)
    assert out_path.exists()
    plt.close(fig)

def test_plot_cdf_returns_figure_when_not_saving(toy_bias_dataset):
    fig = plot_cdf(toy_bias_dataset, feature="length", group_col="group")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

def test_plot_enrichment_dotplot_saves_png(dummy_enrichment, tmp_path):
    out_path = tmp_path / "dotplot.png"
    fig = plot_enrichment_dotplot(dummy_enrichment, output_path=out_path)
    assert isinstance(fig, plt.Figure)
    assert out_path.exists()
    plt.close(fig)

def test_plot_enrichment_dotplot_returns_figure(dummy_enrichment):
    fig = plot_enrichment_dotplot(dummy_enrichment)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

def test_plot_enrichment_dotplot_empty(tmp_path):
    empty_df = pd.DataFrame(columns=["term_id", "term_name", "p_value", "fdr", "odds_ratio"])
    with pytest.raises(ValueError, match="Enrichment DataFrame is empty"):
        plot_enrichment_dotplot(empty_df, output_path=tmp_path / "dotplot.png")
