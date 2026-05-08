from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from biastracker.dataset import BiasDataset


def plot_violin(
    dataset: BiasDataset,
    feature: str,
    group_col: str | None = None,
    output_path: str | Path | None = None,
    title: str | None = None,
) -> plt.Figure:
    if feature not in dataset.table.columns:
        raise ValueError(f"Feature '{feature}' not found in dataset")
        
    df = dataset.table.dropna(subset=[feature])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    if group_col:
        if group_col not in df.columns:
            raise ValueError(f"Group column '{group_col}' not found in dataset")
        
        # Drop rows where group is NaN
        df = df.dropna(subset=[group_col])
        groups = df[group_col].unique()
        # Sort groups for consistent plotting if they are strings/numbers
        try:
            groups = sorted(groups)
        except TypeError:
            pass
            
        data = [df[df[group_col] == g][feature].values for g in groups]
        
        # filter out empty arrays in case a group only had NaNs for the feature
        valid_data = []
        valid_groups = []
        for g, d in zip(groups, data):
            if len(d) > 0:
                valid_data.append(d)
                valid_groups.append(g)
                
        if not valid_data:
            plt.close(fig)
            raise ValueError("No valid data available to plot after dropping NaNs.")

        parts = ax.violinplot(valid_data, showmeans=False, showmedians=True)
        ax.set_xticks(range(1, len(valid_groups) + 1))
        ax.set_xticklabels(valid_groups)
        ax.set_xlabel(group_col)
    else:
        data = df[feature].values
        if len(data) == 0:
            plt.close(fig)
            raise ValueError("No valid data available to plot after dropping NaNs.")
        parts = ax.violinplot([data], showmeans=False, showmedians=True)
        ax.set_xticks([1])
        ax.set_xticklabels([dataset.name])
        
    ax.set_ylabel(feature)
    if title:
        ax.set_title(title)
        
    if output_path:
        fig.savefig(output_path, bbox_inches='tight')
        plt.close(fig)
        
    return fig


def plot_cdf(
    dataset: BiasDataset,
    feature: str,
    group_col: str | None = None,
    output_path: str | Path | None = None,
    title: str | None = None,
) -> plt.Figure:
    if feature not in dataset.table.columns:
        raise ValueError(f"Feature '{feature}' not found in dataset")
        
    df = dataset.table.dropna(subset=[feature])
    
    if len(df) == 0:
        raise ValueError("No valid data available to plot after dropping NaNs.")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    if group_col:
        if group_col not in df.columns:
            raise ValueError(f"Group column '{group_col}' not found in dataset")
        
        df = df.dropna(subset=[group_col])
        groups = df[group_col].unique()
        try:
            groups = sorted(groups)
        except TypeError:
            pass
            
        for g in groups:
            g_data = df[df[group_col] == g][feature].values
            if len(g_data) == 0:
                continue
            x = np.sort(g_data)
            y = np.arange(1, len(x) + 1) / len(x)
            ax.plot(x, y, label=str(g), linewidth=2)
        ax.legend(title=group_col)
    else:
        data = df[feature].values
        x = np.sort(data)
        y = np.arange(1, len(x) + 1) / len(x)
        ax.plot(x, y, label=dataset.name, linewidth=2)
        ax.legend()
        
    ax.set_xlabel(feature)
    ax.set_ylabel("CDF")
    if title:
        ax.set_title(title)
        
    if output_path:
        fig.savefig(output_path, bbox_inches='tight')
        plt.close(fig)
        
    return fig


def plot_enrichment_dotplot(
    enrichment_df: pd.DataFrame,
    output_path: str | Path | None = None,
    top_n: int = 20,
    title: str | None = None,
) -> plt.Figure:
    if enrichment_df.empty:
        raise ValueError("Enrichment DataFrame is empty")
        
    metric = "fdr" if "fdr" in enrichment_df.columns else "p_value"
    if metric not in enrichment_df.columns:
        raise ValueError(f"Neither 'fdr' nor 'p_value' found in columns")
    if "term_name" not in enrichment_df.columns:
        raise ValueError("'term_name' column not found")
        
    df = enrichment_df.sort_values(metric, ascending=True).head(top_n).copy()
    
    # Reverse so the most significant is at the top of the plot
    df = df.iloc[::-1]
    
    fig, ax = plt.subplots(figsize=(8, max(4, len(df) * 0.4)))
    
    x = -np.log10(df[metric].astype(float) + 1e-300)  # avoid log(0)
    y = df["term_name"].astype(str)
    
    if "odds_ratio" in df.columns:
        sizes = df["odds_ratio"].astype(float) * 50
        # handle nan / inf
        sizes = sizes.replace([np.inf, -np.inf], np.nan).fillna(50)
        sizes = np.clip(sizes, 20, 500)
        scatter = ax.scatter(x, y, s=sizes, c=x, cmap="viridis", alpha=0.8)
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label(f"-log10({metric})")
    else:
        ax.scatter(x, y, c=x, cmap="viridis", s=100, alpha=0.8)
        
    ax.set_xlabel(f"-log10({metric})")
    ax.set_ylabel("Term")
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    if title:
        ax.set_title(title)
        
    if output_path:
        fig.savefig(output_path, bbox_inches='tight')
        plt.close(fig)
        
    return fig
