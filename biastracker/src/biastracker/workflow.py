import logging
from pathlib import Path
from typing import Dict, Any, Union

from biastracker.dataset import load_standard_table
from biastracker.annotations.custom import load_long_annotation_table
from biastracker.reports import (
    save_dataset_summary,
    prepare_output_dirs,
)
from biastracker.analysis.compare import compare_groups
from biastracker.analysis.enrichment import run_group_ora
from biastracker.plots import plot_violin, plot_cdf, plot_enrichment_dotplot

logger = logging.getLogger(__name__)

def run_workflow(config: Dict[str, Any], config_path: Union[str, Path, None] = None) -> None:
    """
    Executes a full BiasTracker workflow driven by a configuration dictionary.

    Args:
        config: The configuration dictionary.
        config_path: Optional path to the configuration file, used for resolving relative paths.
    """
    if config_path:
        base_dir = Path(config_path).parent
    else:
        base_dir = Path.cwd()

    def resolve_path(p: str) -> Path:
        path_obj = Path(p)
        if path_obj.is_absolute():
            return path_obj
        return base_dir / path_obj

    output_dir = resolve_path(config.get("output", {}).get("directory", "results"))
    dirs = prepare_output_dirs(output_dir)

    loaded_datasets = {}
    for ds_conf in config.get("datasets", []):
        ds_type = ds_conf.get("type")
        if ds_type != "standard_csv":
            raise NotImplementedError(f"Dataset type '{ds_type}' is not currently supported in config-driven runs. Only 'standard_csv' is supported.")
        
        path = resolve_path(ds_conf["path"])
        name = ds_conf["name"]
        level = ds_conf.get("level", "protein")
        group_col = ds_conf.get("group_col")
        
        dataset = load_standard_table(path, name=name, level=level, group_col=group_col)
        loaded_datasets[name] = dataset

    loaded_annotations = {}
    for ann_conf in config.get("annotations", []):
        ann_type = ann_conf.get("type")
        if ann_type not in ("long", "panther"):
            raise NotImplementedError(f"Annotation type '{ann_type}' is not currently supported in config-driven runs. Only 'long' and 'panther' are supported.")
        
        path = resolve_path(ann_conf["path"])
        name = ann_conf["name"]
        if ann_type == "long":
            source = ann_conf.get("source", "custom")
            ann_set = load_long_annotation_table(path, name=name, source=source)
        elif ann_type == "panther":
            panther_type = ann_conf.get("panther_type")
            if not panther_type:
                raise ValueError(f"panther_type is required for panther annotation '{name}'")
            from biastracker.annotations.panther import load_panther_annotation
            ann_set = load_panther_annotation(
                path=path,
                name=name,
                panther_type=panther_type,
                id_col=ann_conf.get("id_col"),
                term_id_col=ann_conf.get("term_id_col"),
                term_name_col=ann_conf.get("term_name_col"),
                category=ann_conf.get("category"),
            )
            
        loaded_annotations[name] = ann_set

    analysis_conf = config.get("analysis", {})

    # 1. Summaries
    if analysis_conf.get("summary"):
        for name, dataset in loaded_datasets.items():
            save_dataset_summary(dataset, output_dir=output_dir, filename=f"dataset_summary__{name}.csv")

    # 2. Comparisons
    for comp_conf in analysis_conf.get("comparisons", []):
        ds_name = comp_conf["dataset"]
        if ds_name not in loaded_datasets:
            continue
        dataset = loaded_datasets[ds_name]
        
        group_col = comp_conf["group_col"]
        group_a = comp_conf["group_a"]
        group_b = comp_conf["group_b"]
        features = comp_conf.get("features")
        
        available_features = dataset.available_features(features)
        missing = set(features or []) - set(available_features)
        if missing:
            logger.warning(f"Features missing from dataset '{ds_name}': {missing}")

        if not available_features:
            continue

        res = compare_groups(dataset, group_col, group_a, group_b, features=available_features)
        
        out_csv = dirs["tables"] / f"feature_statistics__{ds_name}__{group_a}_vs_{group_b}.csv"
        res.to_csv(out_csv, index=False)
        
        for feat in res["feature"].unique():
            safe_feat = str(feat).replace("/", "_").replace("\\", "_")
            v_path = dirs["figures"] / f"violin__{ds_name}__{safe_feat}__{group_a}_vs_{group_b}.png"
            c_path = dirs["figures"] / f"cdf__{ds_name}__{safe_feat}__{group_a}_vs_{group_b}.png"
            
            try:
                plot_violin(dataset, feature=feat, group_col=group_col, output_path=v_path)
            except ValueError:
                pass
            try:
                plot_cdf(dataset, feature=feat, group_col=group_col, output_path=c_path)
            except ValueError:
                pass

    # 3. Enrichment
    for enr_conf in analysis_conf.get("enrichment", []):
        ds_name = enr_conf["dataset"]
        ann_name = enr_conf["annotation"]
        if ds_name not in loaded_datasets or ann_name not in loaded_annotations:
            continue
            
        dataset = loaded_datasets[ds_name]
        if dataset.id_col not in dataset.table.columns:
            raise ValueError(f"Enrichment requires protein IDs. Column '{dataset.id_col}' not found in dataset '{ds_name}'.")

        ann_set = loaded_annotations[ann_name]
        group_col = enr_conf["group_col"]
        query_group = enr_conf["query_group"]
        min_term_size = enr_conf.get("min_term_size", 3)
        
        res = run_group_ora(dataset, group_col, query_group, ann_set, min_term_size=min_term_size)
        
        out_csv = dirs["tables"] / f"enrichment__{ds_name}__{query_group}__{ann_name}.csv"
        res.to_csv(out_csv, index=False)
        
        if not res.empty:
            out_plot = dirs["figures"] / f"enrichment_dotplot__{ds_name}__{query_group}__{ann_name}.png"
            try:
                plot_enrichment_dotplot(res, output_path=out_plot)
            except ValueError:
                logger.warning(f"Could not generate enrichment dotplot for '{ds_name}' ({query_group})")
        else:
            logger.warning(f"Enrichment results empty for '{ds_name}' ({query_group}). Skipping dotplot.")
