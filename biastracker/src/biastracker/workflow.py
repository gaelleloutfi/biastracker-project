import logging
import shutil
import warnings
from pathlib import Path
from typing import Dict, Any, Union

from biastracker.annotations._http import request_with_retries
from biastracker.dataset import (
    load_standard_table,
    load_manual_table_via_protperties,
    load_diann_report,
    load_maxquant_evidence,
    load_maxquant_proteingroups,
    load_diann_pg_matrix,
)
from biastracker.annotations.custom import load_long_annotation_table, load_gmt
from biastracker.reports import (
    export_annotations_for_workflow,
    save_dataset_summary,
    prepare_output_dirs,
)
from biastracker.analysis.compare import compare_groups
from biastracker.analysis.enrichment import run_group_ora
from biastracker.plots import plot_violin, plot_cdf, plot_enrichment_dotplot

logger = logging.getLogger(__name__)

HPA_SUBCELLULAR_DEFAULT_URL = (
    "https://www.proteinatlas.org/download/tsv/subcellular_location.tsv.zip"
)

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

    def resolve_optional_path(p: str | None) -> Path | None:
        if p is None:
            return None
        return resolve_path(p)

    output_dir = resolve_path(config.get("output", {}).get("directory", "results"))
    dirs = prepare_output_dirs(output_dir)
    if config_path:
        shutil.copy2(config_path, output_dir / "config_used.yaml")

    loaded_datasets = {}
    for ds_conf in config.get("datasets", []):
        ds_type = ds_conf.get("type")
        path = resolve_path(ds_conf["path"])
        name = ds_conf["name"]
        level = ds_conf.get("level", "protein")
        group_col = ds_conf.get("group_col")

        kwargs = {k: v for k, v in ds_conf.items() if k not in {"name", "type", "path", "level"}}
        if ds_type == "standard_csv":
            dataset = load_standard_table(path, name=name, level=level, **kwargs)
        elif ds_type == "manual":
            dataset = load_manual_table_via_protperties(path, name=name, level=level, **kwargs)
        elif ds_type == "diann_report":
            dataset = load_diann_report(path, name=name, **kwargs)
        elif ds_type == "maxquant_evidence":
            dataset = load_maxquant_evidence(path, name=name, **kwargs)
        elif ds_type == "maxquant_proteingroups":
            dataset = load_maxquant_proteingroups(path, name=name, **kwargs)
        elif ds_type == "diann_pg_matrix":
            dataset = load_diann_pg_matrix(path, name=name, **kwargs)
        else:
            raise ValueError(f"Unsupported dataset type '{ds_type}' for dataset '{name}'")
        loaded_datasets[name] = dataset

    loaded_annotations = {}
    for ann_conf in config.get("annotations", []):
        ann_type = ann_conf.get("type")
        name = ann_conf["name"]

        def annotation_path() -> Path:
            if "path" not in ann_conf:
                raise ValueError(f"path is required for annotation '{name}'")
            return resolve_path(ann_conf["path"])

        if ann_type == "long":
            path = annotation_path()
            source = ann_conf.get("source", "custom")
            ann_set = load_long_annotation_table(
                path,
                name=name,
                source=source,
                id_col=ann_conf.get("id_col", "primary_id"),
                term_col=ann_conf.get("term_col", "term_name"),
                term_id_col=ann_conf.get("term_id_col"),
                category_col=ann_conf.get("category_col"),
            )
        elif ann_type == "gmt":
            path = annotation_path()
            ann_set = load_gmt(
                path,
                name=name,
                source=ann_conf.get("source", "custom"),
                category=ann_conf.get("category", "unknown"),
            )
        elif ann_type in {"panther", "panther_file"}:
            if ann_type == "panther":
                warnings.warn(
                    "Annotation type 'panther' is deprecated; use 'panther_file' "
                    f"for annotation '{name}'",
                    DeprecationWarning,
                    stacklevel=2,
                )
            path = annotation_path()
            panther_type = ann_conf.get("panther_type")
            if not panther_type:
                raise ValueError(
                    f"panther_type is required for panther_file annotation '{name}'"
                )
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
        elif ann_type == "panther_api":
            dataset_name = ann_conf.get("dataset")
            if not dataset_name:
                raise ValueError(f"dataset is required for panther_api annotation '{name}'")
            if dataset_name not in loaded_datasets:
                raise ValueError(
                    f"panther_api annotation '{name}' references dataset "
                    f"'{dataset_name}', which is not loaded"
                )
            dataset = loaded_datasets[dataset_name]
            id_col = ann_conf.get("id_col", dataset.id_col)
            if id_col not in dataset.table.columns:
                raise ValueError(
                    f"panther_api annotation '{name}' requires id_col '{id_col}' "
                    f"in dataset '{dataset_name}'"
                )

            from biastracker.annotations.panther import load_panther_api_annotations

            ids = dataset.table[id_col].dropna().astype(str).tolist()
            ann_set = load_panther_api_annotations(
                ids=ids,
                name=name,
                organism=ann_conf.get("organism", 9606),
                categories=ann_conf.get("categories"),
                batch_size=ann_conf.get("batch_size", 500),
                cache_dir=resolve_optional_path(ann_conf.get("cache_dir"))
                or ".cache/biastracker/panther",
                use_cache=ann_conf.get("use_cache", True),
            )
        elif ann_type == "hpa_subcellular":
            from biastracker.annotations.hpa import load_hpa_subcellular
            hpa_path = _resolve_hpa_subcellular_source(
                ann_conf=ann_conf,
                name=name,
                resolve_path=resolve_path,
            )
            map_to_uniprot = ann_conf.get("map_to_uniprot", False)
            ann_set = load_hpa_subcellular(
                path=hpa_path,
                name=name,
                map_to_uniprot=map_to_uniprot,
                ensembl_to_uniprot=ann_conf.get("ensembl_to_uniprot"),
                cache_dir=resolve_optional_path(ann_conf.get("cache_dir")),
                use_uniprot_api=ann_conf.get(
                    "use_uniprot_api",
                    bool(map_to_uniprot and ann_conf.get("ensembl_to_uniprot") is None),
                ),
                from_ns=ann_conf.get("from_ns", "Ensembl"),
                to_ns=ann_conf.get("to_ns", "UniProtKB"),
            )
        elif ann_type == "umap_subcellular":
            path = annotation_path()
            from biastracker.annotations.umap_subcellular import load_umap_subcellular
            ann_set = load_umap_subcellular(
                path=path,
                name=name,
                id_col=ann_conf.get("id_col"),
                location_col=ann_conf.get("location_col"),
            )
        else:
            raise ValueError(f"Unsupported annotation type '{ann_type}' for annotation '{name}'")
        loaded_annotations[name] = ann_set

    export_annotations_for_workflow(loaded_annotations, output_dir)

    analysis_conf = config.get("analysis", {})

    # 1. Summaries
    if analysis_conf.get("summary"):
        for name, dataset in loaded_datasets.items():
            save_dataset_summary(dataset, output_dir=output_dir, filename=f"dataset_summary__{name}.csv")

    # 2. Comparisons
    for comp_conf in analysis_conf.get("comparisons", []):
        ds_name = comp_conf["dataset"]
        if ds_name not in loaded_datasets:
            raise ValueError(f"Comparison references unknown dataset '{ds_name}'")
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
            raise ValueError(
                f"No valid numeric features available for comparison in dataset '{ds_name}'. "
                f"Requested features: {features}. "
                f"Available numeric features: {dataset.available_features()}."
            )

        res = compare_groups(dataset, group_col, group_a, group_b, features=available_features)
        
        out_csv = dirs["tables"] / f"feature_statistics__{ds_name}__{group_a}_vs_{group_b}.csv"
        res.to_csv(out_csv, index=False)
        
        for feat in res["feature"].unique():
            safe_feat = str(feat).replace("/", "_").replace("\\", "_")
            v_path = dirs["figures"] / f"violin__{ds_name}__{safe_feat}__{group_a}_vs_{group_b}.png"
            c_path = dirs["figures"] / f"cdf__{ds_name}__{safe_feat}__{group_a}_vs_{group_b}.png"
            
            try:
                plot_violin(dataset, feature=feat, group_col=group_col, output_path=v_path)
            except Exception as e:
                logger.warning("Skipping violin plot for '%s' in dataset '%s': %s", feat, ds_name, e)
            try:
                plot_cdf(dataset, feature=feat, group_col=group_col, output_path=c_path)
            except Exception as e:
                logger.warning("Skipping CDF plot for '%s' in dataset '%s': %s", feat, ds_name, e)

    # 3. Enrichment
    for enr_conf in analysis_conf.get("enrichment", []):
        ds_name = enr_conf["dataset"]
        ann_name = enr_conf["annotation"]
        if ds_name not in loaded_datasets:
            raise ValueError(f"Enrichment references unknown dataset '{ds_name}'")
        if ann_name not in loaded_annotations:
            raise ValueError(f"Enrichment references unknown annotation '{ann_name}'")
            
        dataset = loaded_datasets[ds_name]
        if dataset.id_col not in dataset.table.columns:
            raise ValueError(f"Enrichment requires protein IDs. Column '{dataset.id_col}' not found in dataset '{ds_name}'.")

        ann_set = loaded_annotations[ann_name]
        group_col = enr_conf["group_col"]
        query_group = enr_conf["query_group"]
        min_term_size = enr_conf.get("min_term_size", 3)
        
        res = run_group_ora(
            dataset=dataset,
            group_col=group_col,
            query_group=query_group,
            annotations=ann_set,
            min_term_size=min_term_size,
        )
        
        out_csv = dirs["tables"] / f"enrichment__{ds_name}__{query_group}__{ann_name}.csv"
        res.to_csv(out_csv, index=False)
        
        if not res.empty:
            out_plot = dirs["figures"] / f"enrichment_dotplot__{ds_name}__{query_group}__{ann_name}.png"
            try:
                plot_enrichment_dotplot(res, output_path=out_plot)
            except Exception as e:
                logger.warning("Could not generate enrichment dotplot for '%s' (%s): %s", ds_name, query_group, e)
        else:
            logger.warning(f"Enrichment results empty for '{ds_name}' ({query_group}). Skipping dotplot.")


def _resolve_hpa_subcellular_source(
    ann_conf: dict[str, Any],
    name: str,
    resolve_path,
) -> Path:
    if "path" in ann_conf and ann_conf["path"]:
        return resolve_path(ann_conf["path"])

    url = ann_conf.get("url") or HPA_SUBCELLULAR_DEFAULT_URL
    cache_dir = resolve_path(ann_conf.get("cache_dir", ".cache/biastracker/hpa"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    filename = Path(str(url).split("?", 1)[0]).name or "subcellular_location.tsv.zip"
    out_path = cache_dir / filename

    if out_path.exists() and ann_conf.get("use_cache", True):
        return out_path

    logger.info("Downloading HPA subcellular annotation '%s' from %s", name, url)
    response = request_with_retries("GET", url, timeout=ann_conf.get("timeout", 120))
    content = getattr(response, "content", None)
    if content is None:
        content = response.text.encode("utf-8")
    out_path.write_bytes(content)
    return out_path
