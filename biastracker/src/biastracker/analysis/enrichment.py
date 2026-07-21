"""Enrichment analysis for BiasTracker.

This module provides two complementary enrichment workflows:

* :func:`run_ora`  — core function that accepts explicit ID sets.
* :func:`run_group_ora` — convenience wrapper that extracts IDs from a
  :class:`~biastracker.dataset.BiasDataset` group column.
* :func:`run_fgsea` — pre-ranked gene set enrichment analysis.
* :func:`run_dataset_fgsea` — convenience wrapper that ranks a dataset column.
"""
from __future__ import annotations

import logging
from typing import Literal, Set

import numpy as np
import pandas as pd
from scipy import stats

_logger = logging.getLogger(__name__)

from biastracker.dataset import AnnotationSet, BiasDataset
from biastracker.stats import adjust_pvalues

# ---------------------------------------------------------------------------
# Column order for the returned DataFrame
# ---------------------------------------------------------------------------
_ORA_COLS = [
    "term_id",
    "term_name",
    "source",
    "category",
    "query_count",
    "query_size",
    "background_count",
    "background_size",
    "odds_ratio",
    "p_value",
    "fdr",
    "direction",
    "query_hits",
]

_GROUP_ORA_COLS = ["dataset", "group_col", "query_group"] + _ORA_COLS

_FGSEA_COLS = [
    "term_id",
    "term_name",
    "source",
    "category",
    "set_size",
    "es",
    "nes",
    "p_value",
    "fdr",
    "leading_edge",
]

_DATASET_FGSEA_COLS = ["dataset", "score_col"] + _FGSEA_COLS

# Default FDR significance threshold shared across enrichment visualisations.
DEFAULT_SIG_THRESHOLD = 0.05
# Floor applied to the significance statistic before -log10 so 0 does not
# become +inf.
_FDR_FLOOR = 1e-300

# Selectable significance criteria: label -> (statistic column, threshold).
# 0.25 is the conventional GSEA FDR cutoff.
SIGNIFICANCE_PRESETS: dict[str, tuple[str, float]] = {
    "Nominal p ≤ 0.05": ("p_value", 0.05),
    "FDR ≤ 0.05": ("fdr", 0.05),
    "FDR ≤ 0.25": ("fdr", 0.25),
}
DEFAULT_SIGNIFICANCE_PRESET = "FDR ≤ 0.05"

# Extra columns added by prepare_enrichment_volcano_data.
_VOLCANO_EXTRA_COLS = [
    "effect", "neg_log10_sig", "direction", "significant",
    "effect_col", "significance_metric", "sig_threshold",
]


def prepare_enrichment_volcano_data(
    result: pd.DataFrame,
    effect_col: str | None = None,
    significance_metric: str = "fdr",
    sig_threshold: float = DEFAULT_SIG_THRESHOLD,
    sig_floor: float = _FDR_FLOOR,
) -> pd.DataFrame:
    """Augment an fGSEA (or ORA) result table with volcano-plot coordinates.

    Adds, without mutating *result*:

    * ``effect`` — the signed effect size. When *effect_col* is ``None`` it is
      chosen as ``"nes"`` if present, otherwise ``"es"`` (fallback); pass an
      explicit column to override.
    * ``neg_log10_sig`` — ``-log10`` of the chosen significance statistic
      (*significance_metric*, ``"fdr"`` or ``"p_value"``), clipped to *sig_floor*
      so a value of 0 yields a large finite number instead of ``+inf``. Rows with
      a missing statistic keep ``NaN`` here. The original ``p_value`` / ``fdr``
      columns are left untouched for hover text.
    * ``direction`` — ``"positive"`` (effect ≥ 0) or ``"negative"``.
    * ``significant`` — ``statistic <= sig_threshold`` (missing → ``False``).
    * ``effect_col`` / ``significance_metric`` / ``sig_threshold`` — the choices
      used, so plotting code can label axes and reference lines.

    An empty *result* returns an empty frame carrying the extra columns so
    callers can rely on the schema.

    Raises
    ------
    ValueError
        If no usable effect column can be found, or *significance_metric* is
        missing from *result*.
    """
    if effect_col is None:
        effect_col = "nes" if "nes" in result.columns else (
            "es" if "es" in result.columns else None
        )
        if effect_col is None:
            raise ValueError(
                "No effect-size column found (expected 'nes' or 'es'); "
                "pass effect_col explicitly."
            )
    elif effect_col not in result.columns:
        raise ValueError(f"effect_col '{effect_col}' not found in result columns.")
    if significance_metric not in result.columns:
        raise ValueError(
            f"significance_metric '{significance_metric}' not found in result columns."
        )

    out = result.copy()
    if out.empty:
        for col in _VOLCANO_EXTRA_COLS:
            out[col] = pd.Series(dtype="object" if col in {"direction", "effect_col",
                                 "significance_metric"} else "float64")
        return out

    stat = pd.to_numeric(out[significance_metric], errors="coerce")
    out["effect"] = pd.to_numeric(out[effect_col], errors="coerce")
    out["neg_log10_sig"] = -np.log10(stat.clip(lower=sig_floor))
    # A missing statistic must not masquerade as significant or a real y-coord.
    out.loc[stat.isna(), "neg_log10_sig"] = np.nan
    out["direction"] = np.where(out["effect"] >= 0, "positive", "negative")
    out["significant"] = (stat <= sig_threshold).fillna(False)
    out["effect_col"] = effect_col
    out["significance_metric"] = significance_metric
    out["sig_threshold"] = sig_threshold
    return out


# ---------------------------------------------------------------------------
# run_ora
# ---------------------------------------------------------------------------

def run_ora(
    query_ids: Set[str],
    background_ids: Set[str],
    annotations: AnnotationSet,
    min_term_size: int = 3,
    correction: str = "fdr_bh",
) -> pd.DataFrame:
    """Run over-representation analysis (ORA) using Fisher's exact test.

    For every term in *annotations* the function builds a 2×2 contingency
    table and tests whether the query set is enriched or depleted.

    Parameters
    ----------
    query_ids:
        Set of entity identifiers in the query group.
    background_ids:
        Universe of all observable identifiers.  Query IDs not present in the
        background are silently excluded.
    annotations:
        :class:`~biastracker.dataset.AnnotationSet` providing term→ID mappings.
    min_term_size:
        Minimum number of background proteins annotated with a term to include
        it in the analysis.  Terms below this threshold are skipped.
    correction:
        Multiple-testing correction method forwarded to
        :func:`~biastracker.stats.adjust_pvalues` (default ``"fdr_bh"``).

    Returns
    -------
    pd.DataFrame
        One row per tested term with columns defined in ``_ORA_COLS``.
    """
    background_ids = set(str(b) for b in background_ids)
    if not background_ids:
        raise ValueError("background_ids is empty; ORA requires a non-empty background universe")

    # Restrict query to background universe
    query_ids = set(str(q) for q in query_ids) & background_ids
    if not query_ids:
        raise ValueError("query_ids is empty after intersecting with background_ids")

    n_query = len(query_ids)
    n_background = len(background_ids)
    n_not_query = n_background - n_query

    ann_df = annotations.table
    id_col = annotations.id_col
    term_col = annotations.term_col
    term_id_col = annotations.term_id_col
    cat_col = annotations.category_col
    source = annotations.source

    # Group annotation table by term once for efficiency
    term_groups = ann_df.groupby(term_col, sort=False)

    rows: list[dict] = []
    for term_name, term_df in term_groups:
        # IDs in background annotated with this term
        annotated_ids = set(term_df[id_col].astype(str)) & background_ids
        background_count = len(annotated_ids)

        if background_count < min_term_size:
            continue

        # IDs in query annotated with this term
        query_hits_ids = annotated_ids & query_ids
        query_count = len(query_hits_ids)

        # 2×2 contingency table
        a = query_count                          # query ∩ term
        b = n_query - query_count                # query ∩ ¬term
        c = background_count - query_count       # ¬query ∩ term
        d = n_not_query - c                      # ¬query ∩ ¬term

        odds_ratio, p_value = stats.fisher_exact([[a, b], [c, d]], alternative="two-sided")

        if odds_ratio > 1:
            direction = "enriched"
        elif odds_ratio < 1:
            direction = "depleted"
        else:
            direction = "neutral"

        # Representative term_id and category from annotation table
        term_id = term_df[term_id_col].iloc[0] if term_id_col in term_df.columns else term_name
        category = term_df[cat_col].iloc[0] if cat_col in term_df.columns else "unknown"

        rows.append({
            "term_id": term_id,
            "term_name": term_name,
            "source": source,
            "category": category,
            "query_count": query_count,
            "query_size": n_query,
            "background_count": background_count,
            "background_size": n_background,
            "odds_ratio": odds_ratio,
            "p_value": p_value,
            "fdr": np.nan,          # filled below
            "direction": direction,
            "query_hits": ";".join(sorted(query_hits_ids)),
        })

    if not rows:
        return pd.DataFrame(columns=_ORA_COLS)

    result = pd.DataFrame(rows)
    result = adjust_pvalues(result, p_col="p_value", method=correction, out_col="fdr")

    return result[_ORA_COLS].reset_index(drop=True)


# ---------------------------------------------------------------------------
# run_group_ora
# ---------------------------------------------------------------------------

def run_group_ora(
    dataset: BiasDataset,
    group_col: str,
    query_group: str,
    annotations: AnnotationSet,
    background: Literal["all", "other"] = "all",
    id_col: str = "primary_id",
    min_term_size: int = 3,
    correction: str = "fdr_bh",
) -> pd.DataFrame:
    """Convenience wrapper around :func:`run_ora` that extracts IDs from a dataset group.

    Parameters
    ----------
    dataset:
        Source :class:`~biastracker.dataset.BiasDataset`.
    group_col:
        Column in ``dataset.table`` that identifies groups.
    query_group:
        Value in *group_col* that defines the query set.
    annotations:
        :class:`~biastracker.dataset.AnnotationSet` providing term→ID mappings.
    background:
        How to define the background universe:

        * ``"all"`` — all IDs in the dataset (default).
        * ``"other"`` — all IDs in the dataset that do *not* belong to
          *query_group* (i.e. the complement within the dataset).
    id_col:
        Column containing entity identifiers (default ``"primary_id"``).
    min_term_size:
        Forwarded to :func:`run_ora`.
    correction:
        Forwarded to :func:`run_ora`.

    Returns
    -------
    pd.DataFrame
        Result of :func:`run_ora` with three extra leading columns:
        ``dataset``, ``group_col``, and ``query_group``.

    Raises
    ------
    ValueError
        If *group_col* is not found in ``dataset.table``.
    ValueError
        If *id_col* is not found in ``dataset.table``.
    """
    df = dataset.table
    if background not in {"all", "other"}:
        raise ValueError("background must be 'all' or 'other'")

    if group_col not in df.columns:
        raise ValueError(
            f"group_col '{group_col}' not found in dataset.table. "
            f"Available columns: {list(df.columns)}"
        )
    if id_col not in df.columns:
        raise ValueError(
            f"id_col '{id_col}' not found in dataset.table. "
            f"Available columns: {list(df.columns)}"
        )
    groups = set(df[group_col].dropna().astype(str))
    if str(query_group) not in groups:
        raise ValueError(
            f"query_group '{query_group}' not found in group_col '{group_col}'. "
            f"Available groups: {sorted(groups)}"
        )
    group_values = df[group_col].astype(str)
    query_group_str = str(query_group)

    query_ids = set(
        df.loc[group_values == query_group_str, id_col]
        .dropna()
        .astype(str)
    )

    if background == "other":
        background_ids = set(
            df.loc[group_values != query_group_str, id_col]
            .dropna()
            .astype(str)
        )
        # Include query IDs in the background universe too so the Fisher table
        # is built over the full dataset (consistent with standard ORA practice
        # when background="other" means "complement within dataset").
        background_ids = background_ids | query_ids
    else:  # "all"
        background_ids = set(df[id_col].dropna().astype(str))

    result = run_ora(
        query_ids=query_ids,
        background_ids=background_ids,
        annotations=annotations,
        min_term_size=min_term_size,
        correction=correction,
    )

    if result.empty:
        for col in ("dataset", "group_col", "query_group"):
            result[col] = pd.Series(dtype=str)
        return result[_GROUP_ORA_COLS]

    result = result.copy()
    result.insert(0, "query_group", query_group)
    result.insert(0, "group_col", group_col)
    result.insert(0, "dataset", dataset.name)

    return result[_GROUP_ORA_COLS].reset_index(drop=True)


def _enrichment_score(
    scores: np.ndarray,
    hit_mask: np.ndarray,
    weight: float = 1.0,
) -> tuple[float, list[int]]:
    """Compute the weighted pre-ranked enrichment score for one gene set."""
    n_total = len(scores)
    n_hits = int(hit_mask.sum())
    if n_hits == 0:
        raise ValueError("hit_mask contains no hits")
    if n_hits == n_total:
        raise ValueError("hit_mask contains all ranked IDs")

    hit_weights = np.abs(scores) ** weight
    hit_weights = np.where(hit_mask, hit_weights, 0.0)
    weight_sum = hit_weights.sum()
    if weight_sum == 0:
        hit_weights = np.where(hit_mask, 1.0 / n_hits, 0.0)
    else:
        hit_weights = hit_weights / weight_sum

    miss_step = 1.0 / (n_total - n_hits)
    running = np.cumsum(np.where(hit_mask, hit_weights, -miss_step))
    max_idx = int(np.argmax(running))
    min_idx = int(np.argmin(running))
    max_es = float(running[max_idx])
    min_es = float(running[min_idx])

    if abs(max_es) >= abs(min_es):
        leading_edge = np.flatnonzero(hit_mask[: max_idx + 1]).tolist()
        return max_es, leading_edge

    leading_edge = np.flatnonzero(hit_mask[min_idx:]).tolist()
    leading_edge = [idx + min_idx for idx in leading_edge]
    return min_es, leading_edge


def _normalise_fgsea_pvalue(
    es: float,
    null_es: np.ndarray,
) -> tuple[float, float]:
    """Return normalized ES and a same-tail permutation p-value."""
    if es >= 0:
        same_tail = null_es[null_es >= 0]
        if len(same_tail) == 0:
            same_tail = null_es
        positive_tail = same_tail[same_tail > 0]
        denom = np.mean(positive_tail) if len(positive_tail) else np.nan
        nes = es / denom if not np.isnan(denom) else np.nan
        p_value = (np.sum(same_tail >= es) + 1) / (len(same_tail) + 1)
    else:
        same_tail = null_es[null_es < 0]
        if len(same_tail) == 0:
            same_tail = null_es
        negative_tail = same_tail[same_tail < 0]
        denom = np.mean(np.abs(negative_tail)) if len(negative_tail) else np.nan
        nes = es / denom if not np.isnan(denom) else np.nan
        p_value = (np.sum(same_tail <= es) + 1) / (len(same_tail) + 1)
    return float(nes), float(p_value)


def run_fgsea(
    ranked_scores: pd.Series,
    annotations: AnnotationSet,
    min_term_size: int = 3,
    max_term_size: int | None = None,
    n_permutations: int = 1000,
    weight: float = 1.0,
    seed: int | None = None,
    correction: str = "fdr_bh",
) -> pd.DataFrame:
    """Run pre-ranked gene set enrichment analysis.

    Parameters
    ----------
    ranked_scores:
        Numeric scores indexed by entity identifier. Higher scores are treated
        as the top of the ranked list. Missing scores and missing IDs are
        removed. Duplicate IDs are resolved by keeping the first entry after
        sorting by descending score.
    annotations:
        Annotation set containing term-to-ID mappings.
    min_term_size:
        Minimum number of ranked IDs annotated with a term.
    max_term_size:
        Optional maximum number of ranked IDs annotated with a term.
    n_permutations:
        Number of gene-label permutations used to estimate p-values and NES.
    weight:
        Exponent applied to absolute ranking scores in the running-sum statistic.
        ``1.0`` matches the common weighted GSEA setting; ``0.0`` is unweighted.
    seed:
        Optional random seed for reproducible permutation results.
    correction:
        Multiple-testing correction method forwarded to
        :func:`~biastracker.stats.adjust_pvalues`.

    Returns
    -------
    pd.DataFrame
        One row per tested term with enrichment score, normalized enrichment
        score, permutation p-value, FDR, and leading-edge IDs.
    """
    if not isinstance(ranked_scores, pd.Series):
        raise TypeError("ranked_scores must be a pandas Series indexed by ID")
    if n_permutations < 1:
        raise ValueError("n_permutations must be >= 1")
    if min_term_size < 1:
        raise ValueError("min_term_size must be >= 1")
    if max_term_size is not None and max_term_size < min_term_size:
        raise ValueError("max_term_size must be >= min_term_size")
    if weight < 0:
        raise ValueError("weight must be >= 0")

    ranked = pd.DataFrame(
        {
            "primary_id": ranked_scores.index.astype(str),
            "score": pd.to_numeric(ranked_scores, errors="coerce").to_numpy(),
        }
    ).dropna(subset=["score"])
    # Drop NaN index entries that astype(str) converts to the literal "nan"
    ranked = ranked[ranked["primary_id"] != "nan"]
    # Infinite scores produce NaN enrichment statistics; exclude them
    ranked = ranked[np.isfinite(ranked["score"])]
    ranked = ranked.sort_values("score", ascending=False, kind="mergesort")
    ranked = ranked.drop_duplicates(subset=["primary_id"], keep="first")

    if len(ranked) < 2:
        raise ValueError("ranked_scores must contain at least two valid IDs")

    ranked_ids = ranked["primary_id"].to_numpy(dtype=str)
    scores = ranked["score"].to_numpy(dtype=float)
    id_to_pos = {identifier: i for i, identifier in enumerate(ranked_ids)}
    ranked_id_set = set(ranked_ids)

    rng = np.random.default_rng(seed)
    null_cache: dict[int, np.ndarray] = {}

    ann_df = annotations.table
    id_col = annotations.id_col
    term_col = annotations.term_col
    term_id_col = annotations.term_id_col
    cat_col = annotations.category_col
    source = annotations.source

    all_ann_ids = set(ann_df[id_col].astype(str))
    if all_ann_ids:
        overlap_frac = len(all_ann_ids & ranked_id_set) / len(all_ann_ids)
        if overlap_frac < 0.1:
            _logger.warning(
                "Low identifier overlap: only %.1f%% of annotation IDs are present in "
                "the ranked list. Check that identifier types match (e.g., UniProt "
                "accession vs. gene symbol). Enrichment results may be unreliable.",
                100 * overlap_frac,
            )

    rows: list[dict] = []
    for term_name, term_df in ann_df.groupby(term_col, sort=False):
        term_ids = set(term_df[id_col].astype(str)) & ranked_id_set
        set_size = len(term_ids)
        if set_size < min_term_size:
            continue
        if max_term_size is not None and set_size > max_term_size:
            continue
        if set_size >= len(ranked_ids):
            continue

        hit_mask = np.zeros(len(ranked_ids), dtype=bool)
        hit_positions = [id_to_pos[identifier] for identifier in term_ids]
        hit_mask[hit_positions] = True
        es, leading_positions = _enrichment_score(scores, hit_mask, weight=weight)

        if set_size not in null_cache:
            null_scores = np.empty(n_permutations, dtype=float)
            for i in range(n_permutations):
                permuted_mask = np.zeros(len(ranked_ids), dtype=bool)
                permuted_hits = rng.choice(len(ranked_ids), size=set_size, replace=False)
                permuted_mask[permuted_hits] = True
                null_scores[i], _ = _enrichment_score(scores, permuted_mask, weight=weight)
            null_cache[set_size] = null_scores

        nes, p_value = _normalise_fgsea_pvalue(es, null_cache[set_size])
        term_id = term_df[term_id_col].iloc[0] if term_id_col in term_df.columns else term_name
        category = term_df[cat_col].iloc[0] if cat_col in term_df.columns else "unknown"
        leading_edge = ";".join(ranked_ids[pos] for pos in leading_positions)

        rows.append(
            {
                "term_id": term_id,
                "term_name": term_name,
                "source": source,
                "category": category,
                "set_size": set_size,
                "es": es,
                "nes": nes,
                "p_value": p_value,
                "fdr": np.nan,
                "leading_edge": leading_edge,
            }
        )

    if not rows:
        return pd.DataFrame(columns=_FGSEA_COLS)

    result = pd.DataFrame(rows)
    result = adjust_pvalues(result, p_col="p_value", method=correction, out_col="fdr")
    return result[_FGSEA_COLS].sort_values("fdr", ascending=True).reset_index(drop=True)


def run_dataset_fgsea(
    dataset: BiasDataset,
    score_col: str,
    annotations: AnnotationSet,
    id_col: str = "primary_id",
    min_term_size: int = 3,
    max_term_size: int | None = None,
    n_permutations: int = 1000,
    weight: float = 1.0,
    seed: int | None = None,
    correction: str = "fdr_bh",
) -> pd.DataFrame:
    """Run pre-ranked GSEA using a numeric score column from a dataset."""
    df = dataset.table
    if id_col not in df.columns:
        raise ValueError(
            f"id_col '{id_col}' not found in dataset.table. "
            f"Available columns: {list(df.columns)}"
        )
    if score_col not in df.columns:
        raise ValueError(
            f"score_col '{score_col}' not found in dataset.table. "
            f"Available columns: {list(df.columns)}"
        )

    ranked_scores = pd.Series(
        pd.to_numeric(df[score_col], errors="coerce").to_numpy(),
        index=df[id_col].astype(str),
        name=score_col,
    )
    result = run_fgsea(
        ranked_scores=ranked_scores,
        annotations=annotations,
        min_term_size=min_term_size,
        max_term_size=max_term_size,
        n_permutations=n_permutations,
        weight=weight,
        seed=seed,
        correction=correction,
    )

    if result.empty:
        for col in ("dataset", "score_col"):
            result[col] = pd.Series(dtype=str)
        return result[_DATASET_FGSEA_COLS]

    result = result.copy()
    result.insert(0, "score_col", score_col)
    result.insert(0, "dataset", dataset.name)
    return result[_DATASET_FGSEA_COLS].reset_index(drop=True)
