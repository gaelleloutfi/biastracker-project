"""Tests for biastracker.analysis.ranking."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from biastracker.analysis.ranking import (
    CUSTOM,
    MEAN_EXPRESSION,
    compute_mean_expression,
    detect_expression_columns,
    prepare_fgsea_ranking,
)


def _df(**cols) -> pd.DataFrame:
    return pd.DataFrame(cols)


# ---------------------------------------------------------------------------
# detect_expression_columns / compute_mean_expression
# ---------------------------------------------------------------------------

def test_detect_and_mean_expression_from_lfq_columns():
    df = _df(
        primary_id=["P1", "P2"],
        **{"LFQ intensity a": [10.0, 20.0], "LFQ intensity b": [30.0, 40.0]},
        other=[1, 2],
    )
    assert detect_expression_columns(df) == ["LFQ intensity a", "LFQ intensity b"]
    m = compute_mean_expression(df)
    assert list(m) == [20.0, 30.0]  # row means


def test_mean_expression_skips_missing_values():
    df = _df(**{"LFQ intensity a": [10.0, np.nan], "LFQ intensity b": [np.nan, np.nan]})
    m = compute_mean_expression(df)
    assert m.iloc[0] == 10.0          # mean of [10, NaN] -> 10
    assert np.isnan(m.iloc[1])        # all-NaN row -> NaN


def test_mean_expression_coerces_numeric_strings():
    df = _df(**{"LFQ intensity a": ["10", "20"], "LFQ intensity b": ["30", "abc"]})
    m = compute_mean_expression(df)
    assert m.iloc[0] == 20.0
    assert m.iloc[1] == 20.0          # ("20", NaN) -> 20


def test_mean_expression_falls_back_to_precomputed_column():
    df = _df(primary_id=["P1"], mean_lfq=[123.0], gravy=[-0.1])
    m = compute_mean_expression(df)
    assert list(m) == [123.0]


def test_mean_expression_no_columns_raises():
    df = _df(primary_id=["P1"], gravy=[-0.1])  # no LFQ, no precomputed mean
    with pytest.raises(ValueError, match="No LFQ/expression columns"):
        compute_mean_expression(df)


def test_mean_expression_explicit_missing_column_raises():
    df = _df(**{"LFQ intensity a": [1.0]})
    with pytest.raises(ValueError, match="not found"):
        compute_mean_expression(df, columns=["LFQ intensity a", "nope"])


# ---------------------------------------------------------------------------
# prepare_fgsea_ranking
# ---------------------------------------------------------------------------

def test_prepare_ranking_mean_expression_sorted_descending():
    df = _df(
        primary_id=["P1", "P2", "P3"],
        **{"LFQ intensity a": [1.0, 3.0, 2.0]},
    )
    s = prepare_fgsea_ranking(df, "primary_id", MEAN_EXPRESSION)
    assert s.name == MEAN_EXPRESSION
    assert list(s.index) == ["P2", "P3", "P1"]     # descending
    assert list(s.values) == [3.0, 2.0, 1.0]


def test_prepare_ranking_custom_metric():
    df = _df(primary_id=["P1", "P2"], my_metric=[5.0, 9.0])
    s = prepare_fgsea_ranking(df, "primary_id", CUSTOM, custom_col="my_metric")
    assert s.name == "my_metric"
    assert list(s.index) == ["P2", "P1"]


def test_prepare_ranking_custom_non_numeric_rejected():
    df = _df(primary_id=["P1", "P2"], label=["a", "b"])
    with pytest.raises(ValueError, match="no finite numeric values"):
        prepare_fgsea_ranking(df, "primary_id", CUSTOM, custom_col="label")


def test_prepare_ranking_custom_requires_col():
    df = _df(primary_id=["P1", "P2"], x=[1.0, 2.0])
    with pytest.raises(ValueError, match="requires a custom_col"):
        prepare_fgsea_ranking(df, "primary_id", CUSTOM)


def test_prepare_ranking_unknown_method():
    df = _df(primary_id=["P1", "P2"], x=[1.0, 2.0])
    with pytest.raises(ValueError, match="Unknown ranking method"):
        prepare_fgsea_ranking(df, "primary_id", "bogus")


def test_prepare_ranking_drops_infinite_and_missing():
    df = _df(
        primary_id=["P1", "P2", "P3", "P4"],
        my_metric=[np.inf, 2.0, np.nan, 5.0],
    )
    s = prepare_fgsea_ranking(df, "primary_id", CUSTOM, custom_col="my_metric")
    assert list(s.index) == ["P4", "P2"]           # inf and NaN dropped
    assert np.isfinite(s.to_numpy()).all()


def test_prepare_ranking_duplicate_accessions_take_max():
    df = _df(
        primary_id=["P1", "P1", "P2"],
        my_metric=[1.0, 7.0, 3.0],
    )
    s = prepare_fgsea_ranking(df, "primary_id", CUSTOM, custom_col="my_metric")
    assert s.loc["P1"] == 7.0                       # max of duplicates
    assert list(s.index) == ["P1", "P2"]


def test_prepare_ranking_nan_index_dropped():
    df = _df(primary_id=["P1", None, "P3"], my_metric=[1.0, 2.0, 3.0])
    s = prepare_fgsea_ranking(df, "primary_id", CUSTOM, custom_col="my_metric")
    assert "nan" not in s.index
    assert set(s.index) == {"P1", "P3"}


def test_prepare_ranking_too_few_values_raises():
    df = _df(primary_id=["P1"], my_metric=[1.0])
    with pytest.raises(ValueError, match="fewer than two finite values"):
        prepare_fgsea_ranking(df, "primary_id", CUSTOM, custom_col="my_metric")


def test_prepare_ranking_deterministic_tie_order():
    df = _df(
        primary_id=["P1", "P2", "P3"],
        my_metric=[5.0, 5.0, 1.0],
    )
    s1 = prepare_fgsea_ranking(df, "primary_id", CUSTOM, custom_col="my_metric")
    s2 = prepare_fgsea_ranking(df, "primary_id", CUSTOM, custom_col="my_metric")
    # Stable sort => ties keep input order, and repeated calls agree.
    assert list(s1.index) == list(s2.index)
    assert list(s1.index[:2]) == ["P1", "P2"]
