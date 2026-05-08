"""Tests for biastracker.analysis.term_properties."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from biastracker.dataset import AnnotationSet, BiasDataset
from biastracker.analysis.term_properties import (
    summarize_term_properties,
    compare_term_vs_rest,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def make_dataset(name: str = "ds") -> BiasDataset:
    """
    12-protein dataset with two numeric features.

    Groups (by annotation term):
      TermA  : P001–P006  (score ~ 1-6,  weight ~ 10-60)
      TermB  : P007–P012  (score ~ 70-75, weight ~ 100-150)
    No protein belongs to both terms.
    """
    df = pd.DataFrame({
        "primary_id": [f"P{i:03d}" for i in range(1, 13)],
        "level":      ["protein"] * 12,
        "sequence":   ["SEQ"] * 12,
        "score":      [1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
                       70.0, 71.0, 72.0, 73.0, 74.0, 75.0],
        "weight":     [10.0, 20.0, 30.0, 40.0, 50.0, 60.0,
                       100.0, 110.0, 120.0, 130.0, 140.0, 150.0],
    })
    return BiasDataset(name=name, table=df, level="protein")


def make_annotations() -> AnnotationSet:
    rows = (
        [{"primary_id": f"P{i:03d}", "term_name": "TermA",
          "term_id": "TA", "source": "custom", "category": "CatX"}
         for i in range(1, 7)]
        + [{"primary_id": f"P{i:03d}", "term_name": "TermB",
            "term_id": "TB", "source": "custom", "category": "CatY"}
           for i in range(7, 13)]
    )
    df = pd.DataFrame(rows)
    return AnnotationSet(
        name="test_ann",
        source="custom",
        table=df,
        id_col="primary_id",
        term_col="term_name",
        term_id_col="term_id",
        category_col="category",
    )


# ---------------------------------------------------------------------------
# summarize_term_properties
# ---------------------------------------------------------------------------

class TestSummarizeTermProperties:

    def test_returns_dataframe(self):
        result = summarize_term_properties(make_dataset(), make_annotations())
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self):
        result = summarize_term_properties(make_dataset(), make_annotations())
        expected = ["dataset", "term_id", "term_name", "source", "category",
                    "feature", "n", "mean", "median", "std", "min", "max"]
        assert list(result.columns) == expected

    def test_one_row_per_term_per_feature(self):
        result = summarize_term_properties(make_dataset(), make_annotations())
        # 2 terms × 2 features = 4 rows
        assert len(result) == 4
        assert set(result["term_name"]) == {"TermA", "TermB"}
        assert set(result["feature"]) == {"score", "weight"}

    def test_statistics_correct_for_term_a(self):
        result = summarize_term_properties(make_dataset(), make_annotations(), features=["score"])
        row = result[(result["term_name"] == "TermA") & (result["feature"] == "score")].iloc[0]
        assert row["n"] == 6
        assert np.isclose(row["mean"], np.mean([1, 2, 3, 4, 5, 6]))
        assert np.isclose(row["median"], np.median([1, 2, 3, 4, 5, 6]))
        assert np.isclose(row["min"], 1.0)
        assert np.isclose(row["max"], 6.0)

    def test_statistics_correct_for_term_b(self):
        result = summarize_term_properties(make_dataset(), make_annotations(), features=["score"])
        row = result[(result["term_name"] == "TermB") & (result["feature"] == "score")].iloc[0]
        assert row["n"] == 6
        assert np.isclose(row["median"], np.median([70, 71, 72, 73, 74, 75]))

    def test_dataset_name_propagated(self):
        ds = make_dataset(name="my_exp")
        result = summarize_term_properties(ds, make_annotations())
        assert (result["dataset"] == "my_exp").all()

    def test_term_id_and_category_propagated(self):
        result = summarize_term_properties(make_dataset(), make_annotations())
        row_a = result[result["term_name"] == "TermA"].iloc[0]
        assert row_a["term_id"] == "TA"
        assert row_a["category"] == "CatX"

    def test_min_term_size_filters_small_terms(self):
        """A term with only 2 proteins should be excluded when min_term_size=3."""
        rows = (
            [{"primary_id": f"P{i:03d}", "term_name": "TermA",
              "term_id": "TA", "source": "custom", "category": "X"}
             for i in range(1, 7)]
            + [{"primary_id": "P007", "term_name": "TinyTerm",
                "term_id": "TT", "source": "custom", "category": "X"},
               {"primary_id": "P008", "term_name": "TinyTerm",
                "term_id": "TT", "source": "custom", "category": "X"}]
        )
        ann = AnnotationSet(
            name="ann", source="custom",
            table=pd.DataFrame(rows),
            id_col="primary_id", term_col="term_name",
            term_id_col="term_id", category_col="category",
        )
        result = summarize_term_properties(make_dataset(), ann, min_term_size=3)
        assert "TinyTerm" not in result["term_name"].values

    def test_explicit_features_restricts_output(self):
        result = summarize_term_properties(
            make_dataset(), make_annotations(), features=["score"]
        )
        assert set(result["feature"]) == {"score"}
        assert len(result) == 2  # 2 terms × 1 feature

    def test_missing_id_col_raises_value_error(self):
        ds = make_dataset()
        ann = make_annotations()
        with pytest.raises(ValueError, match="id_col 'ghost' not found"):
            summarize_term_properties(ds, ann, id_col="ghost")

    def test_empty_result_when_no_terms_pass_filter(self):
        rows = [{"primary_id": "P001", "term_name": "Tiny",
                 "term_id": "T", "source": "custom", "category": "X"}]
        ann = AnnotationSet(
            name="ann", source="custom",
            table=pd.DataFrame(rows),
            id_col="primary_id", term_col="term_name",
            term_id_col="term_id", category_col="category",
        )
        result = summarize_term_properties(make_dataset(), ann, min_term_size=10)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_nan_values_in_feature_handled(self):
        df = pd.DataFrame({
            "primary_id": [f"P{i:03d}" for i in range(1, 7)],
            "level":      ["protein"] * 6,
            "sequence":   ["SEQ"] * 6,
            "score":      [1.0, np.nan, 3.0, np.nan, 5.0, 6.0],
        })
        ds = BiasDataset(name="nan_ds", table=df, level="protein")
        rows = [{"primary_id": f"P{i:03d}", "term_name": "TermA",
                 "term_id": "TA", "source": "custom", "category": "X"}
                for i in range(1, 7)]
        ann = AnnotationSet(
            name="ann", source="custom",
            table=pd.DataFrame(rows),
            id_col="primary_id", term_col="term_name",
            term_id_col="term_id", category_col="category",
        )
        result = summarize_term_properties(ds, ann, features=["score"])
        row = result.iloc[0]
        assert row["n"] == 4   # 4 non-NaN values
        assert not np.isnan(row["mean"])


# ---------------------------------------------------------------------------
# compare_term_vs_rest
# ---------------------------------------------------------------------------

class TestCompareTermVsRest:

    def test_returns_dataframe(self):
        result = compare_term_vs_rest(make_dataset(), make_annotations())
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self):
        result = compare_term_vs_rest(make_dataset(), make_annotations())
        expected = [
            "dataset", "term_id", "term_name", "source", "category",
            "feature",
            "n_term", "n_rest",
            "mean_term", "mean_rest",
            "median_term", "median_rest",
            "delta_median", "direction",
            "mannwhitney_statistic", "mannwhitney_p", "mannwhitney_fdr",
            "ks_statistic", "ks_p", "ks_fdr",
        ]
        assert list(result.columns) == expected

    def test_one_row_per_term_per_feature(self):
        result = compare_term_vs_rest(make_dataset(), make_annotations())
        # 2 terms × 2 features = 4 rows
        assert len(result) == 4

    def test_term_a_score_higher_in_rest(self):
        """TermA scores [1-6]; rest (TermB) scores [70-75] → depleted direction."""
        result = compare_term_vs_rest(
            make_dataset(), make_annotations(), features=["score"]
        )
        row = result[(result["term_name"] == "TermA") & (result["feature"] == "score")].iloc[0]
        assert row["direction"] == "higher_in_rest"
        assert row["mannwhitney_p"] < 0.05

    def test_term_b_score_higher_in_term(self):
        """TermB scores [70-75] vs rest [1-6] → enriched direction."""
        result = compare_term_vs_rest(
            make_dataset(), make_annotations(), features=["score"]
        )
        row = result[(result["term_name"] == "TermB") & (result["feature"] == "score")].iloc[0]
        assert row["direction"] == "higher_in_term"
        assert row["mannwhitney_p"] < 0.05

    def test_fdr_columns_populated(self):
        result = compare_term_vs_rest(make_dataset(), make_annotations())
        assert result["mannwhitney_fdr"].notna().all()
        assert result["ks_fdr"].notna().all()

    def test_delta_median_sign_consistent_with_direction(self):
        result = compare_term_vs_rest(
            make_dataset(), make_annotations(), features=["score"]
        )
        for _, row in result.iterrows():
            if row["direction"] == "higher_in_term":
                assert row["delta_median"] > 0
            elif row["direction"] == "higher_in_rest":
                assert row["delta_median"] < 0

    def test_n_term_and_n_rest_correct(self):
        result = compare_term_vs_rest(
            make_dataset(), make_annotations(), features=["score"]
        )
        row = result[result["term_name"] == "TermA"].iloc[0]
        assert row["n_term"] == 6
        assert row["n_rest"] == 6  # 12 total − 6 in TermA

    def test_ks_statistic_not_nan(self):
        result = compare_term_vs_rest(make_dataset(), make_annotations())
        assert result["ks_statistic"].notna().all()

    def test_dataset_name_propagated(self):
        ds = make_dataset(name="proteomics_run_1")
        result = compare_term_vs_rest(ds, make_annotations())
        assert (result["dataset"] == "proteomics_run_1").all()

    def test_term_id_and_category_in_output(self):
        result = compare_term_vs_rest(make_dataset(), make_annotations())
        row_a = result[result["term_name"] == "TermA"].iloc[0]
        assert row_a["term_id"] == "TA"
        assert row_a["category"] == "CatX"

    def test_min_term_size_filters_small_terms(self):
        result = compare_term_vs_rest(
            make_dataset(), make_annotations(), min_term_size=7
        )
        # Both terms have 6 proteins < 7 → no results
        assert len(result) == 0

    def test_explicit_features_restricts_output(self):
        result = compare_term_vs_rest(
            make_dataset(), make_annotations(), features=["weight"]
        )
        assert set(result["feature"]) == {"weight"}

    def test_missing_id_col_raises_value_error(self):
        ds = make_dataset()
        ann = make_annotations()
        with pytest.raises(ValueError, match="id_col 'no_col' not found"):
            compare_term_vs_rest(ds, ann, id_col="no_col")

    def test_empty_result_when_no_terms_qualify(self):
        rows = [{"primary_id": "P001", "term_name": "Tiny",
                 "term_id": "T", "source": "custom", "category": "X"}]
        ann = AnnotationSet(
            name="ann", source="custom",
            table=pd.DataFrame(rows),
            id_col="primary_id", term_col="term_name",
            term_id_col="term_id", category_col="category",
        )
        result = compare_term_vs_rest(make_dataset(), ann, min_term_size=5)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_nan_in_feature_handled_gracefully(self):
        df = pd.DataFrame({
            "primary_id": [f"P{i:03d}" for i in range(1, 13)],
            "level":      ["protein"] * 12,
            "sequence":   ["SEQ"] * 12,
            "score":      [1.0, np.nan, 3.0, np.nan, 5.0, 6.0,
                           70.0, np.nan, 72.0, 73.0, np.nan, 75.0],
        })
        ds = BiasDataset(name="nan_ds", table=df, level="protein")
        result = compare_term_vs_rest(ds, make_annotations(), features=["score"])
        assert result["mannwhitney_p"].notna().all()
