import numpy as np
import pandas as pd
import pytest
from biastracker.dataset import BiasDataset
from biastracker.analysis.compare import (
    compare_groups,
    compare_multiple_groups,
    compare_datasets,
    feature_significance,
)


def _mk(name, length_vals):
    """Minimal protein dataset carrying a 'length' panel feature."""
    n = len(length_vals)
    return BiasDataset(
        name=name,
        table=pd.DataFrame({
            "primary_id": [f"P{i}" for i in range(n)],
            "level": ["protein"] * n,
            "sequence": ["SEQ"] * n,
            "length": list(length_vals),
        }),
        level="protein",
    )


def test_feature_significance_two_datasets_reports_p_and_fdr():
    a = _mk("A", [10, 12, 11, 13, 10, 12])
    b = _mk("B", [40, 42, 41, 43, 40, 44])
    sig = feature_significance([a, b], "length")
    assert sig["test"] == "Mann-Whitney U"
    assert sig["n_groups"] == 2
    assert 0.0 <= sig["p_value"] <= 1.0
    # 'length' is in the standard panel -> FDR is reported.
    assert sig["fdr"] is not None


def test_feature_significance_three_datasets_kruskal():
    a = _mk("A", [10, 11, 12, 13])
    b = _mk("B", [40, 41, 42, 43])
    c = _mk("C", [90, 91, 92, 93])
    sig = feature_significance([a, b, c], "length")
    assert sig["test"] == "Kruskal-Wallis"
    assert sig["n_groups"] == 3


def test_feature_significance_non_panel_feature_has_no_fdr():
    a = _mk("A", [1, 2, 3, 4]); a.table["paxdb_log10_ppm"] = [1.0, 2.0, 3.0, 4.0]
    b = _mk("B", [1, 2, 3, 4]); b.table["paxdb_log10_ppm"] = [5.0, 6.0, 7.0, 8.0]
    sig = feature_significance([a, b], "paxdb_log10_ppm")
    assert sig is not None
    assert sig["fdr"] is None                      # outside the panel -> nominal p only
    assert 0.0 <= sig["p_value"] <= 1.0


def test_feature_significance_needs_two_datasets_with_feature():
    a = _mk("A", [1, 2, 3])
    b = _mk("B", [4, 5, 6])
    b.table = b.table.drop(columns=["length"])     # only A has 'length'
    assert feature_significance([a, b], "length") is None
    assert feature_significance([a], "length") is None


def test_feature_significance_mixed_levels_returns_none():
    a = _mk("A", [1, 2, 3])
    b = _mk("B", [4, 5, 6])
    object.__setattr__(b, "level", "peptide")      # force a level mismatch
    assert feature_significance([a, b], "length") is None


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

def make_dataset(name="test_ds"):
    """Two-group dataset with two numeric features."""
    df = pd.DataFrame({
        "primary_id": list("ABCDEFGH"),
        "level":      ["protein"] * 8,
        "sequence":   ["SEQ"] * 8,
        "group":      ["A", "A", "A", "A", "B", "B", "B", "B"],
        "score":      [1.0, 2.0, 3.0, 4.0, 10.0, 11.0, 12.0, 13.0],
        "weight":     [5.0, 6.0, 7.0, 8.0, 50.0, 60.0, 70.0, 80.0],
        "length":     [10.0, 20.0, 30.0, 40.0, 100.0, 110.0, 120.0, 130.0],
        "mw":         [100.0, 200.0, 300.0, 400.0, 1000.0, 1100.0, 1200.0, 1300.0],
    })
    return BiasDataset(name=name, table=df, level="protein")


def make_multi_dataset(name="multi_ds"):
    """Three-group dataset."""
    df = pd.DataFrame({
        "primary_id": list("ABCDEFGHI"),
        "level":      ["protein"] * 9,
        "sequence":   ["SEQ"] * 9,
        "group":      ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
        "score":      [1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 100.0, 101.0, 102.0],
    })
    return BiasDataset(name=name, table=df, level="protein")


# ---------------------------------------------------------------------------
# compare_groups
# ---------------------------------------------------------------------------

class TestCompareGroups:
    def test_basic_two_group_comparison(self):
        ds = make_dataset()
        result = compare_groups(ds, group_col="group", group_a="A", group_b="B", features=["score"])

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        row = result.iloc[0]

        assert row["dataset"] == "test_ds"
        assert row["group_col"] == "group"
        assert row["group_a"] == "A"
        assert row["group_b"] == "B"
        assert row["feature"] == "score"

        assert row["n_a"] == 4
        assert row["n_b"] == 4
        assert np.isclose(row["mean_a"], 2.5)
        assert np.isclose(row["mean_b"], 11.5)
        assert np.isclose(row["median_a"], 2.5)
        assert np.isclose(row["median_b"], 11.5)
        assert np.isclose(row["delta_median"], 2.5 - 11.5)

        assert row["direction"] == "higher_in_B"

        assert not np.isnan(row["mannwhitney_statistic"])
        assert not np.isnan(row["mannwhitney_p"])
        assert row["mannwhitney_p"] < 0.05
        assert not np.isnan(row["ks_p"])

    def test_all_expected_columns_present(self):
        ds = make_dataset()
        result = compare_groups(ds, group_col="group", group_a="A", group_b="B", features=["score"])
        expected_cols = [
            "dataset", "group_col", "group_a", "group_b", "feature",
            "n_a", "n_b", "mean_a", "mean_b", "median_a", "median_b",
            "delta_median", "direction",
            "mannwhitney_statistic", "mannwhitney_p", "mannwhitney_fdr",
            "ks_statistic", "ks_p", "ks_fdr",
        ]
        assert list(result.columns) == expected_cols

    def test_fdr_columns_exist_and_are_numeric(self):
        ds = make_dataset()
        result = compare_groups(ds, group_col="group", group_a="A", group_b="B")
        # Both features should produce non-NaN FDR values
        assert "mannwhitney_fdr" in result.columns
        assert "ks_fdr" in result.columns
        assert result["mannwhitney_fdr"].notna().all()
        assert result["ks_fdr"].notna().all()

    def test_multiple_features(self):
        ds = make_dataset()
        result = compare_groups(ds, group_col="group", group_a="A", group_b="B")
        assert len(result) == 2  # curated defaults present in this fixture
        assert set(result["feature"]) == {"length", "mw"}

    def test_missing_feature_is_ignored(self):
        ds = make_dataset()
        result = compare_groups(
            ds,
            group_col="group",
            group_a="A",
            group_b="B",
            features=["score", "nonexistent_feature"],
        )
        # nonexistent_feature is silently skipped
        assert len(result) == 1
        assert result.iloc[0]["feature"] == "score"

    def test_missing_group_col_raises_value_error(self):
        ds = make_dataset()
        with pytest.raises(ValueError, match="group_col 'no_such_col' not found"):
            compare_groups(ds, group_col="no_such_col", group_a="A", group_b="B")

    def test_group_typo_raises_value_error(self):
        ds = make_dataset()
        with pytest.raises(ValueError, match="group value"):
            compare_groups(ds, group_col="group", group_a="A", group_b="Typo")

    def test_empty_result_when_no_valid_features(self):
        ds = make_dataset()
        result = compare_groups(
            ds,
            group_col="group",
            group_a="A",
            group_b="B",
            features=["nonexistent"],
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_direction_column_values(self):
        ds = make_dataset()
        result = compare_groups(ds, group_col="group", group_a="A", group_b="B", features=["score"])
        # Group B has higher scores
        assert result.iloc[0]["direction"] == "higher_in_B"

        result2 = compare_groups(ds, group_col="group", group_a="B", group_b="A", features=["score"])
        assert result2.iloc[0]["direction"] == "higher_in_B"

    def test_dataset_name_propagated(self):
        ds = make_dataset(name="my_experiment")
        result = compare_groups(ds, group_col="group", group_a="A", group_b="B", features=["score"])
        assert (result["dataset"] == "my_experiment").all()

    def test_nan_values_handled_gracefully(self):
        df = pd.DataFrame({
            "primary_id": list("ABCDEF"),
            "level":      ["protein"] * 6,
            "sequence":   ["SEQ"] * 6,
            "group":      ["A", "A", "A", "B", "B", "B"],
            "score":      [1.0, np.nan, 3.0, 10.0, np.nan, 12.0],
        })
        ds = BiasDataset(name="nan_ds", table=df, level="protein")
        result = compare_groups(ds, group_col="group", group_a="A", group_b="B", features=["score"])
        assert not np.isnan(result.iloc[0]["mannwhitney_p"])
        assert result.iloc[0]["n_a"] == 2
        assert result.iloc[0]["n_b"] == 2


# ---------------------------------------------------------------------------
# compare_multiple_groups
# ---------------------------------------------------------------------------

class TestCompareMultipleGroups:
    def test_basic_multi_group_comparison(self):
        ds = make_multi_dataset()
        result = compare_multiple_groups(ds, group_col="group", features=["score"])

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        row = result.iloc[0]

        assert row["dataset"] == "multi_ds"
        assert row["group_col"] == "group"
        assert row["feature"] == "score"
        assert row["n_groups"] == 3
        assert not np.isnan(row["kruskal_statistic"])
        assert not np.isnan(row["kruskal_p"])
        assert row["kruskal_p"] < 0.05

    def test_all_expected_columns_present(self):
        ds = make_multi_dataset()
        result = compare_multiple_groups(ds, group_col="group")
        expected_cols = [
            "dataset", "group_col", "feature", "n_groups", "groups",
            "kruskal_statistic", "kruskal_p", "kruskal_fdr",
        ]
        assert list(result.columns) == expected_cols

    def test_fdr_column_exists_and_is_populated(self):
        ds = make_multi_dataset()
        result = compare_multiple_groups(ds, group_col="group")
        assert "kruskal_fdr" in result.columns
        assert result["kruskal_fdr"].notna().all()

    def test_missing_group_col_raises_value_error(self):
        ds = make_multi_dataset()
        with pytest.raises(ValueError, match="group_col 'bad_col' not found"):
            compare_multiple_groups(ds, group_col="bad_col")

    def test_groups_field_contains_all_group_names(self):
        ds = make_multi_dataset()
        result = compare_multiple_groups(ds, group_col="group", features=["score"])
        groups_str = result.iloc[0]["groups"]
        assert "A" in groups_str
        assert "B" in groups_str
        assert "C" in groups_str

    def test_missing_feature_is_ignored(self):
        ds = make_multi_dataset()
        result = compare_multiple_groups(
            ds, group_col="group", features=["score", "ghost_feature"]
        )
        assert len(result) == 1
        assert result.iloc[0]["feature"] == "score"

    def test_small_groups_excluded_from_n_groups(self):
        """Groups with fewer than 2 valid values should not count."""
        df = pd.DataFrame({
            "primary_id": list("ABCDE"),
            "level":      ["protein"] * 5,
            "sequence":   ["SEQ"] * 5,
            "group":      ["A", "A", "A", "B", "C"],  # B and C have 1 member each
            "score":      [1.0, 2.0, 3.0, 10.0, 20.0],
        })
        ds = BiasDataset(name="small_ds", table=df, level="protein")
        result = compare_multiple_groups(ds, group_col="group", features=["score"])
        # Only group A has >= 2 valid values; result should have n_groups < 2
        # and kruskal_p should be NaN (cannot run with < 2 valid groups)
        assert result.iloc[0]["n_groups"] == 1
        assert np.isnan(result.iloc[0]["kruskal_p"])


# ---------------------------------------------------------------------------
# compare_datasets
# ---------------------------------------------------------------------------

def make_ds_a(name="dataset_A"):
    """Small protein dataset with score and weight."""
    df = pd.DataFrame({
        "primary_id": list("ABCD"),
        "level":      ["protein"] * 4,
        "sequence":   ["SEQ"] * 4,
        "score":      [1.0, 2.0, 3.0, 4.0],
        "weight":     [10.0, 20.0, 30.0, 40.0],
        "length":     [10.0, 20.0, 30.0, 40.0],
        "mw":         [100.0, 200.0, 300.0, 400.0],
    })
    return BiasDataset(name=name, table=df, level="protein")


def make_ds_b(name="dataset_B"):
    """Small protein dataset with score and weight (different values) and an extra column."""
    df = pd.DataFrame({
        "primary_id": list("EFGH"),
        "level":      ["protein"] * 4,
        "sequence":   ["SEQ"] * 4,
        "score":      [100.0, 110.0, 120.0, 130.0],
        "weight":     [200.0, 210.0, 220.0, 230.0],
        "length":     [100.0, 110.0, 120.0, 130.0],
        "mw":         [1000.0, 1100.0, 1200.0, 1300.0],
        "extra_only_b": [1.0, 2.0, 3.0, 4.0],  # not in ds_a
    })
    return BiasDataset(name=name, table=df, level="protein")


class TestCompareDatasets:
    def test_basic_comparison_returns_dataframe(self):
        result = compare_datasets(make_ds_a(), make_ds_b())
        assert isinstance(result, pd.DataFrame)

    def test_one_row_per_shared_feature(self):
        result = compare_datasets(make_ds_a(), make_ds_b())
        # shared curated default features: length, mw (extra_only_b excluded)
        assert len(result) == 2
        assert set(result["feature"]) == {"length", "mw"}

    def test_comparison_type_column(self):
        result = compare_datasets(make_ds_a(), make_ds_b())
        assert "comparison_type" in result.columns
        assert (result["comparison_type"] == "between_datasets").all()

    def test_group_col_is_dataset(self):
        result = compare_datasets(make_ds_a(), make_ds_b())
        assert (result["group_col"] == "dataset").all()

    def test_group_a_and_b_are_dataset_names(self):
        ds_a = make_ds_a(name="exp_A")
        ds_b = make_ds_b(name="exp_B")
        result = compare_datasets(ds_a, ds_b)
        assert (result["group_a"] == "exp_A").all()
        assert (result["group_b"] == "exp_B").all()

    def test_all_compare_cols_present(self):
        result = compare_datasets(make_ds_a(), make_ds_b())
        expected_cols = [
            "dataset", "group_col", "group_a", "group_b", "feature",
            "n_a", "n_b", "mean_a", "mean_b", "median_a", "median_b",
            "delta_median", "direction",
            "mannwhitney_statistic", "mannwhitney_p", "mannwhitney_fdr",
            "ks_statistic", "ks_p", "ks_fdr",
            "comparison_type",
        ]
        assert list(result.columns) == expected_cols

    def test_fdr_columns_populated(self):
        result = compare_datasets(make_ds_a(), make_ds_b())
        assert result["mannwhitney_fdr"].notna().all()
        assert result["ks_fdr"].notna().all()

    def test_significantly_different_scores(self):
        # ds_a scores [1-4], ds_b scores [100-130] — should be very significant
        result = compare_datasets(make_ds_a(), make_ds_b(), features=["score"])
        row = result.iloc[0]
        assert row["mannwhitney_p"] < 0.05
        assert row["direction"] == "higher_in_dataset_B"

    def test_explicit_features_restricts_output(self):
        result = compare_datasets(make_ds_a(), make_ds_b(), features=["score"])
        assert len(result) == 1
        assert result.iloc[0]["feature"] == "score"

    def test_feature_absent_from_one_dataset_is_excluded(self):
        # extra_only_b is only in ds_b; it should not appear in result
        result = compare_datasets(make_ds_a(), make_ds_b())
        assert "extra_only_b" not in result["feature"].values

    def test_no_shared_features_raises_value_error(self):
        ds_a = make_ds_a()  # has: score, weight
        # Build a dataset with no overlapping numeric feature
        df_c = pd.DataFrame({
            "primary_id": ["X", "Y", "Z"],
            "level":      ["protein"] * 3,
            "sequence":   ["SEQ"] * 3,
            "only_c":     [5.0, 6.0, 7.0],
        })
        ds_c = BiasDataset(name="dataset_C", table=df_c, level="protein")
        with pytest.raises(ValueError, match="No shared numeric features"):
            compare_datasets(ds_a, ds_c)

    def test_explicit_feature_not_shared_raises_value_error(self):
        # 'extra_only_b' is not in ds_a, so requesting it explicitly should raise
        with pytest.raises(ValueError, match="No shared numeric features"):
            compare_datasets(make_ds_a(), make_ds_b(), features=["extra_only_b"])

    def test_identical_names_raises_value_error(self):
        ds_a = make_ds_a(name="same_name")
        ds_b = make_ds_b(name="same_name")
        with pytest.raises(ValueError, match="same_name"):
            compare_datasets(ds_a, ds_b)

    def test_level_mismatch_raises_value_error(self):
        ds_a = make_ds_a()
        df_b = make_ds_b().table.copy()
        df_b["level"] = "peptide"
        ds_b = BiasDataset(name="dataset_B", table=df_b, level="peptide")
        with pytest.raises(ValueError, match="does not match"):
            compare_datasets(ds_a, ds_b)

    def test_unequal_sizes_are_handled(self):
        """Datasets with different numbers of rows should work fine."""
        df_small = pd.DataFrame({
            "primary_id": ["A", "B"],
            "level":      ["protein"] * 2,
            "sequence":   ["SEQ"] * 2,
            "score":      [1.0, 2.0],
        })
        ds_small = BiasDataset(name="small", table=df_small, level="protein")
        result = compare_datasets(ds_small, make_ds_b(), features=["score"])
        assert len(result) == 1
        assert result.iloc[0]["n_a"] == 2
        assert result.iloc[0]["n_b"] == 4

    def test_original_datasets_are_not_mutated(self):
        """compare_datasets must not add __dataset_group to either source table."""
        ds_a = make_ds_a()
        ds_b = make_ds_b()
        _ = compare_datasets(ds_a, ds_b)
        assert "__dataset_group" not in ds_a.table.columns
        assert "__dataset_group" not in ds_b.table.columns
