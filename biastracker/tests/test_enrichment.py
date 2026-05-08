"""Tests for biastracker.analysis.enrichment."""
from __future__ import annotations

import pandas as pd
import pytest

from biastracker.dataset import AnnotationSet, BiasDataset
from biastracker.analysis.enrichment import run_ora, run_group_ora


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def make_annotation_set(rows: list[dict], name: str = "test_ann") -> AnnotationSet:
    """Build an AnnotationSet from a list of dicts with primary_id/term_name."""
    df = pd.DataFrame(rows)
    if "term_id" not in df.columns:
        df["term_id"] = df["term_name"]
    if "source" not in df.columns:
        df["source"] = "custom"
    if "category" not in df.columns:
        df["category"] = "unknown"
    return AnnotationSet(
        name=name,
        source="custom",
        table=df,
        id_col="primary_id",
        term_col="term_name",
        term_id_col="term_id",
        category_col="category",
    )


def make_dataset(name: str = "ds") -> BiasDataset:
    """
    10-protein dataset:
    - Query group: P001–P005
    - Background group: P006–P010

    Annotation:
    - TermA: P001, P002, P003, P004 (4/5 in query → enriched)
    - TermB: P006, P007, P008, P009 (0/5 in query → depleted)
    - TermC: P001, P006 (split)
    """
    df = pd.DataFrame({
        "primary_id": [f"P{i:03d}" for i in range(1, 11)],
        "level":      ["protein"] * 10,
        "sequence":   ["SEQ"] * 10,
        "group":      ["query"] * 5 + ["bg"] * 5,
        "score":      list(range(10)),
    })
    return BiasDataset(name=name, table=df, level="protein")


def make_annotations() -> AnnotationSet:
    rows = (
        [{"primary_id": f"P{i:03d}", "term_name": "TermA"} for i in [1, 2, 3, 4]]
        + [{"primary_id": f"P{i:03d}", "term_name": "TermB"} for i in [6, 7, 8, 9]]
        + [{"primary_id": "P001", "term_name": "TermC"},
           {"primary_id": "P006", "term_name": "TermC"}]
    )
    return make_annotation_set(rows)


# ---------------------------------------------------------------------------
# run_ora — basic
# ---------------------------------------------------------------------------

class TestRunOra:

    def test_returns_dataframe(self):
        ann = make_annotations()
        result = run_ora(
            query_ids={"P001", "P002", "P003", "P004", "P005"},
            background_ids={f"P{i:03d}" for i in range(1, 11)},
            annotations=ann,
        )
        assert isinstance(result, pd.DataFrame)

    def test_all_expected_columns_present(self):
        ann = make_annotations()
        result = run_ora(
            query_ids={"P001", "P002", "P003"},
            background_ids={f"P{i:03d}" for i in range(1, 11)},
            annotations=ann,
        )
        expected = [
            "term_id", "term_name", "source", "category",
            "query_count", "query_size", "background_count", "background_size",
            "odds_ratio", "p_value", "fdr", "direction", "query_hits",
        ]
        assert list(result.columns) == expected

    def test_term_clearly_enriched_in_query(self):
        """TermA has 4/4 query hits; should be strongly enriched."""
        ann = make_annotations()
        result = run_ora(
            query_ids={"P001", "P002", "P003", "P004", "P005"},
            background_ids={f"P{i:03d}" for i in range(1, 11)},
            annotations=ann,
        )
        row = result[result["term_name"] == "TermA"].iloc[0]
        assert row["odds_ratio"] > 1
        assert row["direction"] == "enriched"
        assert row["p_value"] < 0.05

    def test_term_clearly_depleted_in_query(self):
        """TermB has 0/4 background hits in query; should be depleted."""
        ann = make_annotations()
        result = run_ora(
            query_ids={"P001", "P002", "P003", "P004", "P005"},
            background_ids={f"P{i:03d}" for i in range(1, 11)},
            annotations=ann,
        )
        row = result[result["term_name"] == "TermB"].iloc[0]
        assert row["odds_ratio"] < 1 or row["odds_ratio"] == 0
        assert row["direction"] in {"depleted", "neutral"}

    def test_fdr_column_populated(self):
        ann = make_annotations()
        result = run_ora(
            query_ids={"P001", "P002", "P003", "P004", "P005"},
            background_ids={f"P{i:03d}" for i in range(1, 11)},
            annotations=ann,
        )
        assert "fdr" in result.columns
        assert result["fdr"].notna().all()

    def test_query_hits_semicolon_separated(self):
        ann = make_annotations()
        result = run_ora(
            query_ids={"P001", "P002", "P003", "P004", "P005"},
            background_ids={f"P{i:03d}" for i in range(1, 11)},
            annotations=ann,
        )
        row = result[result["term_name"] == "TermA"].iloc[0]
        hits = set(row["query_hits"].split(";"))
        assert hits == {"P001", "P002", "P003", "P004"}

    def test_query_ids_outside_background_ignored(self):
        """IDs in query but not in background should be silently excluded."""
        ann = make_annotations()
        result = run_ora(
            query_ids={"P001", "P002", "GHOST1", "GHOST2"},
            background_ids={f"P{i:03d}" for i in range(1, 11)},
            annotations=ann,
        )
        # query_size should reflect background-restricted count
        for _, row in result.iterrows():
            assert row["query_size"] == 2  # only P001, P002 are in background

    def test_min_term_size_filters_small_terms(self):
        """A term with only 1 background protein should be skipped."""
        rows = [{"primary_id": "P001", "term_name": "TinyTerm"}]
        ann = make_annotation_set(rows)
        result = run_ora(
            query_ids={"P001", "P002"},
            background_ids={"P001", "P002", "P003"},
            annotations=ann,
            min_term_size=3,
        )
        assert "TinyTerm" not in result["term_name"].values

    def test_term_below_min_size_included_when_threshold_lowered(self):
        rows = [
            {"primary_id": "P001", "term_name": "SmallTerm"},
            {"primary_id": "P002", "term_name": "SmallTerm"},
        ]
        ann = make_annotation_set(rows)
        result = run_ora(
            query_ids={"P001"},
            background_ids={"P001", "P002", "P003"},
            annotations=ann,
            min_term_size=2,
        )
        assert "SmallTerm" in result["term_name"].values

    def test_empty_result_when_no_terms_pass_filter(self):
        rows = [{"primary_id": "P001", "term_name": "TinyTerm"}]
        ann = make_annotation_set(rows)
        result = run_ora(
            query_ids={"P001"},
            background_ids={"P001", "P002", "P003"},
            annotations=ann,
            min_term_size=5,
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_query_count_and_background_count_correct(self):
        ann = make_annotations()
        result = run_ora(
            query_ids={"P001", "P002", "P003", "P004", "P005"},
            background_ids={f"P{i:03d}" for i in range(1, 11)},
            annotations=ann,
        )
        row_a = result[result["term_name"] == "TermA"].iloc[0]
        assert row_a["query_count"] == 4
        assert row_a["background_count"] == 4
        assert row_a["query_size"] == 5
        assert row_a["background_size"] == 10

    def test_source_and_category_propagated(self):
        rows = [
            {"primary_id": "P001", "term_name": "GO:0001",
             "term_id": "GO:0001", "source": "GO", "category": "BP"},
            {"primary_id": "P002", "term_name": "GO:0001",
             "term_id": "GO:0001", "source": "GO", "category": "BP"},
            {"primary_id": "P003", "term_name": "GO:0001",
             "term_id": "GO:0001", "source": "GO", "category": "BP"},
        ]
        ann = make_annotation_set(rows, name="go_ann")
        # Override source on the AnnotationSet
        ann.source = "GO"
        result = run_ora(
            query_ids={"P001", "P002"},
            background_ids={"P001", "P002", "P003", "P004"},
            annotations=ann,
            min_term_size=2,
        )
        assert result.iloc[0]["category"] == "BP"

    def test_direction_neutral_when_odds_ratio_one(self):
        """When annotated fraction is the same in query and background → neutral."""
        rows = [
            {"primary_id": "P001", "term_name": "BalancedTerm"},
            {"primary_id": "P002", "term_name": "BalancedTerm"},
            {"primary_id": "P003", "term_name": "BalancedTerm"},
            {"primary_id": "P004", "term_name": "BalancedTerm"},
        ]
        ann = make_annotation_set(rows)
        # query = P001, P002; background = P001–P004
        # All 4 background IDs annotated; a=2, b=0, c=2, d=0 → OR=1
        result = run_ora(
            query_ids={"P001", "P002"},
            background_ids={"P001", "P002", "P003", "P004"},
            annotations=ann,
            min_term_size=3,
        )
        row = result[result["term_name"] == "BalancedTerm"].iloc[0]
        assert row["direction"] == "neutral"


# ---------------------------------------------------------------------------
# run_group_ora
# ---------------------------------------------------------------------------

class TestRunGroupOra:

    def test_returns_dataframe(self):
        ds = make_dataset()
        ann = make_annotations()
        result = run_group_ora(ds, group_col="group", query_group="query", annotations=ann)
        assert isinstance(result, pd.DataFrame)

    def test_metadata_columns_present(self):
        ds = make_dataset()
        ann = make_annotations()
        result = run_group_ora(ds, group_col="group", query_group="query", annotations=ann)
        assert "dataset" in result.columns
        assert "group_col" in result.columns
        assert "query_group" in result.columns

    def test_metadata_values_correct(self):
        ds = make_dataset(name="my_experiment")
        ann = make_annotations()
        result = run_group_ora(ds, group_col="group", query_group="query", annotations=ann)
        assert (result["dataset"] == "my_experiment").all()
        assert (result["group_col"] == "group").all()
        assert (result["query_group"] == "query").all()

    def test_enriched_term_detected_with_background_all(self):
        ds = make_dataset()
        ann = make_annotations()
        result = run_group_ora(
            ds, group_col="group", query_group="query",
            annotations=ann, background="all",
        )
        row = result[result["term_name"] == "TermA"].iloc[0]
        assert row["direction"] == "enriched"
        assert row["p_value"] < 0.05

    def test_enriched_term_detected_with_background_other(self):
        ds = make_dataset()
        ann = make_annotations()
        result = run_group_ora(
            ds, group_col="group", query_group="query",
            annotations=ann, background="other",
        )
        row = result[result["term_name"] == "TermA"].iloc[0]
        assert row["direction"] == "enriched"

    def test_fdr_column_present(self):
        ds = make_dataset()
        ann = make_annotations()
        result = run_group_ora(ds, group_col="group", query_group="query", annotations=ann)
        assert "fdr" in result.columns
        assert result["fdr"].notna().all()

    def test_query_hits_present(self):
        ds = make_dataset()
        ann = make_annotations()
        result = run_group_ora(ds, group_col="group", query_group="query", annotations=ann)
        assert "query_hits" in result.columns

    def test_missing_group_col_raises_value_error(self):
        ds = make_dataset()
        ann = make_annotations()
        with pytest.raises(ValueError, match="group_col 'bad_col' not found"):
            run_group_ora(ds, group_col="bad_col", query_group="query", annotations=ann)

    def test_missing_id_col_raises_value_error(self):
        ds = make_dataset()
        ann = make_annotations()
        with pytest.raises(ValueError, match="id_col 'ghost_id' not found"):
            run_group_ora(
                ds, group_col="group", query_group="query",
                annotations=ann, id_col="ghost_id",
            )

    def test_full_column_order(self):
        ds = make_dataset()
        ann = make_annotations()
        result = run_group_ora(ds, group_col="group", query_group="query", annotations=ann)
        expected_start = ["dataset", "group_col", "query_group"]
        assert list(result.columns[:3]) == expected_start

    def test_odds_ratio_column_present_and_positive(self):
        ds = make_dataset()
        ann = make_annotations()
        result = run_group_ora(ds, group_col="group", query_group="query", annotations=ann)
        assert "odds_ratio" in result.columns
        assert (result["odds_ratio"] >= 0).all()
