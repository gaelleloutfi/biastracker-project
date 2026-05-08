"""Tests for biastracker.annotations.custom."""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from biastracker.dataset import AnnotationSet
from biastracker.annotations.custom import load_long_annotation_table, load_gmt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def write_file(tmp_path: Path, filename: str, content: str) -> Path:
    p = tmp_path / filename
    p.write_text(textwrap.dedent(content), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# load_long_annotation_table
# ---------------------------------------------------------------------------

class TestLoadLongAnnotationTable:

    def test_load_csv(self, tmp_path):
        p = write_file(tmp_path, "ann.csv", """\
            primary_id,term_name
            P001,GO:0001
            P002,GO:0002
            P003,GO:0001
        """)
        ann = load_long_annotation_table(p, name="test_ann")

        assert isinstance(ann, AnnotationSet)
        assert len(ann.table) == 3
        assert set(ann.table["primary_id"]) == {"P001", "P002", "P003"}
        assert set(ann.table["term_name"]) == {"GO:0001", "GO:0002"}

    def test_load_tsv_by_extension(self, tmp_path):
        p = write_file(tmp_path, "ann.tsv", """\
            primary_id\tterm_name
            P001\tGO:0001
            P002\tGO:0002
        """)
        ann = load_long_annotation_table(p, name="test_tsv")

        assert isinstance(ann, AnnotationSet)
        assert len(ann.table) == 2
        assert "P001" in ann.table["primary_id"].values

    def test_load_tsv_by_sniffing(self, tmp_path):
        """A .csv file that is actually tab-separated should be detected."""
        p = write_file(tmp_path, "ann.csv", """\
            primary_id\tterm_name
            P001\tGO:0001
            P002\tGO:0002
        """)
        ann = load_long_annotation_table(p, name="sniff_test")
        assert len(ann.table) == 2

    def test_missing_term_id_is_filled_from_term_name(self, tmp_path):
        """When term_id_col is not provided, term_id should equal term_name."""
        p = write_file(tmp_path, "ann.csv", """\
            primary_id,term_name
            P001,Pathway_A
            P002,Pathway_B
        """)
        ann = load_long_annotation_table(p, name="no_term_id")

        assert "term_id" in ann.table.columns
        assert (ann.table["term_id"] == ann.table["term_name"]).all()

    def test_term_id_col_used_when_present(self, tmp_path):
        p = write_file(tmp_path, "ann.csv", """\
            primary_id,term_name,my_term_id
            P001,Pathway_A,PA001
            P002,Pathway_B,PB002
        """)
        ann = load_long_annotation_table(
            p, name="with_term_id", term_id_col="my_term_id"
        )
        assert list(ann.table["term_id"]) == ["PA001", "PB002"]

    def test_missing_category_defaults_to_unknown(self, tmp_path):
        """When category_col is not provided, category should be 'unknown'."""
        p = write_file(tmp_path, "ann.csv", """\
            primary_id,term_name
            P001,GO:0001
        """)
        ann = load_long_annotation_table(p, name="no_cat")

        assert "category" in ann.table.columns
        assert (ann.table["category"] == "unknown").all()

    def test_category_col_used_when_present(self, tmp_path):
        p = write_file(tmp_path, "ann.csv", """\
            primary_id,term_name,cat
            P001,GO:0001,molecular_function
            P002,GO:0002,biological_process
        """)
        ann = load_long_annotation_table(
            p, name="with_cat", category_col="cat"
        )
        assert set(ann.table["category"]) == {"molecular_function", "biological_process"}

    def test_source_column_set_correctly(self, tmp_path):
        p = write_file(tmp_path, "ann.csv", """\
            primary_id,term_name
            P001,GO:0001
        """)
        ann = load_long_annotation_table(p, name="src_test", source="my_db")
        assert (ann.table["source"] == "my_db").all()

    def test_rows_with_missing_primary_id_are_dropped(self, tmp_path):
        p = write_file(tmp_path, "ann.csv", """\
            primary_id,term_name
            P001,GO:0001
            ,GO:0002
            P003,GO:0003
        """)
        ann = load_long_annotation_table(p, name="drop_id")
        assert len(ann.table) == 2
        assert "" not in ann.table["primary_id"].values

    def test_rows_with_missing_term_name_are_dropped(self, tmp_path):
        p = write_file(tmp_path, "ann.csv", """\
            primary_id,term_name
            P001,GO:0001
            P002,
            P003,GO:0003
        """)
        ann = load_long_annotation_table(p, name="drop_term")
        assert len(ann.table) == 2

    def test_custom_id_col_and_term_col(self, tmp_path):
        p = write_file(tmp_path, "ann.csv", """\
            protein,pathway
            P001,KEGG:001
            P002,KEGG:002
        """)
        ann = load_long_annotation_table(
            p, name="custom_cols", id_col="protein", term_col="pathway"
        )
        assert len(ann.table) == 2
        assert set(ann.table["primary_id"]) == {"P001", "P002"}

    def test_missing_required_column_raises_value_error(self, tmp_path):
        p = write_file(tmp_path, "ann.csv", """\
            primary_id,wrong_col
            P001,GO:0001
        """)
        with pytest.raises(ValueError, match="term_name"):
            load_long_annotation_table(p, name="bad_col")

    def test_output_schema(self, tmp_path):
        p = write_file(tmp_path, "ann.csv", """\
            primary_id,term_name
            P001,GO:0001
        """)
        ann = load_long_annotation_table(p, name="schema_check")
        for col in ("primary_id", "term_id", "term_name", "source", "category"):
            assert col in ann.table.columns


# ---------------------------------------------------------------------------
# load_gmt
# ---------------------------------------------------------------------------

class TestLoadGmt:

    def test_basic_gmt_conversion(self, tmp_path):
        p = write_file(tmp_path, "pathways.gmt", """\
            Pathway_A\tA description\tP001\tP002\tP003
            Pathway_B\tB description\tP004\tP005
        """)
        ann = load_gmt(p, name="test_gmt")

        assert isinstance(ann, AnnotationSet)
        # 3 + 2 = 5 rows
        assert len(ann.table) == 5

    def test_ids_mapped_to_correct_terms(self, tmp_path):
        p = write_file(tmp_path, "pathways.gmt", """\
            Pathway_A\tdesc\tP001\tP002
            Pathway_B\tdesc\tP003
        """)
        ann = load_gmt(p, name="id_map")
        term_for_p1 = ann.table.loc[ann.table["primary_id"] == "P001", "term_name"].iloc[0]
        assert term_for_p1 == "Pathway_A"
        term_for_p3 = ann.table.loc[ann.table["primary_id"] == "P003", "term_name"].iloc[0]
        assert term_for_p3 == "Pathway_B"

    def test_term_id_equals_term_name(self, tmp_path):
        p = write_file(tmp_path, "pathways.gmt", """\
            Pathway_A\tdesc\tP001
        """)
        ann = load_gmt(p, name="term_id_check")
        assert (ann.table["term_id"] == ann.table["term_name"]).all()

    def test_category_default(self, tmp_path):
        p = write_file(tmp_path, "pathways.gmt", """\
            Pathway_A\tdesc\tP001
        """)
        ann = load_gmt(p, name="cat_default")
        assert (ann.table["category"] == "unknown").all()

    def test_custom_category(self, tmp_path):
        p = write_file(tmp_path, "pathways.gmt", """\
            Pathway_A\tdesc\tP001
        """)
        ann = load_gmt(p, name="cat_custom", category="KEGG")
        assert (ann.table["category"] == "KEGG").all()

    def test_source_column(self, tmp_path):
        p = write_file(tmp_path, "pathways.gmt", """\
            Pathway_A\tdesc\tP001
        """)
        ann = load_gmt(p, name="src_check", source="MSigDB")
        assert (ann.table["source"] == "MSigDB").all()

    def test_descriptions_stored_in_metadata(self, tmp_path):
        p = write_file(tmp_path, "pathways.gmt", """\
            Pathway_A\tFirst pathway\tP001
            Pathway_B\tSecond pathway\tP002
        """)
        ann = load_gmt(p, name="desc_check")
        assert "descriptions" in ann.metadata
        assert ann.metadata["descriptions"]["Pathway_A"] == "First pathway"
        assert ann.metadata["descriptions"]["Pathway_B"] == "Second pathway"

    def test_empty_gmt_returns_empty_annotationset(self, tmp_path):
        p = write_file(tmp_path, "empty.gmt", "")
        ann = load_gmt(p, name="empty_gmt")
        assert isinstance(ann, AnnotationSet)
        assert len(ann.table) == 0

    def test_output_schema(self, tmp_path):
        p = write_file(tmp_path, "pathways.gmt", """\
            Pathway_A\tdesc\tP001
        """)
        ann = load_gmt(p, name="schema_gmt")
        for col in ("primary_id", "term_id", "term_name", "source", "category"):
            assert col in ann.table.columns

    def test_gmt_ids_available_via_annotationset_api(self, tmp_path):
        p = write_file(tmp_path, "pathways.gmt", """\
            Pathway_A\tdesc\tP001\tP002
        """)
        ann = load_gmt(p, name="api_check")
        ids = ann.ids_for_term("Pathway_A")
        assert ids == {"P001", "P002"}
