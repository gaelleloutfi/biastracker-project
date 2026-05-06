import math
import pytest

from protperties.features_basic import (
    _clean,
    basic_props,
    amino_acid_composition,
)


def test_clean_removes_noncanonical():
    # X, B, Z, J, *, numbers, spaces should be removed
    assert _clean("ACDXBZJ*12") == "ACD"
    assert _clean("pep tide*") == "PEPTIDE"
    assert _clean("") == ""
    assert _clean(None) == ""  # seq=None → ""


def test_basic_props_length_and_mw():
    props = basic_props("ACDE")
    assert props["length"] == 4
    assert props["mw"] > 300  # molecular weight must be positive


def test_basic_props_pi_is_reasonable():
    props = basic_props("ACDEFGHIKLMNPQRSTVWY")
    assert 2 < props["pi"] < 12  # pI always between 2 and 12


def test_basic_props_gravy_signs():
    # Hydrophobic peptide should have positive GRAVY
    props = basic_props("VVVVVV")
    assert props["gravy"] > 0

    # Hydrophilic peptide
    props2 = basic_props("DEDEDE")
    assert props2["gravy"] < 0


def test_basic_props_stability_index():
    props = basic_props("ACDEFGHIK")
    # Instability index: typically between 0 and ~150
    assert 0 <= props["instability"] <= 150


def test_basic_props_aliphatic_index():
    props = basic_props("AVIL")
    # aliphatic index large for A/V/I/L
    assert props["aliphatic_index"] > 50

def test_basic_props_charge_at_pH_default():
    props = basic_props("ABCDEFGHIK")
    assert "charge_at_pH" in props 
    # net charge should not be absurdly large for a short peptide
    assert -10.0 < props["charge_at_pH"] < 10.0


def test_amino_acid_composition_basic():
    comp = amino_acid_composition("AAAAC")
    assert len(comp) == 20  # All 20 canonical AAs
    assert comp["A"] == 0.8
    assert comp["C"] == 0.2
    assert comp["G"] == 0.0  # Not present
    assert abs(sum(comp.values()) - 1.0) < 1e-9


def test_amino_acid_composition_all_keys_present():
    comp = amino_acid_composition("ACDE")
    assert set(comp.keys()) == set("ACDEFGHIKLMNPQRSTVWY")


def test_amino_acid_composition_empty():
    comp = amino_acid_composition("")
    assert len(comp) == 20
    assert all(v == 0.0 for v in comp.values())


def test_amino_acid_composition_noncanonical_cleaned():
    # X and * are non-canonical; only A and C remain
    comp = amino_acid_composition("AAXC*")
    assert comp["A"] == pytest.approx(2 / 3)
    assert comp["C"] == pytest.approx(1 / 3)
    assert abs(sum(comp.values()) - 1.0) < 1e-9


def test_amino_acid_composition_sums_to_one():
    comp = amino_acid_composition("ACDEFGHIKLMNPQRSTVWY")
    assert abs(sum(comp.values()) - 1.0) < 1e-9
