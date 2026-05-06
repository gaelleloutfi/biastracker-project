from protperties.features_digest import (
    trypsin_sites,
    in_silico_peptides,
    missed_cleavages_in_peptide
)


def test_trypsin_sites_simple():
    # K or R not followed by P
    assert trypsin_sites("AKR") == 2      # A|K and K|R - to match ExPaSy PeptideCutter definition
    assert trypsin_sites("AKP") == 0      # only A|K
    assert trypsin_sites("ARP") == 0      # only A|R

    # No K/R
    assert trypsin_sites("AAAA") == 0

    # Single K / R internal, next not P
    assert trypsin_sites("AKA") == 1      # cut after K
    assert trypsin_sites("ARA") == 1      # cut after R

    # Single K / R at C-terminus
    assert trypsin_sites("AAK") == 1      # last is K → cut
    assert trypsin_sites("AAR") == 1      # last is R → cut

    # K/R followed by P → no cut
    assert trypsin_sites("AKP") == 0
    assert trypsin_sites("ARP") == 0
    assert trypsin_sites("KP") == 0
    assert trypsin_sites("RP") == 0

    # Mix of allowed and blocked sites
    assert trypsin_sites("AKRP") == 1     # only after K 
    assert trypsin_sites("AKPR") == 1     # A K P R → only after R

def test_in_silico_peptides_empty():
    assert in_silico_peptides("", mc=0) == []

def test_in_silico_peptides_no_miscleavages():
    # "AKR" cuts -> AK | R
    peptides = in_silico_peptides("AKR", mc=0, min_len =1, max_len=10)
    # assert peptides == ["AK", "R"] <- this gives an error because of the order of the elements
    assert set(peptides) == {"AK", "R"}


def test_in_silico_peptides_allowing_miscleavages():
    # If mc=1, allow joining adjacent tryptic peptides
    peptides = in_silico_peptides("AKR", mc=1, min_len=1, max_len=10)
    # Possible: "A-K", "K-R", "A-K-R"
    # Returned as raw sequences:
    assert "AK" in peptides
    assert "R" in peptides
    assert "AKR" in peptides


def test_missed_cleavages_zero():
    assert missed_cleavages_in_peptide("PEPTIDEK") == 0
    assert missed_cleavages_in_peptide("AAAAA") == 0
    assert missed_cleavages_in_peptide("") == 0
    assert missed_cleavages_in_peptide("AK") == 0

def test_missed_cleavages_positive():
    #here we want to know given this as an observed tryptic product,
    # how many sites were missed - which is why we dont count the terminal as a site
    assert missed_cleavages_in_peptide("AKRPQK") == 1
    assert missed_cleavages_in_peptide("AKR") == 1