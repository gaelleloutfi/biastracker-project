"""
features_digest.py
==================

Enzymatic digestion helpers (trypsin by default) and missed-cleavage utilities. 

This module encapsulates the rules used to evaluate/produce tryptic peptides 
and to quantify missed cleavages. It is written to be general enough for both
peptides and full proteins.

EDIT LATER: In this current workflow it is applied to peptide sequences from DiaNN.

Main public API
---------------
- `trypsin_sites(seq: str)-> int`
- `in_silico_peptides(seq: str, mc: int = 0, min_len: int = 6, max_len: int = 65) -> list[str]`
- `missed_cleavages_in_peptide(pep: str) -> int`

Definitions
-----------
- Trypsin rule (ExPASy/PeptideCutter convention):
    cleave after K or R, except when followed by P (proline).
- Missed cleavage (MC) for a peptide: 
    number of internal tryptic sites still present inside the peptide. 

Implementation
--------------
We use `pyteomics.parser` which ships ExPASy-style rules and utilities. 
"""
from __future__ import annotations
from typing import List
from pyteomics import parser

# ExPASy/PeptideCutter-style trypsin rule: K/R not followed by P
TRYPSIN_RULE = parser.expasy_rules["trypsin"]

#u sed for undigested sequences (like whole proteins)
# counts all potential trypsin sites in the given sequence
# tells us how many cuts trypsin would make if we digested this protein
def trypsin_sites(seq: str)-> int:
    """
    Count the theoretical number of trypsin cleavage sites in a sequence.
    We include the terminal position, to match ExPaSy PeptideCutter definition.
    Example: 
        AKR → 2 (after K and after R)
        AKP → 0 (after K only, R is followed by P)
    
    This is the number of positions where trypsin would cut (K/R not before P),
    given the provided sequence. For a peptide (already produced by trypsin), 
    this is often zero. For proteins, this yields the total potential cut count.

    Parameters
    ----------
    seq: str
        Amino acid sequence (letters only). This function is case sensitive

    Returns 
    -------
    int
        Number of trypsin cleavage sites (>=0)
    """
    if not seq:
        return 0

    seq = seq.upper()
    count = 0

    for i, aa in enumerate(seq):
        # if aa is K or R and next aa is NOT P
        if aa in ("K", "R"):
            if i == len(seq) - 1:       # last residue → cleavage allowed
                count += 1
            elif seq[i+1] != "P":       # internal cleavage
                count += 1

    return count

def in_silico_peptides(
        seq: str,
        mc: int =0,
        min_len: int = 6,
        max_len: int =65,
) -> List[str]:
    """
    Generate tryptic peptides from a sequence allowing up to `mc` miscleavages.
    For DiaNN peptide rows (already digested), this function is typically not 
    required.

    Parameters
    ----------
    seq: str
        Input sequence (protein or long peptide)
    mc: int, default =0
        Maximum allowed miscleavages when generating peptides 
    min_len: int, default = 6
        Minimum peptide length to keep
    max_len: int, default = 65
        Maximum peptide length to keep
    """
    if not seq: 
        return []
    peps = parser.cleave(seq.upper(), TRYPSIN_RULE, mc)
    return [p for p in peps if min_len <= len(p) <= max_len]

# used for already digested peptides like diann output
# counts the internal cleavage sites that remain inside the peptide
# use it to assess digestion quality
def missed_cleavages_in_peptide(pep:str)-> int: 
    """
    Count missed cleavages inside a peptide sequence. 

    Definition
    ----------
    MC is the number of internal trypsin sites in the peptide according to
    the trypsin rule (K/R not followed by P). If MC >= 1, it suggests that
    the peptide contains residues where trypsin could have cleaved but did not.

    Note: `pyteomics.parser.num_sites()` reports potential sites within the
    given string; for already-tryptic peptides, internal sites indicate MC.

    Parameters
    ----------
    pep : str
        Peptide sequence (letters only). Case-insensitive.

    Returns
    -------
    int
        Number of missed cleavages (>= 0).
    """
    if not pep:
        return 0
    return int(parser.num_sites(pep.upper(), TRYPSIN_RULE))