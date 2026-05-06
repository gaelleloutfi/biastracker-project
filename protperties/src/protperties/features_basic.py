"""
features_basic.py
=================

Core physicochemical computations for peptide/protein sequences. 

This module defines pure fucntions that take an amino-acid sequence 
(letters only) and returns physicochemical descriptors similar to 
ExPASy/ProtParam. These functions do not know anything about DIA-NN,
MaxQuant, or tables - they only operate on strings.

Main public API
---------------
- `basic_props(seq: str) -> dict` 
- `aliphatic_index(seq: str) -> float`

Design notes
------------
- Only the 20 canonical (Classic) amino acids are considered by default.
any other symbols are dropped by `_clean()`

- Computations rely on Biopython's `ProteinAnalysis` for most descriptors,
which implements widely used ProtParam algorithms: 
    - molecular_weight()
    - isoelectric_point()            (Bjellqvist method)
    - gravy()                        (Kyte-Doolittle)
    - instability_index()            (Guruprasad et al.)
    - aromaticity()                  (fraction of F/W/Y)
    - molar_extinction_coefficient() (reduced vs cystine)

+ potentially add references
"""

from __future__ import annotations
from collections import Counter
from typing import Dict

# Biopython implements most of ProtParam's calculations
from Bio.SeqUtils.ProtParam import ProteinAnalysis
#=============Constants=============

#The set if canonical amino acids. Any other letter will be removed by `_clean()`
AA20 = set("ACDEFGHIKLMNPQRSTVWY")

# Ikai (1980) aliphatic index weights: 
# AI = 100 * (xAla + 2.9*xVal + 3.9*(xIle + xLeu))

# where xX are mole fractions in the sequence. (so 100*xX will be the mole 
#percent of the amino acid X)

# 2.9 and 3.9 are respectively the relative volume of valine and chain and 
#leu/ile side chains to the side chain of alanine

IKAI_VAL = 2.9 #weight for val
IKAI_ILE_LEU = 3.9 # weight for Ile and Leu 

#=============Helpers=============
def _clean(seq: str) -> str:
    """
    Return an uppercase sequence restricted to the 20 canonical amino acids: 
    This function : 
    - uppercases the input 
    - strips any trailing stop ('*')
    - removes any non-canonical symbols (X, J ...)

    Parameters
    ----------
    seq:str
        Raw sequence that may contain non standard symbols. 

    Returns
    -------
    str
        Cleaned sequence containing only letters from AA20

    Notes
    -----
    cleaning ensures deterministic behavior for all downstream metrics. 
    if the cleaned sequence is empty, callers should handle that case explicitly. 
    """
    s = (seq or "").upper().rstrip("*")
    return "".join(a for a in s if a in AA20)

def _empty_props() -> Dict[str, float]:
    """
    Returns a zero-filled dictionary for the standard property keys. 
    Used when the cleaned sequence is empty (length = 0). 
    """
    return {
        "length":0,
        "mw":0.0,
        "pi":0.0,
        "gravy":0.0,
        "instability":0.0,
        "aromaticity":0.0,
        "aliphatic_index":0.0,
        "ext_reduced":0,
        "ext_cystine":0,
        "charge_at_pH":0.0,
        
    }

#=============public API=============
def aliphatic_index(seq: str)-> float: 
    """
    Compute the aliphatic index (Ikai, 1980).

    Formula 
    -------
    Let xA, xV, xI, xL be the mole fractions of ala, val, ile, leu. 
    the aliphatic index is: 
        AI = 100 * ( xA + 2.9 * xV + 3.9 * (xI + xL) ) 

    Parameters
    ----------
    seq:str
        Amino acid sequence (letters only). Non-canonical symbols will be removed.

    Returns
    -------
    float
        Aliphatic index (unitless). 0.0 if the cleaned sequence is empty. 

    Note
    -----
    Although the index was made for proteins, it still remains informative 
    for peptides as a proxy for aliphatic side-chain content. 
    """
    s= _clean(seq)
    L=len(s)
    if L ==0: 
        return 0.0
    
    c = Counter(s)
    xA = c.get("A",0)/L
    xV = c.get("V",0)/L
    xI = c.get("I",0)/L
    xL = c.get("L",0)/L
    return 100.0 * (xA + IKAI_VAL * xV + IKAI_ILE_LEU * (xI + xL))

def amino_acid_composition(seq: str) -> Dict[str, float]:
    """
    Calculate the amino acid composition (fraction) for each of the 20 canonical amino acids.

    Parameters
    ----------
    seq : str
        Amino acid sequence (letters only). Non-canonical symbols will be removed.

    Returns
    -------
    Dict[str, float]
        Dictionary mapping each amino acid (single-letter code) to its fraction in the sequence.
        Amino acids not present have a value of 0.0.
    """
    s = _clean(seq)
    L = len(s)
    c = Counter(s)
    return {aa: c.get(aa, 0) / L if L > 0 else 0.0 for aa in sorted(AA20)}


def basic_props(seq: str, ph: float =8.5)-> Dict[str, float]: 
    """
    Compute a standard set of ProtParam-like properties for one sequence.
    
    Properties returned
    -------------------
    - length: int
        Number of residues after cleaning 
    - mw: float
        Molecular weight (Da), from Biopython's `molecular_weight()`.
    - pi: float
        Theoretical isoelectric point (pI), Bjellqvist method. 
    - gravy: float
        Grand average of hydropathy (Kyte-Doolittle), mean hydropathy across residues.
    - instability: float
        instability index (Guruprasad); values > 40 suggest instability.
    - aromaticity: float
        Fraction of aromatic residues (F/W/Y).
    - aliphatic_index: float
        Ikai aliphatic index (see `aliphatic_index()`).
    - ext_reduced: int
        Molar extinction coefficient at 280 nm, assuming reduced Cys. 
        aka: cysteines are not forming disulfide bonds, all cys residues are reduced.
    - ext_cystine: int
        Molar extinction coefficient assumin Cys form disulfides (cystine).
        cysteines (oxidized state) are forming disulfide bridges.
    - charge_at_pH: float 
        Net charge of the peptide at a given pH (default 8.5). 

    Parameters
    ----------
    seq: str
        Amino acid sequence (letters only). Non-canonical symbols are removed.

    Returns
    -------
    dict
        Dictionary with the properties listed above
    
    Implementation details
    ----------------------
    - For empty sequences (or sequences that were fully cleaned away), all values are 0.
    - Biopython's `ProteinAnalysis` is used where possible for the formulas. 
    - Extinction coefficients are returned as integers (Biopython returns a tuple: 
      (reduced, cystine)), and may raise in older versions, handled defensively. 
      So if biopython behaves unexpectedly, the program won't break. 
    """
    s = _clean(seq)
    if not s: 
        return _empty_props()
    
    pa = ProteinAnalysis(s)

    #Base metrics from Biopython/ProtParam
    mw = pa.molecular_weight()
    pi = pa.isoelectric_point()
    gravy = pa.gravy()
    instab = pa.instability_index()
    arom = pa.aromaticity()

    #Aliphatic index - custom implementation
    ai = aliphatic_index(s)

    #Extinction coefficients:
    #Biopython retrns a 2-tuple (reduced, cystine)
    ext_red: int
    ext_cys: int
    try: 
        ext_tuple = pa.molar_extinction_coefficient()
        if isinstance(ext_tuple, tuple) and len(ext_tuple)==2: 
            ext_red, ext_cys = int(ext_tuple[0]), int(ext_tuple[1])
        else:
            # rare/old edge case where a single value is returned -> replicate
            ext_red = ext_cys= int(ext_tuple)
    except (TypeError, ValueError, AttributeError): 
        # If computation fails (e.g., invalid sequence residues), use zeros
        # This can happen with unusual sequences or Biopython compatibility issues
        ext_red = ext_cys = 0
    
    charge_at_pH = pa.charge_at_pH(ph)
    return {
        "length": len(s),
        "mw": float(mw),
        "pi": float(pi),
        "gravy": float(gravy),
        "instability": float(instab),
        "aromaticity": float(arom),
        "aliphatic_index": float(ai),
        "ext_reduced": int(ext_red),
        "ext_cystine": int(ext_cys),
        "charge_at_pH": float(charge_at_pH),
    }