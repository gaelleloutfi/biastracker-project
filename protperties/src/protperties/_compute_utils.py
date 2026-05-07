"""
Internal utilities for computing properties on pandas Series.

This module provides shared helper functions used by multiple I/O modules
to compute physicochemical properties on peptide/protein sequences stored
in pandas Series.
"""
from __future__ import annotations

import pandas as pd

from .features_basic import basic_props
from .features_digest import missed_cleavages_in_peptide as mc_pep


def compute_props_on_series(
    seq: pd.Series,
    include_missed_cleavages: bool = True,
) -> pd.DataFrame:
    """
    Compute physicochemical properties for each sequence in a Series.

    For each amino-acid sequence string, this applies:
        - basic_props()  → length, MW, pI, GRAVY, instability, aromaticity,
                           aliphatic index, extinction coefficients.
        - mc_pep()       → number of potential missed cleavages
                           (only when *include_missed_cleavages* is True).

    Parameters
    ----------
    seq : pandas.Series
        Series of amino-acid sequences.
    include_missed_cleavages : bool, default True
        When True, also compute and include ``missed_cleavages``.

    Returns
    -------
    pandas.DataFrame
        One row per sequence with computed property columns.
        Columns include:
        ['length', 'mw', 'pi', 'gravy', 'instability', 'aromaticity',
         'aliphatic_index', 'ext_reduced', 'ext_cystine']
        and optionally 'missed_cleavages' if include_missed_cleavages=True.
    """
    seq = seq.astype(str)
    records = []
    for s in seq:
        rec = basic_props(s)
        if include_missed_cleavages:
            rec["missed_cleavages"] = mc_pep(s)
        records.append(rec)
    return pd.DataFrame(records)
