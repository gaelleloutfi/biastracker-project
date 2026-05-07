"""
IO utilities for FASTA reference files.

This module provides functionality to parse FASTA files and generate theoretical
tryptic peptidomes. It takes protein sequences from a FASTA file, performs
in-silico digestion, and returns both protein-level and peptide-level DataFrames
with computed physicochemical properties.

The main entry point is:
    from_fasta_reference(...)

which:
    - parses a FASTA file using BioPython SeqIO,
    - generates tryptic peptides from each protein sequence,
    - computes physicochemical properties using basic_props(),
    - computes missed cleavages for peptides,
    - standardizes the results with ensure_sequence_table().
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
from Bio import SeqIO

from .features_basic import basic_props, amino_acid_composition, _clean
from .features_digest import in_silico_peptides, missed_cleavages_in_peptide as mc_pep, trypsin_sites
from .id_utils import normalize_uniprot_accession
from .schema import ensure_sequence_table


def from_fasta_reference(
    path: str | Path,
    mc: int = 0,
    min_len: int = 6,
    max_len: int = 65,
    normalize_ids: bool = True,
    default_expression: float = 0.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parse a FASTA file and generate a theoretical tryptic peptidome.

    This function reads protein sequences from a FASTA file, performs in-silico
    tryptic digestion on each protein, and returns two DataFrames: one for proteins
    and one for peptides. Both DataFrames include computed physicochemical properties.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the input FASTA file.
    mc : int, default=0
        Maximum number of missed cleavages allowed when generating peptides.
    min_len : int, default=6
        Minimum peptide length to keep.
    max_len : int, default=65
        Maximum peptide length to keep.
    normalize_ids : bool, default=True
        If True, normalize protein IDs using `normalize_uniprot_accession()`.
        If False, use the raw record.id from FASTA header.
        The raw header description is always stored in the 'header' column.
    default_expression : float, default=0.0
        Default expression value to assign to all proteins. This ensures
        protein tables always contain an 'expression' column, matching the
        structure of build_protein_table() output.

    Returns
    -------
    proteins_df : pd.DataFrame
        DataFrame with one row per protein, containing:
        - primary_id: protein identifier (normalized if normalize_ids=True)
        - header: raw FASTA header description
        - sequence: amino acid sequence (cleaned to canonical AA only)
        - expression: expression value (from default_expression parameter)
        - length: sequence length
        - ... additional physicochemical properties from basic_props()
        - trypsin_sites: number of trypsin cleavage sites
        - aa_A, aa_C, ..., aa_Y: amino acid composition (20 columns)
        - level: "protein"
    peptides_df : pd.DataFrame
        DataFrame with one row per peptide, containing:
        - primary_id: protein identifier (source protein)
        - sequence: peptide amino acid sequence
        - length: peptide sequence length (from basic_props)
        - missed_cleavages: number of missed cleavages in the peptide
        - ... additional physicochemical properties from basic_props()
        - level: "peptide"

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.

    Example
    -------
    >>> proteins_df, peptides_df = from_fasta_reference("proteins.fasta", mc=1)
    >>> print(f"Found {len(proteins_df)} proteins")
    >>> print(f"Generated {len(peptides_df)} peptides")
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"FASTA file not found: {path}")

    # Parse FASTA file and build protein records
    protein_records = []
    peptide_records = []

    for record in SeqIO.parse(path, "fasta"):
        # Extract and potentially normalize protein ID
        if normalize_ids:
            protein_id = normalize_uniprot_accession(record.id) or record.id
        else:
            protein_id = record.id
        
        # Clean the sequence to canonical amino acids only
        # This ensures consistent behavior across all property calculations
        protein_seq_raw = str(record.seq)
        protein_seq = _clean(protein_seq_raw)
        header = record.description  # Full FASTA header

        # Compute protein properties using the cleaned sequence
        protein_props = basic_props(protein_seq)
        
        # Compute trypsin sites using the cleaned sequence
        protein_trypsin_sites = trypsin_sites(protein_seq)
        
        # Compute amino acid composition using the cleaned sequence
        aa_comp = amino_acid_composition(protein_seq)
        
        # Prefix amino acid columns with "aa_"
        aa_comp_prefixed = {f"aa_{aa}": frac for aa, frac in aa_comp.items()}
        
        protein_record = {
            "primary_id": protein_id,
            "header": header,
            "sequence": protein_seq,
            "expression": default_expression,
            **protein_props,
            "trypsin_sites": protein_trypsin_sites,
            **aa_comp_prefixed,
        }
        protein_records.append(protein_record)

        # Generate tryptic peptides for this protein using the cleaned sequence
        peptides = in_silico_peptides(protein_seq, mc=mc, min_len=min_len, max_len=max_len)

        for peptide_seq in peptides:
            # Compute peptide properties
            peptide_props = basic_props(peptide_seq)
            peptide_mc = mc_pep(peptide_seq)

            peptide_record = {
                "primary_id": protein_id,  # Source protein
                "sequence": peptide_seq,   # Store directly in 'sequence' column
                "missed_cleavages": peptide_mc,
                **peptide_props,
            }
            peptide_records.append(peptide_record)

    # Create DataFrames
    # When records are empty, create DataFrame with required columns for ensure_sequence_table
    if not protein_records:
        proteins_df = pd.DataFrame(columns=["primary_id", "sequence"])
    else:
        proteins_df = pd.DataFrame(protein_records)
    
    if not peptide_records:
        peptides_df = pd.DataFrame(columns=["primary_id", "sequence", "missed_cleavages"])
    else:
        peptides_df = pd.DataFrame(peptide_records)

    # Standardize with ensure_sequence_table
    # Pass id_col="primary_id" to properly handle the primary_id column
    proteins_df = ensure_sequence_table(proteins_df, level="protein", id_col="primary_id")
    peptides_df = ensure_sequence_table(peptides_df, level="peptide", id_col="primary_id")

    return proteins_df, peptides_df
