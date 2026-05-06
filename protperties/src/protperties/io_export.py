"""
CSV export utilities for annotated sequence tables.

This module provides a simple, deterministic export function that always
exports DataFrames in CSV format without format detection or inference.

The main entry point is:
    export_table(...)

which:
    - Always exports using pandas DataFrame.to_csv()
    - Automatically replaces any existing extension with ".csv" or adds it if missing
    - Overwrites existing files
    - Uses comma as separator
    - Returns the final Path written
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def export_table(
    df: pd.DataFrame,
    path: str | Path,
    index: bool = False,
) -> Path:
    """
    Export a DataFrame to CSV format.

    This function always writes to CSV format with a comma separator.
    Extensions are normalized to ".csv". TSV or parquet support is
    intentionally not implemented yet.
    If the provided path does not end with ".csv", any existing extension
    is replaced with ".csv" (or ".csv" is added if no extension exists).
    Existing files are overwritten without warning.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to export.
    path : str or pathlib.Path
        The output file path. If it does not end with ".csv", the extension
        is replaced with ".csv" (or added if no extension exists).
    index : bool, default False
        Whether to include the DataFrame index in the output.

    Returns
    -------
    pathlib.Path
        The final path where the CSV file was written (with ".csv" extension).

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    >>> export_table(df, "output.csv")
    PosixPath('output.csv')
    >>> export_table(df, "output")  # automatically adds .csv
    PosixPath('output.csv')
    >>> export_table(df, "output.txt")  # replaces .txt with .csv
    PosixPath('output.csv')
    """
    path = Path(path)
    
    # Ensure the path ends with .csv
    if path.suffix != ".csv":
        path = path.with_suffix(".csv")
    
    # Export using pandas to_csv (comma is the default separator)
    df.to_csv(path, index=index)
    
    return path
