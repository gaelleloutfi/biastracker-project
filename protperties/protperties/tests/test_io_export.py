
import pandas as pd
from pathlib import Path

from protperties.io_export import export_table


def test_export_table_basic(tmp_path):
    """Test basic CSV export with a simple DataFrame."""
    # Create a small test DataFrame
    df = pd.DataFrame({
        "sequence": ["PEPTIDE", "PROTEIN"],
        "length": [7, 7],
        "mw": [799.89, 799.89],
    })
    
    # Export to CSV
    output_path = tmp_path / "test_output.csv"
    result_path = export_table(df, output_path)
    
    # Verify the file was created
    assert result_path.exists()
    assert result_path == output_path
    
    # Read it back
    df_read = pd.read_csv(result_path)
    
    # Assert equality of shape and column names
    assert df_read.shape == df.shape
    assert list(df_read.columns) == list(df.columns)
    
    # Verify content
    assert list(df_read["sequence"]) == ["PEPTIDE", "PROTEIN"]
    assert list(df_read["length"]) == [7, 7]


def test_export_table_auto_append_csv_suffix(tmp_path):
    """Test that .csv suffix is automatically appended if missing."""
    df = pd.DataFrame({"col": [1, 2, 3]})
    
    # Provide path without .csv extension
    output_path = tmp_path / "output"
    result_path = export_table(df, output_path)
    
    # Check that .csv was appended
    assert result_path.suffix == ".csv"
    assert result_path.name == "output.csv"
    assert result_path.exists()
    
    # Verify readability
    df_read = pd.read_csv(result_path)
    assert df_read.shape == df.shape


def test_export_table_overwrite_existing(tmp_path):
    """Test that existing files are overwritten."""
    df1 = pd.DataFrame({"a": [1, 2]})
    df2 = pd.DataFrame({"b": [3, 4, 5]})
    
    output_path = tmp_path / "output.csv"
    
    # First export
    export_table(df1, output_path)
    df_read1 = pd.read_csv(output_path)
    assert df_read1.shape == (2, 1)
    
    # Second export (overwrite)
    export_table(df2, output_path)
    df_read2 = pd.read_csv(output_path)
    assert df_read2.shape == (3, 1)
    assert "b" in df_read2.columns
    assert "a" not in df_read2.columns


def test_export_table_with_index(tmp_path):
    """Test that index parameter works correctly."""
    df = pd.DataFrame({"col": [10, 20]}, index=["row1", "row2"])
    
    # Export without index (default)
    path1 = tmp_path / "without_index.csv"
    export_table(df, path1)
    df_read1 = pd.read_csv(path1)
    assert "Unnamed: 0" not in df_read1.columns
    assert df_read1.shape == (2, 1)
    
    # Export with index
    path2 = tmp_path / "with_index.csv"
    export_table(df, path2, index=True)
    df_read2 = pd.read_csv(path2)
    # pandas reads it back with first column as unnamed or as actual index name
    assert df_read2.shape[0] == 2


def test_export_table_comma_separator(tmp_path):
    """Test that comma is used as separator."""
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    
    output_path = tmp_path / "output.csv"
    export_table(df, output_path)
    
    # Read raw file content to verify comma separator
    with open(output_path, "r") as f:
        content = f.read()
    
    # Check that commas are present (header and data rows)
    assert "a,b" in content
    assert "1,3" in content


def test_export_table_path_as_string(tmp_path):
    """Test that path can be provided as a string."""
    df = pd.DataFrame({"x": [100]})
    
    # Provide path as string
    output_path = str(tmp_path / "string_path.csv")
    result_path = export_table(df, output_path)
    
    # Should return Path object
    assert isinstance(result_path, Path)
    assert result_path.exists()
    
    # Verify content
    df_read = pd.read_csv(result_path)
    assert df_read["x"].iloc[0] == 100


def test_export_table_replaces_non_csv_extension(tmp_path):
    """Test that non-.csv extensions are replaced with .csv."""
    df = pd.DataFrame({"col": [1, 2, 3]})
    
    # Provide path with .txt extension
    output_path = tmp_path / "output.txt"
    result_path = export_table(df, output_path)
    
    # Check that extension was replaced with .csv
    assert result_path.suffix == ".csv"
    assert result_path.name == "output.csv"
    assert result_path.exists()
    
    # Verify readability
    df_read = pd.read_csv(result_path)
    assert df_read.shape == df.shape
    
    # Ensure .txt file was not created
    assert not output_path.exists()
