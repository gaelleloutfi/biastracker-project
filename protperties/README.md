# protperties

`protperties` computes peptide/protein physicochemical properties and adds those properties to proteomics tables from DIA-NN and MaxQuant.

---

## Table of Contents

- [Key Features](#key-features)
- [Repository Tree](#repository-tree)
- [Root-Level Files](#root-level-files)
- [Package Layout (`src/`)](#package-layout-src)
  - [`protperties/__init__.py` — Public API](#protperties__initpy--public-api)
  - [`features_basic.py` — Physicochemical properties](#features_basicpy--physicochemical-properties)
  - [`features_digest.py` — Digestion & missed cleavages](#features_digestpy--digestion--missed-cleavages)
  - [`io_tables.py` — DIA-NN Parquet I/O](#io_tablespy--dia-nn-parquet-io)
  - [`io_maxquant.py` — MaxQuant evidence.txt I/O](#io_maxquantpy--maxquant-evidencetxt-io)
- [Tests (`tests/`)](#tests-tests)
- [Test Data (`test_data/`)](#test-data-test_data)
- [Architecture & Data Flow](#architecture--data-flow)
- [Technology Stack](#technology-stack)
- [How Everything Fits Together](#how-everything-fits-together)

---

> [!WARNING]
> **Important Disclaimer**: `protperties` is designed as a low-level computational engine and data standardizer. It is **not** a final analysis tool. It provides the building blocks (physicochemical properties, canonical schemas, unified proteomics I/O) that downstream dashboards and analytical pipelines consume.

## Installation

This project uses [Pixi](https://pixi.sh/) for reproducible environment management.

1. Clone the repository:
   ```bash
   git clone https://github.com/user/protperties.git
   cd protperties
   ```
2. Run tests or start using the environment:
   ```bash
   pixi run test
   pixi run python
   ```
   Or install it as a standard Python package in your environment:
   ```bash
   pip install -e .
   ```

## Quick Start & Examples

`protperties` unifies output from different tools into a canonical schema.

### DIA-NN (Peptide-level)
```python
from protperties import from_diann_parquet

# Loads DIA-NN report, applies Q-value/PEP filters, and computes peptide properties
df = from_diann_parquet("report.parquet")
print(df[["primary_id", "sequence", "expression", "mw", "gravy"]].head())
```

### MaxQuant (Protein-level)
```python
from protperties import from_maxquant_proteingroups

# Loads proteinGroups.txt, computes median LFQ expression across samples
df_pg = from_maxquant_proteingroups("proteinGroups.txt", agg_method="median", drop_zeros=True)
print(df_pg[["primary_id", "expression"]].head())
```

### Manual Tables
```python
from protperties import from_manual_table

# Automatically detects sequence/expression columns in arbitrary CSVs
df_manual = from_manual_table("my_data.csv", level="peptide")
```

### FASTA Reference Proteome
```python
from protperties import build_reference_proteome

# Queries UniProt, caches the human FASTA, and computes in silico tryptic peptides
proteins_df, peptides_df = build_reference_proteome("homo sapiens", mc=1)
```

---

## Key Features

### Core computations (sequence-level)
- Compute protein/peptide physicochemical descriptors (ExPASy/ProtParam-like), including:
  - sequence length, molecular weight, isoelectric point, GRAVY (Kyte–Doolittle),
  - instability index, aromaticity fraction,
  - aliphatic index (Ikai),
  - extinction coefficients (reduced and cystine),
  - net charge at pH 7 (default),
  - amino acid composition (fraction of each of the 20 canonical residues),
  - plus canonical amino acid cleaning.
- Perform enzymatic digestion logic (trypsin by default), including:
  - theoretical cleavage site counting,
  - in silico peptide generation with configurable missed cleavages and peptide length limits,
  - missed-cleavage counting within a peptide sequence.

### Proteomics I/O integrations (table-level)
- Load and process **DIA-NN** Parquet reports:
  - apply recommended quality filters (q-values, PEP thresholds, decoys, etc.),
  - compute peptide properties from **`Stripped.Sequence`**,
  - support deduplication and grouping options,
  - standardize output as a pandas DataFrame.
- Load and process **MaxQuant** `evidence.txt`:
  - filter by PEP and remove reverse/contaminant hits according to config,
  - map MaxQuant columns to a DIA-NN-like schema,
  - compute the same peptide properties for consistent downstream analysis.

### Modern packaging & reproducibility
- `src/` layout package with a clean exported API.
- **Pixi** environment management for reproducible dependency resolution (`pixi.lock`).
- Full unit test coverage with **pytest**, including synthetic minimal datasets for I/O testing.

---

## Repository Tree

```
protperties/  
├── .gitattributes # Git attributes for file handling  
├── .gitignore # Files/dirs excluded from version control  
├── LICENSE.md # License information  
├── README.md # readme  
├── pyproject.toml # Package metadata/config (setuptools/pip)  
├── pixi.toml # Pixi environment configuration  
├── pixi.lock # Locked dependency versions for reproducibility  
├── src/  
│ └── protperties/  
│   └── __init__.py # Public API exports  
│   ├── features_basic.py # Physicochemical property computations  
│   ├── features_digest.py # Trypsin digestion + missed-cleavage utilities  
│   ├── io_tables.py # DIA-NN Parquet I/O + filtering + property annotation  
│   └── io_maxquant.py # MaxQuant evidence.txt I/O + mapping + filtering  
├── tests/  
│ ├── test_features_basic.py # Unit tests for features_basic.py  
│ ├── test_features_digest.py # Unit tests for features_digest.py  
│ ├── test_io_tables.py # Unit tests for io_tables.py (DIA-NN)  
│ └── test_io_maxquant.py # Unit tests for io_maxquant.py (MaxQuant)  
└── test_data/  
└── report.parquet # Sample DIA-NN Parquet for notebooks/tests 
├── evidence.txt # Sample MaxQuant evidence file for notebooks/tests  
```

---

## Root-Level Files

### `.gitattributes` 
Git configuration file defining how specific file types should be handled by Git (e.g., normalization, diff/merge behaviors depending on patterns).

### `.gitignore` 
Defines which files and directories are excluded from version control (typical examples include build artifacts, local environments, caches, etc.).

### `LICENSE.md` 
Contains the project’s licensing terms.

### `README.md` 
Project overview, architecture notes, and usage guidance.

### `pyproject.toml` 
Defines Python project metadata and packaging configuration for setuptools/pip (name, versioning metadata, build system, dependencies configuration approach, etc.).

### `pixi.toml` 
Pixi environment configuration:
- Declares dependencies and environment setup for development and usage.
- Supports modern reproducible workflows for Python environments.

### `pixi.lock` 
A lockfile pinning exact dependency versions to ensure reproducible environments across machines and time.

---

## Package Layout (`src/`)

The repository follows the **`src/` layout**, meaning the importable package lives in:

- `src/protperties/`

This keeps project tooling and repository files separated from importable code.

### `protperties/__init__.py` — Public API

This file is the **entry point** for users of the package. It defines and exposes the **public API** by exporting:

#### Exported functions
- `basic_props()`  
  Computes physicochemical properties for a peptide/protein amino-acid sequence.
- `amino_acid_composition()`  
  Computes the fraction of each of the 20 canonical amino acids in a sequence.
- `trypsin_sites()`  
  Counts theoretical trypsin cleavage sites.
- `in_silico_peptides()`  
  Generates tryptic peptides (with configurable missed cleavages and peptide length constraints).
- `missed_cleavages_in_peptide()`  
  Counts internal missed cleavage sites within a peptide.
- `from_diann_parquet()`  
  Loads DIA-NN Parquet data, filters it, and annotates peptide properties.
- `summarize_miscleavage()`  
  Summarizes missed cleavage statistics (DIA-NN workflow utilities).
- `from_maxquant_evidence()`  
  Loads MaxQuant `evidence.txt`, filters/maps it, and annotates peptide properties.

#### Exported classes
- `MaxQuantFilterConfig`  
  Configuration class for MaxQuant filtering behavior.

This design allows consumers to do:

```python
from protperties import from_diann_parquet, basic_props
```
without importing internal modules directly.

---

### `features_basic.py` — Physicochemical properties

Implements **core sequence-level physicochemical computations** for peptide/protein sequences.

### Characteristics

- Built as **pure functions**: input is a sequence string; output is computed descriptors.
- Implements algorithms similar to **ExPASy/ProtParam**.
- Uses **Biopython** `ProteinAnalysis` for most calculations.

### Key functions

### `basic_props(seq: str) -> dict`

Returns a comprehensive dictionary of properties, including:

- `length` — sequence length
- `mw` — molecular weight
- `pi` — isoelectric point
- `gravy` — hydropathy index (Kyte–Doolittle)
- `instability` — instability index
- `aromaticity` — aromatic amino acid fraction
- `aliphatic_index` — aliphatic index (Ikai method)
- `ext_reduced` — extinction coefficient (reduced form)
- `ext_cystine` — extinction coefficient (cystine bridges form)
- `charge_at_pH` — net charge at pH 7 (default behavior described)

### `aliphatic_index(seq: str) -> float`

Computes the aliphatic index according to the Ikai method.

### `amino_acid_composition(seq: str) -> Dict[str, float]`

Returns a dictionary mapping each of the 20 canonical amino acid single-letter codes to its
fraction in the (cleaned) sequence. Amino acids absent from the sequence have a value of `0.0`.
Empty sequences return all values as `0.0`.

**Example:**
```python
from protperties import amino_acid_composition
comp = amino_acid_composition("AAAAC")
# comp["A"] == 0.8, comp["C"] == 0.2, all others == 0.0
```

### Helper function

### `_clean()`

Removes **non-canonical amino acids** to ensure downstream computations (Biopython/ProtParam-style) operate on valid sequences.

---

### `features_digest.py` — Digestion & missed cleavages

Provides enzymatic digestion utilities (trypsin by default) and missed-cleavage logic.

### Trypsin rule implemented

- Cleaves **after K or R**
- **Except when followed by P**
- Matches ExPASy/PeptideCutter-style trypsin behavior

### Implementation details

- Uses `pyteomics.parser` to implement ExPASy-style rules.

### Key functions

### `trypsin_sites(seq: str) -> int`

Counts theoretical trypsin cleavage sites in the sequence.

### `in_silico_peptides(seq, mc=0, min_len=6, max_len=65) -> list[str]`

Generates in silico tryptic peptides:

- `mc`: allowed missed cleavages (0 by default)
- `min_len`: minimum peptide length (default 6)
- `max_len`: maximum peptide length (default 65)

### `missed_cleavages_in_peptide(pep: str) -> int`

Counts **internal** tryptic cleavage opportunities inside a peptide sequence (i.e., missed cleavages).

---

### `io_tables.py` — DIA-NN Parquet I/O

Handles I/O and integration for **DIA-NN Parquet report files**.

### Responsibilities

- Load DIA-NN `.parquet` tables.
- Apply recommended proteomics QC filters (e.g., q-values, decoys, PEP thresholds).
- Compute peptide-level properties from **`Stripped.Sequence`**.
- Support deduplication and grouping options.
- Return a standardized annotated **pandas DataFrame**.

### Key functions

### `from_diann_parquet(...)`

Main entry point for loading and processing DIA-NN data:

- Reads the DIA-NN Parquet file.
- Applies filtering rules.
- Extracts peptide sequences from `Stripped.Sequence`.
- Computes peptide properties (via `features_basic.py` and digestion metrics via `features_digest.py` as needed).
- Performs deduplication/grouping according to configured behavior.
- Outputs an annotated DataFrame.

### `summarize_miscleavage(...)`

Generates missed-cleavage statistics, providing summary-level insight into digestion-related metrics.

### Configuration class

### `FilterConfig` (dataclass)

Encapsulates DIA-NN filtering parameters, including:

- PEP threshold
- q-value threshold(s)
- library q-value
- global q-value

### Key DIA-NN columns used

The module operates with the DIA-NN schema, including (non-exhaustive list as explicitly used/mentioned):

- `Run`
- `Precursor.Id`
- `Stripped.Sequence`
- `Precursor.Charge`
- `Protein.Group`
- `Protein.Ids`
- `Protein.Names`
- `Genes`
- `PEP`
- `Q.Value`
- and related DIA-NN report columns needed for filtering and grouping

---

### `io_maxquant.py` — MaxQuant `evidence.txt` I/O

Provides an adapter layer for **MaxQuant** output (`evidence.txt`) to align with the DIA-NN-like interface.

### Responsibilities

- Read `evidence.txt` (tab-separated).
- Apply quality filters (PEP, reverse hits, contaminant flags depending on config).
- Map MaxQuant columns into DIA-NN-equivalent names.
- Compute the same peptide properties as in the DIA-NN workflow.
- Return a standardized annotated DataFrame.

### Key functions

### `from_maxquant_evidence(...)`

Main entry point:

- Loads evidence file.
- Applies MaxQuant-specific filters.
- Builds/constructs a DIA-NN-like schema (including precursor identifiers).
- Computes peptide properties and charge-at-pH metrics.
- Outputs an annotated DataFrame.

### `_load_evidence(path)`

Loads the tab-separated `evidence.txt` file.

### `_apply_mq_filters(df, cfg)`

Applies MaxQuant filters according to the provided configuration (PEP max, reverse/contaminant handling, intensity selection behavior).

### Configuration class

### `MaxQuantFilterConfig` (dataclass)

Controls MaxQuant filtering and mapping, including:

- maximum PEP (`PEP max`)
- which intensity column to use
- flags controlling removal of reverse hits / contaminants

---

## Tests (`tests/`)

The test suite provides **comprehensive pytest coverage** for all modules, including both sequence-level logic and table-level I/O.

### `test_features_basic.py`

Tests for `features_basic.py`:

- `test_clean_removes_noncanonical()`
    
    Ensures `_clean()` removes non-canonical amino acids correctly.
    
- `test_basic_props_length_and_mw()`
    
    Validates sequence length and molecular weight computation.
    
- `test_basic_props_pi_is_reasonable()`
    
    Ensures computed pI falls within a sensible range.
    
- `test_basic_props_gravy_signs()`
    
    Checks expected hydrophobic/hydrophilic GRAVY behavior.
    
- `test_basic_props_stability_index()`
    
    Validates instability index ranges.
    
- `test_basic_props_aliphatic_index()`
    
    Ensures aliphatic index behaves as expected for hydrophobic sequences.
    
- `test_basic_props_charge_at_pH_default()`
    
    Validates net charge calculations at default pH behavior.
    

### `test_features_digest.py`

Tests for `features_digest.py`:

- `test_trypsin_sites_simple()`
    
    Validates cleavage site counting.
    
- `test_in_silico_peptides_empty()`
    
    Ensures empty sequences are handled gracefully.
    
- `test_in_silico_peptides_no_miscleavages()`
    
    Validates peptide generation with `mc=0`.
    
- `test_in_silico_peptides_allowing_miscleavages()`
    
    Validates peptide generation with multiple allowed miscleavages.
    
- `test_missed_cleavages_zero()`
    
    Ensures peptides with no internal tryptic sites are recognized.
    
- `test_missed_cleavages_positive()`
    
    Validates counting of internal tryptic sites (missed cleavages).
    

### `test_io_tables.py`

Tests DIA-NN I/O (`io_tables.py`):

- `_make_tiny_parquet()`
    
    Helper to create minimal Parquet-like test data.
    
- `test_from_diann_parquet_filters_and_props()`
    
    Verifies filtering + property computation end-to-end on synthetic DIA-NN-like data.
    
- `test_dedup_max_on_normalised()`
    
    Ensures deduplication logic works correctly (including the “max on normalized” behavior).
    

Notes explicitly covered by the tests:

- Synthetic DIA-NN-like DataFrames with **3 rows** are created to validate filtering/grouping/property computation.
- Verifies that filtering removes rows correctly (e.g., based on Q.Value / PEP logic as described).

### `test_io_maxquant.py`

Tests MaxQuant I/O (`io_maxquant.py`):

- `_make_tiny_evidence()`
    
    Helper to create a minimal `evidence.txt` test dataset.
    
- `test_from_maxquant_basic_filters_and_mapping()`
    
    Validates:
    
    - PEP filtering,
    - reverse hit removal,
    - correct column mapping to DIA-NN-like names,
    - precursor ID construction,
    - property computation (including charge at pH).
- `test_disable_pep_filter()`
    
    Ensures PEP filtering can be disabled as expected.
    

---

## Test Data (`test_data/`)

### `report.parquet`

A sample DIA-NN report in Parquet format used by notebooks and tests.

This directory is also intended to expand over time:

- “Will contain more real data that will be used for validation.”

---

## Architecture & Data Flow

### Separation of concerns (modular design)

1. **Feature Computation Layer**
    - `features_basic.py`, `features_digest.py`
    - Pure sequence functions (no dependency on DIA-NN/MaxQuant schemas)
    - Reusable across tools and workflows
2. **I/O & Integration Layer**
    - `io_tables.py`, `io_maxquant.py`
    - Loads tool-specific formats
    - Applies quality control and filtering
    - Extracts peptide sequences and annotates features
    - Returns standardized pandas DataFrames
3. **Public API Layer**
    - `__init__.py`
    - Exposes a clean, user-facing interface:
        - `from protperties import from_diann_parquet, basic_props`

### End-to-end data flow

```
Input (DIA-NN/MaxQuant)
    ↓
Load &Filter (io_tables.py / io_maxquant.py)
    ↓
Extract Peptide Sequences
    ↓
Compute Properties (features_basic.py + features_digest.py)
    ↓
Annotated DataFrame Output
```

---

## Technology Stack

- **Language composition:** ~92% Jupyter Notebook, ~8% Python (repository-level language stats)
- **Core dependencies:**
    - **Biopython** (`ProteinAnalysis`) for ProtParam-like properties
    - **Pyteomics** (`pyteomics.parser`) for ExPASy-style trypsin rule parsing
    - **Pandas** for table manipulation and standardized DataFrame outputs
    - **NumPy** for numerical operations
- **Environment management:** **Pixi** (`pixi.toml`, `pixi.lock`)
- **Testing:** **pytest** (unit tests across all modules)

---

## How Everything Fits Together

- If you have a **raw amino-acid sequence**, you can compute properties directly via:
    - `basic_props(seq)`
    - digestion metrics with `trypsin_sites(seq)`, `in_silico_peptides(seq, ...)`, and `missed_cleavages_in_peptide(pep)`
- If you have **proteomics tool outputs**, you can load, filter, and annotate them into a **property-enriched DataFrame**:
    - DIA-NN Parquet → `from_diann_parquet(...)`
    - MaxQuant evidence.txt → `from_maxquant_evidence(...)`

In both cases, the design ensures:

- consistent metrics computed from the same feature modules,
- standardized outputs suitable for downstream QC, bias analysis, and proteomics reporting pipelines,
- strong test coverage to prevent regressions and validate behavior across formats.

