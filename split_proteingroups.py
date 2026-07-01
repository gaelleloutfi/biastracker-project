"""Split the combined MaxQuant proteinGroups table into one file per method.

The combined table (DVPT191203_proteinGroups_DOC_SP3_STRAP_no25ug.txt) holds
LFQ intensities for all three sample-prep methods (DOC, SP3, STRAP) side by
side. Each output file keeps the shared annotation columns plus that method's
LFQ replicate columns, and — importantly — is filtered to the proteins actually
*detected* in that method (n_detected_<method> > 0).

Filtering by detection is what makes the per-method files genuinely different:
without it every file would contain the identical protein set and the
physicochemical bias comparison would be meaningless.
"""
from __future__ import annotations

import csv
from pathlib import Path

SRC = Path(__file__).parent / (
    "biastracker/data/processed/DVPT191203/"
    "DVPT191203_proteinGroups_DOC_SP3_STRAP_no25ug.txt"
)
OUT_DIR = SRC.parent

# Shared annotation columns kept in every output file.
SHARED_COLS = [
    "Protein IDs",
    "Majority protein IDs",
    "Gene names",
    "Protein names",
    "Fasta headers",
    "Only identified by site",
    "Reverse",
    "Potential contaminant",
]

# method label -> (LFQ column prefix in source, output filename suffix)
METHODS = {
    "DOC":   ("LFQ intensity DOC_",   "DOC"),
    "SP3":   ("LFQ intensity SP3_",   "SP3"),
    "Strap": ("LFQ intensity Strap_", "STRAP"),
}


def _is_detected(value: str) -> bool:
    try:
        return float(value.strip()) > 0
    except (TypeError, ValueError):
        return False


def split() -> None:
    with SRC.open(encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        header = reader.fieldnames or []
        rows = list(reader)

    for method, (lfq_prefix, suffix) in METHODS.items():
        lfq_cols = [c for c in header if c.startswith(lfq_prefix)]
        if not lfq_cols:
            raise SystemExit(f"No LFQ columns found for prefix {lfq_prefix!r}")
        n_det_col = f"n_detected_{method}"

        out_cols = SHARED_COLS + lfq_cols
        out_path = OUT_DIR / f"DVPT191203_proteinGroups_{suffix}.txt"

        kept = 0
        with out_path.open("w", encoding="utf-8", newline="") as out:
            writer = csv.DictWriter(
                out, fieldnames=out_cols, delimiter="\t", extrasaction="ignore"
            )
            writer.writeheader()
            for row in rows:
                # Prefer the precomputed n_detected column; fall back to LFQ cols.
                if n_det_col in row:
                    detected = _is_detected(row[n_det_col])
                else:
                    detected = any(_is_detected(row[c]) for c in lfq_cols)
                if detected:
                    writer.writerow(row)
                    kept += 1

        print(f"{suffix:5s}: {kept:5d} detected proteins -> {out_path.name}")


if __name__ == "__main__":
    split()
