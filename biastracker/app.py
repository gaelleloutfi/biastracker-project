"""BiasTracker — Streamlit GUI

Run from the biastracker/ directory:
    streamlit run app.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

# Make the in-repo packages importable on hosts that install only PyPI deps
# (e.g. Streamlit Community Cloud installs from requirements.txt, not the local
# `biastracker`/`protperties` packages). Adding their src/ dirs to the path lets
# `import biastracker` / `import protperties` work without a pip install.
_APP_DIR = Path(__file__).resolve().parent
for _src in (_APP_DIR / "src", _APP_DIR.parent / "protperties" / "src"):
    if _src.is_dir() and str(_src) not in sys.path:
        sys.path.insert(0, str(_src))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="BiasTracker",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Brand palette (from logo) ─────────────────────────────────────────────────
_NAVY   = "#0E1F3D"
_BLUE   = "#1565C0"
_CYAN   = "#00B4D8"
_COLORS = [_CYAN, "#FF6B6B", "#FFD93D", "#6BCB77", "#845EC2", "#F9A825"]


def _default_dataset_color(idx: int) -> str:
    """Fallback palette colour for the idx-th dataset."""
    return _COLORS[idx % len(_COLORS)]


def _dataset_color(name: str, fallback_idx: int = 0) -> str:
    """The user-chosen colour for a dataset (by name), or a palette fallback.

    Colours are stored in ``st.session_state['ds_colors']`` keyed by dataset
    name and set by the per-dataset colour pickers in the sidebar, so a
    dataset's colour is consistent across every visualisation.
    """
    return st.session_state.get("ds_colors", {}).get(name, _default_dataset_color(fallback_idx))

# ── Feature registry ──────────────────────────────────────────────────────────
_FEAT_META: dict[str, str] = {
    "length":          "Sequence Length",
    "mw":              "Molecular Weight (Da)",
    "pi":              "Isoelectric Point (pI)",
    "gravy":           "GRAVY Score",
    "instability":     "Instability Index",
    "aromaticity":     "Aromaticity",
    "aliphatic_index": "Aliphatic Index",
    "charge_at_pH":    "Net Charge at pH",
    "trypsin_sites":   "Trypsin Sites",
    "missed_cleavages":"Missed Cleavages",
    "ext_cystine":     "Extinction Coeff. (cystine)",
    "ext_reduced":     "Extinction Coeff. (reduced)",
    "expression":      "Expression",
    "PTR_AML":         "PTR (AML)",
}

# Abundance / large-magnitude features whose raw units dwarf the per-residue
# physicochemical features on a shared Δ axis — optionally hidden from charts.
_INTENSITY_FEATS: set[str] = {"expression", "PTR_AML", "ext_reduced", "ext_cystine"}

_DS_TYPES: dict[str, str] = {
    "DIA-NN Report (.parquet)":     "diann_report",
    "MaxQuant evidence.txt":        "maxquant_evidence",
    "MaxQuant proteinGroups.txt":   "maxquant_proteingroups",
    "DIA-NN PG Matrix":             "diann_pg_matrix",
    "Manual / Custom CSV":          "manual",
    "Standard CSV (pre-processed)": "standard_csv",
}

# Level is fixed for tool-specific types; manual/standard let the user choose.
_FIXED_LEVEL: dict[str, str] = {
    "diann_report":           "peptide",
    "maxquant_evidence":      "peptide",
    "maxquant_proteingroups": "protein",
    "diann_pg_matrix":        "protein",
}

_EXTENSIONS: dict[str, list[str]] = {
    "diann_report":           ["parquet"],
    "maxquant_evidence":      ["txt"],
    "maxquant_proteingroups": ["txt"],
    "diann_pg_matrix":        ["tsv", "txt", "csv"],
    "manual":                 ["csv", "tsv", "txt"],
    "standard_csv":           ["csv", "tsv", "txt"],
}

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
/* Sidebar gradient */
[data-testid="stSidebar"] > div:first-child {{
    background: linear-gradient(170deg, {_NAVY} 0%, {_BLUE} 100%);
}}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stTextInput label,
[data-testid="stSidebar"] .stFileUploader label {{
    color: rgba(255,255,255,0.9) !important;
}}
[data-testid="stSidebar"] .stExpander summary p {{
    color: rgba(255,255,255,0.85) !important;
    font-weight: 600;
}}

/* Metric cards */
.bt-card {{
    background: white;
    border-left: 4px solid {_CYAN};
    border-radius: 10px;
    padding: 1rem 1.2rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
}}
.bt-card .val {{
    font-size: 1.9rem;
    font-weight: 700;
    color: {_NAVY};
    line-height: 1.2;
}}
.bt-card .lbl {{
    font-size: 0.82rem;
    color: #666;
    margin-top: 0.2rem;
}}

/* Section headers */
.bt-section {{
    border-bottom: 2px solid {_CYAN};
    padding-bottom: 0.3rem;
    margin: 1.2rem 0 0.8rem;
    color: {_NAVY};
    font-weight: 700;
    font-size: 1.05rem;
}}

/* Tab accent */
.stTabs [data-baseweb="tab-highlight"] {{ background-color: {_CYAN} !important; }}
.stTabs [data-baseweb="tab"][aria-selected="true"] {{ color: {_CYAN} !important; }}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════

def _apply_lfq_filter(ds):
    """Drop proteins with no LFQ value so they don't skew any downstream analysis.

    Only proteins actually quantified (non-zero LFQ / ``n_samples_used`` > 0) are
    kept, so physicochemical statistics reflect detected proteins rather than the
    full identification table. Contaminants are **exempt**: they are retained as
    ID-only rows (no sequence, so they never enter physicochemical stats) for the
    contaminant ORA. Datasets with no quantification column (e.g. manual/standard
    tables) are left unchanged. The number removed is stored in
    ``ds.metadata['n_removed_no_lfq']``.
    """
    df = ds.table
    # Key strictly on the LFQ-derived quantification count so manual/standard
    # tables (which have no LFQ concept) are left untouched.
    if "n_samples_used" not in df.columns:
        ds.metadata["n_removed_no_lfq"] = 0
        return ds
    has_lfq = df["n_samples_used"].fillna(0) > 0

    if "is_contaminant" in df.columns:
        is_contam = df["is_contaminant"].fillna(False).astype(bool)
    else:
        is_contam = pd.Series(False, index=df.index)

    keep = has_lfq | is_contam
    ds.metadata["n_removed_no_lfq"] = int((~keep).sum())
    ds.table = df[keep].reset_index(drop=True)
    return ds


@st.cache_data(show_spinner=False)
def _load(file_bytes: bytes, filename: str, ds_type: str, name: str, level: str, ph: float = 8.5):
    """Load a dataset from raw bytes. Returns (BiasDataset | None, error | None).

    ``ph`` sets the pH at which ``charge_at_pH`` is computed; it is part of the
    cache key so changing it re-computes the affected datasets.
    """
    suffix = Path(filename).suffix.lower()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        tmp.write(file_bytes)
        tmp.flush()
        tmp_path = tmp.name
    finally:
        tmp.close()

    try:
        from biastracker.dataset import (
            load_diann_report,
            load_diann_pg_matrix,
            load_manual_table_via_protperties,
            load_maxquant_evidence,
            load_maxquant_proteingroups,
            load_standard_table,
        )
        dispatch = {
            "diann_report":           lambda p: load_diann_report(p, name=name, ph=ph),
            "maxquant_evidence":      lambda p: load_maxquant_evidence(p, name=name, ph=ph),
            "maxquant_proteingroups": lambda p: load_maxquant_proteingroups(p, name=name, ph=ph),
            "diann_pg_matrix":        lambda p: load_diann_pg_matrix(p, name=name, ph=ph),
            "manual":                 lambda p: load_manual_table_via_protperties(p, name=name, level=level, ph=ph),
            "standard_csv":           lambda p: load_standard_table(p, name=name, level=level),
        }
        ds = dispatch[ds_type](tmp_path)
        ds = _apply_lfq_filter(ds)
        return ds, None
    except Exception as exc:
        return None, str(exc)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@st.cache_data(show_spinner=False)
def _load_annotations(
    file_bytes: bytes,
    filename: str,
    fmt: str,
    id_col: str,
    term_col: str,
    term_id_col: str,
    category_col: str,
):
    """Load an annotation file into an AnnotationSet. Returns (ann | None, err | None)."""
    suffix = Path(filename).suffix.lower()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        tmp.write(file_bytes)
        tmp.flush()
        tmp_path = tmp.name
    finally:
        tmp.close()

    try:
        from biastracker.annotations.custom import load_gmt, load_long_annotation_table
        stem = Path(filename).stem
        if fmt == "GMT":
            ann = load_gmt(tmp_path, name=stem)
        else:
            ann = load_long_annotation_table(
                tmp_path,
                name=stem,
                id_col=id_col or "primary_id",
                term_col=term_col or "term_name",
                term_id_col=term_id_col or None,
                category_col=category_col or None,
            )
        return ann, None
    except Exception as exc:
        return None, str(exc)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# Bundled annotation sets: label -> (path relative to app.py, source tag). All
# are long-format, UniProt-keyed, with primary_id/term_name/term_id/category.
_BUILTIN_ANNOTATIONS: dict[str, tuple[str, str]] = {
    "Contaminants DB": ("data/raw/contaminants/contaminants_long.csv", "contaminants"),
    "HPA subcellular location": ("data/raw/hpa/subcellular_location_long_uniprot.csv", "HPA"),
}


_HUMAN_PROTEOME_LABEL = "Whole human proteome (UniProt Swiss-Prot, live)"
_CUSTOM_BG_LABEL = "Upload custom background (TSV)"
_UNIPROT_HUMAN_QUERY = (
    "https://rest.uniprot.org/uniprotkb/stream"
    "?query=reviewed:true+AND+organism_id:9606&fields=accession&format=list"
)


def _parse_background_ids(uploaded) -> tuple[set[str], list[str]]:
    """Extract UniProt accessions from an uploaded background file.

    The background is treated as **UniProt accessions only**. The file may be a
    single column of accessions or any tab/comma-separated table (a header row
    is fine). Each token is validated as a UniProt accession; ``sp|ACC|NAME``
    headers and ``;``-grouped cells are resolved to their accession(s).

    Returns
    -------
    (ids, ignored)
        ``ids`` is the set of valid UniProt accessions found; ``ignored`` is the
        list of distinct tokens that were *not* UniProt accessions (e.g. gene
        symbols, other ID types, header words), so the caller can warn the user.
    """
    import re

    from protperties.id_utils import extract_uniprot_accessions

    raw = uploaded.getvalue().decode("utf-8", errors="replace")
    ids: set[str] = set()
    ignored: list[str] = []
    seen_ignored: set[str] = set()
    for token in re.split(r"[\s,]+", raw):
        token = token.strip()
        if not token:
            continue
        accs = extract_uniprot_accessions(token)
        if accs:
            ids.update(accs)
        elif token not in seen_ignored:
            seen_ignored.add(token)
            ignored.append(token)
    return ids, ignored


_PAXDB_CSV_PATH = Path(__file__).parent / "data" / "raw" / "paxdb" / "human_abundance_uniprot.csv"


@st.cache_data(show_spinner=False)
def _paxdb_ppm() -> pd.Series:
    """Return a UniProt-accession → PaxDb abundance (ppm) Series (empty if missing)."""
    from biastracker.analysis.paxdb import load_paxdb_ppm
    return load_paxdb_ppm(_PAXDB_CSV_PATH)


def _fetch_human_proteome_ids() -> set[str]:
    """Fetch the reviewed human proteome accessions live from the UniProt API.

    Queried fresh on each ORA run so the background is always current. Raises on
    network/HTTP errors so the caller can surface a message instead of silently
    using a stale set.
    """
    import gzip
    import urllib.request

    req = urllib.request.Request(_UNIPROT_HUMAN_QUERY, headers={"User-Agent": "BiasTracker/0.1"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        raw = resp.read()
    if raw[:2] == b"\x1f\x8b":          # gzip magic bytes
        raw = gzip.decompress(raw)
    return {ln.strip() for ln in raw.decode("utf-8", "replace").splitlines() if ln.strip()}


@st.cache_data(show_spinner=False)
def _load_builtin_annotation(rel_path: str, source: str):
    """Load a bundled long-format annotation set. Returns (ann | None, err | None)."""
    path = Path(__file__).parent / rel_path
    if not path.exists():
        return None, f"Built-in annotation file not found: {path}"
    try:
        from biastracker.annotations.custom import load_long_annotation_table
        ann = load_long_annotation_table(
            str(path), name=source, source=source,
            id_col="primary_id", term_col="term_name",
            term_id_col="term_id", category_col="category",
        )
        return ann, None
    except Exception as exc:
        return None, str(exc)


def _available_features(ds) -> list[str]:
    return [
        f for f in _FEAT_META
        if f in ds.table.columns and pd.api.types.is_numeric_dtype(ds.table[f])
    ]


def _lbl(feat: str) -> str:
    return _FEAT_META.get(feat, feat)


def _card_html(val: str, label: str) -> str:
    return (
        f'<div class="bt-card">'
        f'<div class="val">{val}</div>'
        f'<div class="lbl">{label}</div>'
        f'</div>'
    )


def _sig_marker(fdr) -> str:
    try:
        v = float(fdr)
        if v < 0.05:
            return "**"
        if v < 0.1:
            return "*"
    except Exception:
        pass
    return ""


def _standardized_delta(row, ds_a, ds_b) -> float:
    """Δ median expressed in pooled-SD units (a standardized mean difference).

    Dividing by the pooled within-group SD puts every feature on one common,
    unit-free scale so the Δ waterfall isn't dominated by large-magnitude
    features like intensity. Rough Cohen's-d guide: ~0.2 small, ~0.5 medium,
    ~0.8 large.
    """
    if ds_a is None or ds_b is None:
        return np.nan
    feat = row["feature"]
    if feat not in ds_a.table.columns or feat not in ds_b.table.columns:
        return np.nan
    a = ds_a.table[feat].dropna()
    b = ds_b.table[feat].dropna()
    n_a, n_b = len(a), len(b)
    if n_a < 2 or n_b < 2:
        return np.nan
    s_a, s_b = float(a.std(ddof=1)), float(b.std(ddof=1))
    pooled = np.sqrt(((n_a - 1) * s_a ** 2 + (n_b - 1) * s_b ** 2) / (n_a + n_b - 2))
    if pooled == 0 or np.isnan(pooled):
        return np.nan
    return float(row["delta_median"]) / pooled


# ═══════════════════════════════════════════════════════════════
#  Sidebar — dataset loading
# ═══════════════════════════════════════════════════════════════

def render_sidebar() -> dict:
    datasets: dict = {}

    # Dynamic dataset slots — each slot has a stable unique id so that adding or
    # removing one never reshuffles the widget keys of the others.
    if "ds_slots" not in st.session_state:
        st.session_state["ds_slots"]   = [1]
        st.session_state["ds_next_id"] = 2

    with st.sidebar:
        logo_path = Path(__file__).parent / "assets" / "logo.png"
        if logo_path.exists():
            st.image(str(logo_path), use_container_width=True)
        else:
            st.markdown(
                f'<div style="text-align:center;padding:1.2rem 0 1rem 0;">'
                f'<span style="font-size:1.9rem;font-weight:800;color:white;">Bias</span>'
                f'<span style="font-size:1.9rem;font-weight:800;color:{_CYAN};">Tracker</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown(
            '<p style="color:rgba(255,255,255,0.55);font-size:0.78rem;'
            'letter-spacing:0.08em;margin:0 0 0.5rem;">DATASETS</p>',
            unsafe_allow_html=True,
        )

        slots = st.session_state["ds_slots"]
        for pos, slot in enumerate(slots, start=1):
            with st.expander(f"Dataset {pos}", expanded=(pos == len(slots))):
                name = st.text_input("Name", value=f"Dataset {pos}", key=f"ds_name_{slot}")
                nc1, nc2 = st.columns([4, 1])
                type_label = nc1.selectbox("Type", list(_DS_TYPES), key=f"ds_type_{slot}")
                ds_type = _DS_TYPES[type_label]
                # Per-dataset colour, applied consistently across all charts.
                picked = nc2.color_picker(
                    "Colour", value=_default_dataset_color(pos - 1),
                    key=f"ds_color_{slot}",
                    help="This colour represents the dataset in every visualisation "
                         "(distributions, Venn, …).",
                )
                st.session_state.setdefault("ds_colors", {})[name] = picked

                if ds_type in _FIXED_LEVEL:
                    level = _FIXED_LEVEL[ds_type]
                else:
                    level = st.selectbox("Level", ["peptide", "protein"], key=f"ds_level_{slot}")

                ph = st.number_input(
                    "Charge pH",
                    min_value=0.0, max_value=14.0, value=8.5, step=0.1, format="%.2f",
                    key=f"ds_ph_{slot}",
                    help="pH at which net charge (charge_at_pH) is computed for this "
                         "dataset — set it to the pH this analysis was run at "
                         "(proteomics default 8.5). Changing it re-computes the dataset.",
                )

                uploaded = st.file_uploader(
                    "Upload file",
                    type=_EXTENSIONS.get(ds_type, []),
                    key=f"ds_file_{slot}",
                )

                if uploaded is not None:
                    with st.spinner("Loading…"):
                        ds, err = _load(
                            uploaded.getvalue(),
                            uploaded.name,
                            ds_type,
                            name,
                            level,
                            ph,
                        )
                    if err:
                        st.error(f"Error: {err[:300]}")
                    else:
                        if name in datasets:
                            st.warning(f"Name “{name}” already used — rename to avoid overwriting.")
                        datasets[name] = ds
                        st.success(f"✓ {len(ds.table):,} rows ({ds.level})")

                if len(slots) > 1:
                    if st.button("🗑  Remove", key=f"ds_remove_{slot}"):
                        st.session_state["ds_slots"].remove(slot)
                        for k in (f"ds_name_{slot}", f"ds_type_{slot}",
                                  f"ds_level_{slot}", f"ds_ph_{slot}", f"ds_file_{slot}"):
                            st.session_state.pop(k, None)
                        st.rerun()

        if st.button("➕  Add dataset", use_container_width=True):
            st.session_state["ds_slots"].append(st.session_state["ds_next_id"])
            st.session_state["ds_next_id"] += 1
            st.rerun()

        st.markdown(
            f'<p style="color:rgba(255,255,255,0.3);font-size:0.72rem;'
            f'text-align:center;margin-top:1.5rem;">BiasTracker v0.1.0</p>',
            unsafe_allow_html=True,
        )

    return datasets


# ═══════════════════════════════════════════════════════════════
#  Tab: Overview
# ═══════════════════════════════════════════════════════════════

def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert a #RRGGBB string to an rgba() string with the given alpha."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _dataset_id_set(ds) -> set[str]:
    """Unique protein identifiers for a dataset, excluding contaminants.

    Non-quantified proteins are already removed at load (see
    :func:`_apply_lfq_filter`), so this is the set of detected, non-contaminant
    proteins — the meaningful set for cross-dataset overlap.
    """
    if ds.id_col not in ds.table.columns:
        return set()
    df = ds.table
    if "is_contaminant" in df.columns:
        df = df[~df["is_contaminant"].fillna(False).astype(bool)]
    return set(df[ds.id_col].dropna().astype(str))


def _venn_figure(sets: dict[str, set]) -> go.Figure:
    """Build a 2- or 3-set Venn diagram of identifier overlaps with Plotly shapes.

    Circles use fixed (non-area-proportional) geometry; every region is labelled
    with its count so the diagram stays readable regardless of set sizes.
    """
    names = list(sets.keys())
    fig = go.Figure()

    def _circle(cx, cy, r, color):
        fig.add_shape(
            type="circle", xref="x", yref="y",
            x0=cx - r, y0=cy - r, x1=cx + r, y1=cy + r,
            line=dict(color=color, width=2),
            fillcolor=_hex_to_rgba(color, 0.30), layer="below",
        )

    def _count(x, y, n, big=True):
        fig.add_annotation(x=x, y=y, text=f"{n:,}", showarrow=False,
                           font=dict(color=_NAVY, size=16 if big else 13))

    def _title(x, y, label, color):
        fig.add_annotation(x=x, y=y, text=f"<b>{label}</b>", showarrow=False,
                           font=dict(color=color, size=13))

    if len(names) == 2:
        A, B = sets[names[0]], sets[names[1]]
        cA, cB = _dataset_color(names[0], 0), _dataset_color(names[1], 1)
        r = 0.95
        _circle(-0.45, 0, r, cA)
        _circle(0.45, 0, r, cB)
        _count(-0.72, 0, len(A - B))
        _count(0.72, 0, len(B - A))
        _count(0.0, 0, len(A & B))
        _title(-0.55, r + 0.15, names[0], cA)
        _title(0.55, r + 0.15, names[1], cB)
        x_range, y_range = [-1.75, 1.75], [-1.3, 1.4]
    else:  # 3 sets
        A, B, C = sets[names[0]], sets[names[1]], sets[names[2]]
        cA, cB, cC = (_dataset_color(names[0], 0), _dataset_color(names[1], 1),
                      _dataset_color(names[2], 2))
        r = 0.9
        _circle(-0.42, 0.32, r, cA)
        _circle(0.42, 0.32, r, cB)
        _circle(0.0, -0.45, r, cC)
        # Single-set-only regions
        _count(-0.78, 0.68, len(A - B - C))
        _count(0.78, 0.68, len(B - A - C))
        _count(0.0, -1.02, len(C - A - B))
        # Pairwise-only regions
        _count(0.0, 0.72, len((A & B) - C))
        _count(-0.55, -0.28, len((A & C) - B))
        _count(0.55, -0.28, len((B & C) - A))
        # Triple intersection
        _count(0.0, 0.08, len(A & B & C))
        _title(-0.7, 1.32, names[0], cA)
        _title(0.7, 1.32, names[1], cB)
        _title(0.0, -1.5, names[2], cC)
        x_range, y_range = [-1.9, 1.9], [-1.75, 1.6]

    fig.update_xaxes(visible=False, range=x_range)
    fig.update_yaxes(visible=False, range=y_range, scaleanchor="x", scaleratio=1)
    fig.update_layout(
        template="plotly_white",
        height=460,
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    return fig


def _render_overlap_section(datasets: dict) -> None:
    """Venn (2–3 datasets) or an overlap summary (>3) of proteins per dataset."""
    st.markdown('<div class="bt-section">Identification overlap</div>', unsafe_allow_html=True)

    id_sets = {name: _dataset_id_set(ds) for name, ds in datasets.items()}
    id_sets = {n: s for n, s in id_sets.items() if s}
    if len(id_sets) < 2:
        st.caption("Need at least two datasets with quantified proteins to compare.")
        return

    names = list(id_sets.keys())
    shared_all = set.intersection(*id_sets.values())
    id_col = datasets[names[0]].id_col

    if len(names) <= 3:
        st.plotly_chart(_venn_figure(id_sets), use_container_width=True)
        st.caption(
            f"Counts are unique quantified proteins ({id_col}); contaminants excluded. "
            f"{len(shared_all):,} shared across all {len(names)}."
        )
    else:
        # >3 datasets: a proportional Venn is not meaningful for all of them at
        # once. Let the user Venn any 2–3, and show a pairwise overlap matrix.
        pick = st.multiselect(
            "Datasets to compare (pick 2–3 for a Venn)", names,
            default=names[:3], max_selections=3, key="venn_pick",
        )
        if 2 <= len(pick) <= 3:
            st.plotly_chart(_venn_figure({n: id_sets[n] for n in pick}),
                            use_container_width=True)
        st.caption(
            f"{len(names)} datasets loaded (quantified proteins, contaminants "
            f"excluded). Pairwise overlap matrix below (diagonal = dataset size); "
            f"{len(shared_all):,} proteins shared across all {len(names)}."
        )
        mat = pd.DataFrame(index=names, columns=names, dtype=int)
        for a in names:
            for b in names:
                mat.loc[a, b] = len(id_sets[a]) if a == b else len(id_sets[a] & id_sets[b])
        st.dataframe(mat, use_container_width=True)

    # Per-dataset unique counts, useful alongside either view.
    uniq_rows = []
    for n in names:
        others = set.union(*[id_sets[m] for m in names if m != n]) if len(names) > 1 else set()
        uniq_rows.append({"Dataset": n, "Total IDs": len(id_sets[n]),
                          "Unique to it": len(id_sets[n] - others)})
    with st.expander("Unique / shared counts", expanded=False):
        st.dataframe(pd.DataFrame(uniq_rows), use_container_width=True, hide_index=True)


def tab_overview(datasets: dict) -> None:
    if not datasets:
        st.info("⬅ Upload at least one dataset from the sidebar to get started.")
        return

    from biastracker.analysis.summary import summarize_dataset

    for name, ds in datasets.items():
        st.markdown(f'<div class="bt-section">{name}</div>', unsafe_allow_html=True)

        n_rows   = len(ds.table)
        n_unique = ds.table[ds.id_col].nunique() if ds.id_col in ds.table.columns else "—"
        # Proteins dropped at load for having no LFQ (see _apply_lfq_filter).
        n_removed = ds.metadata.get("n_removed_no_lfq")
        n_seq    = int(
            (ds.table["sequence"].notna() &
             (ds.table["sequence"].astype(str).str.strip() != "")).sum()
        )
        n_feats  = len(_available_features(ds))

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.markdown(_card_html(f"{n_rows:,}", f"Rows ({ds.level})"), unsafe_allow_html=True)
        c2.markdown(
            _card_html(
                f"{n_unique:,}" if isinstance(n_unique, int) else str(n_unique),
                "Unique IDs",
            ),
            unsafe_allow_html=True,
        )
        c3.markdown(
            _card_html(f"{n_removed:,}" if n_removed is not None else "—", "Removed (no LFQ)"),
            unsafe_allow_html=True,
        )
        c4.markdown(_card_html(f"{n_seq:,}", "With Sequence"), unsafe_allow_html=True)
        c5.markdown(_card_html(str(n_feats), "Features Available"), unsafe_allow_html=True)

        with st.expander("Feature summary table", expanded=False):
            feats = _available_features(ds)
            st.dataframe(
                summarize_dataset(ds, features=feats),
                use_container_width=True,
                hide_index=True,
            )

        st.markdown("")

    # Cross-dataset identification overlap (Venn for 2–3 datasets).
    _render_overlap_section(datasets)


# ═══════════════════════════════════════════════════════════════
#  Tab: Distributions
# ═══════════════════════════════════════════════════════════════

def tab_distributions(datasets: dict) -> None:
    if not datasets:
        st.info("⬅ Upload at least one dataset from the sidebar.")
        return

    all_feats = sorted({f for ds in datasets.values() for f in _available_features(ds)})
    if not all_feats:
        st.warning("No numeric features found in the loaded datasets.")
        return

    c1, c2, c3, c4 = st.columns([3, 3, 1, 1])
    feat   = c1.selectbox("Property", all_feats, format_func=_lbl)
    ptype  = c2.radio("Plot type", ["Histogram", "Violin", "CDF"], horizontal=True)
    rug    = c3.checkbox("Rug", value=False, help="Add a rug plot: small tick marks along the axis "
                         "showing each individual data point's position.")

    # A log₁₀ transform is undefined for non-positive values. Forbid it entirely
    # for properties that can be negative (e.g. GRAVY, net charge) rather than
    # silently dropping those points, which would misrepresent the distribution.
    feat_has_nonpos = any(
        (ds.table[feat].dropna() <= 0).any()
        for ds in datasets.values()
        if feat in ds.table.columns
    )
    logx   = c4.checkbox(
        "Log₁₀",
        value=False,
        disabled=feat_has_nonpos,
        help=(
            f"Log₁₀ is disabled for “{_lbl(feat)}” because it contains non-positive "
            "values (log₁₀ is only defined for values > 0)."
            if feat_has_nonpos
            else "Log₁₀-transform values (drops ≤ 0)"
        ),
    )
    if feat_has_nonpos:
        logx = False

    axis_lbl = f"log₁₀ {_lbl(feat)}" if logx else _lbl(feat)
    n_dropped = 0

    def _prep(values: np.ndarray) -> np.ndarray:
        """Optionally log₁₀-transform, discarding non-positive values."""
        nonlocal n_dropped
        if not logx:
            return values
        pos = values[values > 0]
        n_dropped += len(values) - len(pos)
        return np.log10(pos)

    fig = go.Figure()
    fig.update_layout(
        template="plotly_white",
        xaxis_title=axis_lbl,
        yaxis_title={
            "Histogram": "Count",
            "Violin":    axis_lbl,
            "CDF":       "Cumulative Probability",
        }[ptype],
        barmode="overlay",
        legend=dict(orientation="h", y=1.06, x=1, xanchor="right"),
        height=460,
        margin=dict(l=50, r=20, t=30, b=50),
        font=dict(color=_NAVY, size=13),
    )

    for i, (name, ds) in enumerate(datasets.items()):
        if feat not in ds.table.columns:
            continue
        data  = _prep(ds.table[feat].dropna().values)
        color = _dataset_color(name, i)
        if len(data) == 0:
            continue

        if ptype == "Histogram":
            fig.add_trace(go.Histogram(
                x=data, name=name,
                opacity=0.65, marker_color=color, nbinsx=60,
            ))
        elif ptype == "Violin":
            fig.add_trace(go.Violin(
                y=data, name=name,
                box_visible=True, meanline_visible=True,
                fillcolor=color, opacity=0.7, line_color=color,
            ))
        else:  # CDF
            sx = np.sort(data)
            fig.add_trace(go.Scatter(
                x=sx, y=np.arange(1, len(sx) + 1) / len(sx),
                name=name, mode="lines",
                line=dict(color=color, width=2.5),
            ))

        if rug and ptype != "Violin":
            fig.add_trace(go.Scatter(
                x=data,
                y=np.full_like(data, -0.015 - 0.012 * i, dtype=float),
                mode="markers",
                marker=dict(symbol="line-ns-open", color=color, size=3, opacity=0.35),
                showlegend=False,
                hoverinfo="skip",
            ))

    st.plotly_chart(fig, use_container_width=True)
    if feat_has_nonpos:
        st.caption(f"Log₁₀ unavailable for “{_lbl(feat)}” — it has non-positive values.")
    elif logx and n_dropped:
        st.caption(f"Log₁₀ scale — {n_dropped:,} non-positive value(s) dropped.")

    # Side-by-side descriptive stats when multiple datasets loaded
    if len(datasets) > 1:
        rows = []
        for name, ds in datasets.items():
            if feat in ds.table.columns:
                s = ds.table[feat].dropna()
                rows.append({
                    "Dataset": name,
                    "N": int(s.count()),
                    "Mean":   round(float(s.mean()), 4),
                    "Median": round(float(s.median()), 4),
                    "Std":    round(float(s.std()), 4),
                    "Min":    round(float(s.min()), 4),
                    "Max":    round(float(s.max()), 4),
                })
        if rows:
            st.markdown('<div class="bt-section">Descriptive Statistics</div>', unsafe_allow_html=True)
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════
#  Tab: Compare
# ═══════════════════════════════════════════════════════════════

def tab_compare(datasets: dict) -> None:
    if len(datasets) < 2:
        st.info("Upload at least **two datasets** to run a comparison.")
        return

    from biastracker.analysis.compare import compare_datasets, compare_multiple_datasets

    names = list(datasets.keys())
    selected = st.multiselect(
        "Datasets to compare",
        names,
        default=names,
        help="Pick 2 for a pairwise test (Mann-Whitney U + KS), "
             "or 3+ for an omnibus Kruskal-Wallis test across all of them.",
    )

    if len(selected) < 2:
        st.info("Select at least **two datasets**.")
        return

    mode     = "pairwise" if len(selected) == 2 else "multi"
    test_lbl = ("Mann-Whitney U + KS" if mode == "pairwise"
                else f"Kruskal-Wallis across {len(selected)} datasets")

    if st.button(f"▶  Run Comparison  ·  {test_lbl}", type="primary"):
        with st.spinner(f"Running {test_lbl} with FDR correction…"):
            try:
                if mode == "pairwise":
                    a, b = selected
                    res = compare_datasets(datasets[a], datasets[b])
                    st.session_state["cmp_a"]     = a
                    st.session_state["cmp_b"]     = b
                    st.session_state["cmp_label"] = f"{a} vs {b}"
                else:
                    res = compare_multiple_datasets([datasets[n] for n in selected])
                    st.session_state["cmp_label"] = " vs ".join(selected)
                st.session_state["cmp_res"]  = res
                st.session_state["cmp_mode"] = mode
                st.session_state["cmp_sel"]  = selected
            except Exception as exc:
                st.error(str(exc))
                return

    if "cmp_res" not in st.session_state:
        return

    if st.session_state.get("cmp_mode") == "multi":
        _render_multi_compare(datasets)
    else:
        _render_pairwise_compare(datasets)


def _render_pairwise_compare(datasets: dict) -> None:
    res: pd.DataFrame = st.session_state["cmp_res"]
    cmp_a: str        = st.session_state.get("cmp_a", "A")
    cmp_b: str        = st.session_state.get("cmp_b", "B")
    label: str        = st.session_state["cmp_label"]

    st.markdown(f'<div class="bt-section">Results — {label}</div>', unsafe_allow_html=True)

    # ── Chart controls ───────────────────────────────────────────────────────
    cc1, cc2 = st.columns([2, 2])
    scale = cc1.radio(
        "Δ scale", ["Raw", "Standardized (effect size)"],
        horizontal=True,
        help="Raw shows Δ median in each feature's own units. Standardized "
             "divides by the pooled SD so features are comparable (like Cohen's d).",
    )
    hide_int = cc2.checkbox(
        "Hide intensity-type features", value=False,
        help="Hide abundance / large-magnitude features (expression, extinction "
             "coefficients, PTR) that otherwise dominate the raw axis.",
    )
    standardized = scale.startswith("Standardized")

    # ── Waterfall chart: Δ per feature ───────────────────────────────────────
    pdf = res.copy()
    if hide_int:
        pdf = pdf[~pdf["feature"].isin(_INTENSITY_FEATS)]

    if standardized:
        ds_a, ds_b = datasets.get(cmp_a), datasets.get(cmp_b)
        pdf["plot_val"] = pdf.apply(lambda r: _standardized_delta(r, ds_a, ds_b), axis=1)
        pdf = pdf.dropna(subset=["plot_val"])
        x_title = f"Standardized Δ  ({cmp_a} − {cmp_b}),  pooled-SD units"
    else:
        pdf["plot_val"] = pdf["delta_median"]
        x_title = f"Δ Median  ({cmp_a} − {cmp_b})"

    pdf["feature_label"] = pdf["feature"].map(_lbl)
    pdf["sig"] = pdf["mannwhitney_fdr"].apply(
        lambda x: "FDR < 0.05" if x < 0.05 else ("FDR < 0.1" if x < 0.1 else "n.s.")
    )
    pdf = pdf.sort_values("plot_val")

    sig_palette = {"FDR < 0.05": "#d32f2f", "FDR < 0.1": "#f57c00", "n.s.": "#BBBBBB"}
    wfig = go.Figure()
    for sig_level, color in sig_palette.items():
        sub = pdf[pdf["sig"] == sig_level]
        if sub.empty:
            continue
        wfig.add_trace(go.Bar(
            x=sub["plot_val"], y=sub["feature_label"],
            orientation="h", name=sig_level,
            marker_color=color, opacity=0.85,
        ))
    wfig.add_vline(x=0, line_dash="dash", line_color=_NAVY, line_width=1.2)
    wfig.update_layout(
        template="plotly_white",
        xaxis_title=x_title,
        barmode="overlay",
        height=max(300, len(pdf) * 40 + 100),
        legend_title="Significance",
        margin=dict(l=160, r=20, t=30, b=50),
        font=dict(color=_NAVY, size=12),
    )
    st.plotly_chart(wfig, use_container_width=True)
    if standardized:
        st.caption(
            "Standardized Δ = (median A − median B) / pooled SD — comparable "
            "across features. Rule of thumb: |Δ| ≈ 0.2 small, 0.5 medium, 0.8 large."
        )

    # ── Results table ─────────────────────────────────────────────────────────
    keep = [
        "feature", "n_a", "n_b",
        "median_a", "median_b", "delta_median", "direction",
        "mannwhitney_p", "mannwhitney_fdr", "ks_fdr",
    ]
    disp = res[[c for c in keep if c in res.columns]].copy()
    disp["sig"] = res["mannwhitney_fdr"].apply(_sig_marker)
    disp = disp.rename(columns={
        "feature":           "Feature",
        "n_a":               f"N ({cmp_a})",
        "n_b":               f"N ({cmp_b})",
        "median_a":          f"Median ({cmp_a})",
        "median_b":          f"Median ({cmp_b})",
        "delta_median":      "Δ Median",
        "direction":         "Direction",
        "mannwhitney_p":     "MW p",
        "mannwhitney_fdr":   "MW FDR",
        "ks_fdr":            "KS FDR",
        "sig":               "Sig",
    })
    fmt_cols = {
        c: "{:.4g}" for c in disp.columns
        if any(k in c for k in ["Median", "Δ", "p", "FDR"])
    }
    st.dataframe(
        disp.style.format(fmt_cols),
        use_container_width=True,
        hide_index=True,
    )
    st.caption("Sig: ** = FDR < 0.05,  * = FDR < 0.1")

    st.download_button(
        "⬇  Download comparison CSV",
        data=res.to_csv(index=False).encode(),
        file_name=f"comparison_{cmp_a}_vs_{cmp_b}.csv".replace(" ", "_"),
        mime="text/csv",
    )


def _render_multi_compare(datasets: dict) -> None:
    res: pd.DataFrame = st.session_state["cmp_res"]
    label: str        = st.session_state["cmp_label"]
    selected: list    = st.session_state.get("cmp_sel", list(datasets.keys()))

    st.markdown(f'<div class="bt-section">Results — {label}</div>', unsafe_allow_html=True)
    st.caption(
        "Omnibus **Kruskal-Wallis** test per feature: does the distribution "
        "differ across *any* of the selected datasets?"
    )

    # Features tested = intersection across all datasets. Flag any that were
    # dropped because at least one selected dataset lacked them.
    tested = set(res["feature"])
    available: dict[str, set] = {}
    for n in selected:
        ds = datasets.get(n)
        if ds is not None:
            for f in _available_features(ds):
                available.setdefault(f, set()).add(n)
    dropped = {f: v for f, v in available.items() if f not in tested}
    if dropped:
        detail = "; ".join(
            f"{_lbl(f)} (only in {', '.join(sorted(present))})"
            for f, present in sorted(dropped.items())
        )
        st.warning(
            f"{len(dropped)} feature(s) excluded — not present in every selected "
            f"dataset: {detail}"
        )

    hide_int = st.checkbox(
        "Hide intensity-type features", value=False, key="multi_hide_int",
        help="Hide abundance / large-magnitude features (expression, extinction "
             "coefficients, PTR) from the charts and tables below.",
    )
    res_v = res[~res["feature"].isin(_INTENSITY_FEATS)] if hide_int else res

    # ── Significance chart: −log₁₀ FDR per feature ───────────────────────────
    pdf = res_v.copy()
    pdf["feature_label"] = pdf["feature"].map(_lbl)
    pdf["neglog_fdr"]    = -np.log10(pdf["kruskal_fdr"].clip(lower=1e-300))
    pdf["sig"] = pdf["kruskal_fdr"].apply(
        lambda x: "FDR < 0.05" if x < 0.05 else ("FDR < 0.1" if x < 0.1 else "n.s.")
    )
    pdf = pdf.sort_values("neglog_fdr")

    sig_palette = {"FDR < 0.05": "#d32f2f", "FDR < 0.1": "#f57c00", "n.s.": "#BBBBBB"}
    bfig = go.Figure()
    for sig_level, color in sig_palette.items():
        sub = pdf[pdf["sig"] == sig_level]
        if sub.empty:
            continue
        bfig.add_trace(go.Bar(
            x=sub["neglog_fdr"], y=sub["feature_label"],
            orientation="h", name=sig_level,
            marker_color=color, opacity=0.85,
        ))
    bfig.add_vline(
        x=-np.log10(0.05), line_dash="dash", line_color=_NAVY, line_width=1.2,
        annotation_text="FDR 0.05", annotation_position="top",
    )
    bfig.update_layout(
        template="plotly_white",
        xaxis_title="−log₁₀ FDR  (Kruskal-Wallis)",
        barmode="overlay",
        height=max(300, len(pdf) * 40 + 100),
        legend_title="Significance",
        margin=dict(l=160, r=20, t=30, b=50),
        font=dict(color=_NAVY, size=12),
    )
    st.plotly_chart(bfig, use_container_width=True)

    # ── Median of each feature per dataset (context for the omnibus test) ────
    med_rows = []
    for feat in res_v["feature"]:
        row = {"Feature": _lbl(feat)}
        for n in selected:
            ds = datasets.get(n)
            row[n] = (
                round(float(ds.table[feat].dropna().median()), 4)
                if ds is not None and feat in ds.table.columns
                else np.nan
            )
        med_rows.append(row)
    if med_rows:
        st.markdown('<div class="bt-section">Median per Dataset</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(med_rows), use_container_width=True, hide_index=True)

    # ── Kruskal-Wallis results table ─────────────────────────────────────────
    keep = ["feature", "n_groups", "groups", "kruskal_statistic", "kruskal_p", "kruskal_fdr"]
    disp = res_v[[c for c in keep if c in res_v.columns]].copy()
    disp["sig"] = res_v["kruskal_fdr"].apply(_sig_marker)
    disp = disp.rename(columns={
        "feature":            "Feature",
        "n_groups":           "N groups",
        "groups":             "Groups",
        "kruskal_statistic":  "KW H",
        "kruskal_p":          "KW p",
        "kruskal_fdr":        "KW FDR",
        "sig":                "Sig",
    })
    fmt_cols = {c: "{:.4g}" for c in disp.columns if c in ("KW H", "KW p", "KW FDR")}
    st.markdown('<div class="bt-section">Test Statistics</div>', unsafe_allow_html=True)
    st.dataframe(disp.style.format(fmt_cols), use_container_width=True, hide_index=True)
    st.caption("Sig: ** = FDR < 0.05,  * = FDR < 0.1")

    st.download_button(
        "⬇  Download comparison CSV",
        data=res.to_csv(index=False).encode(),
        file_name=f"comparison_{label.replace(' ', '_')}.csv",
        mime="text/csv",
    )


# ═══════════════════════════════════════════════════════════════
#  Tab: Enrichment (ORA + fgsea)
# ═══════════════════════════════════════════════════════════════

def _trunc(s: str, n: int = 45) -> str:
    s = str(s)
    return s if len(s) <= n else s[: n - 1] + "…"


def _enrichment_bar(pdf: pd.DataFrame, x_col: str, x_title: str, pos_label: str,
                    neg_label: str, top_n: int = 20) -> None:
    """Horizontal bar of the top terms, coloured by direction (pos/neg of x_col)."""
    sub = pdf.reindex(pdf[x_col].abs().sort_values(ascending=False).index).head(top_n)
    sub = sub.sort_values(x_col)
    # Enriched / high-score terms are warm (red); depleted / low-score are teal.
    colors = ["#FF6B6B" if v >= 0 else _CYAN for v in sub[x_col]]
    fig = go.Figure(go.Bar(
        x=sub[x_col], y=[_trunc(t) for t in sub["term_name"]],
        orientation="h", marker_color=colors, opacity=0.85,
    ))
    fig.add_vline(x=0, line_dash="dash", line_color=_NAVY, line_width=1.2)
    fig.update_layout(
        template="plotly_white",
        xaxis_title=x_title,
        height=max(300, len(sub) * 30 + 120),
        margin=dict(l=260, r=20, t=30, b=50),
        font=dict(color=_NAVY, size=12),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Red = {pos_label}, teal = {neg_label}. Showing top {len(sub)} terms.")


def _enrichment_volcano(vdf: pd.DataFrame, sig_threshold: float, effect_label: str,
                        n_labels: int = 8) -> None:
    """Volcano plot of enrichment terms: effect size (x) vs −log₁₀ FDR (y).

    *vdf* must be the output of
    :func:`biastracker.analysis.enrichment.prepare_enrichment_volcano_data`.
    """
    if vdf.empty:
        st.info("No terms to plot.")
        return

    plot = vdf.dropna(subset=["effect", "neg_log10_fdr"]).copy()
    if plot.empty:
        st.info("No terms with a finite effect size and FDR to plot.")
        return

    # Three visual groups: significant up (red), significant down (teal),
    # non-significant (grey). This encodes direction *and* significance.
    def _group(r):
        if not r["significant"]:
            return "n.s."
        return "enriched (top)" if r["direction"] == "positive" else "enriched (bottom)"

    plot["grp"] = plot.apply(_group, axis=1)
    palette = {"enriched (top)": "#FF6B6B", "enriched (bottom)": _CYAN, "n.s.": "#BBBBBB"}

    fig = go.Figure()
    for grp, color in palette.items():
        sub = plot[plot["grp"] == grp]
        if sub.empty:
            continue
        p_txt = sub["p_value"] if "p_value" in sub.columns else pd.Series([np.nan] * len(sub))
        size_txt = sub["set_size"] if "set_size" in sub.columns else pd.Series([np.nan] * len(sub))
        customdata = np.column_stack([
            sub["fdr"].to_numpy(),
            p_txt.to_numpy(),
            size_txt.to_numpy(),
        ])
        fig.add_trace(go.Scatter(
            x=sub["effect"], y=sub["neg_log10_fdr"],
            mode="markers", name=grp,
            marker=dict(color=color, size=8, opacity=0.75,
                        line=dict(width=0.5, color="white")),
            text=sub["term_name"].astype(str),
            customdata=customdata,
            hovertemplate=(
                "<b>%{text}</b><br>"
                f"{effect_label}: %{{x:.3f}}<br>"
                "FDR: %{customdata[0]:.3g}<br>"
                "p: %{customdata[1]:.3g}<br>"
                "set size: %{customdata[2]:.0f}<extra></extra>"
            ),
        ))

    # Reference lines: significance threshold and effect = 0.
    y_thr = -np.log10(max(sig_threshold, 1e-300))
    fig.add_hline(y=y_thr, line_dash="dash", line_color=_NAVY, line_width=1,
                  annotation_text=f"FDR = {sig_threshold:g}", annotation_position="top left")
    fig.add_vline(x=0, line_dash="dash", line_color=_NAVY, line_width=1)

    # Label a few of the most significant terms, preferring significant ones.
    labelled = plot.sort_values("neg_log10_fdr", ascending=False)
    labelled = pd.concat([labelled[labelled["significant"]], labelled[~labelled["significant"]]])
    for _, r in labelled.head(int(n_labels)).iterrows():
        fig.add_annotation(
            x=r["effect"], y=r["neg_log10_fdr"], text=_trunc(str(r["term_name"]), 28),
            showarrow=True, arrowhead=0, arrowwidth=0.6, arrowcolor="#999",
            font=dict(size=10, color=_NAVY), ax=0, ay=-14,
        )

    fig.update_layout(
        template="plotly_white",
        xaxis_title=effect_label,
        yaxis_title="−log₁₀ FDR",
        height=520,
        legend=dict(orientation="h", y=1.06, x=1, xanchor="right"),
        margin=dict(l=60, r=20, t=30, b=50),
        font=dict(color=_NAVY, size=12),
    )
    st.plotly_chart(fig, use_container_width=True)
    n_sig = int(plot["significant"].sum())
    st.caption(
        f"Red = enriched at the high-score (top) end, teal = enriched at the low-score "
        f"(bottom) end, grey = not significant. {n_sig:,} term(s) at FDR ≤ {sig_threshold:g}."
    )


def _paxdb_scatter(res, dataset_name: str, trend: bool = True) -> None:
    """Scatter of dataset mean abundance vs PaxDb abundance, with Spearman ρ.

    *res* is a
    :class:`biastracker.analysis.paxdb.PaxDbCorrelationResult`.
    """
    m = res.matched
    if m.empty:
        st.info("No proteins could be matched to PaxDb for this dataset.")
        return

    title = f"ρ = {res.rho:.3f}" if np.isfinite(res.rho) else "ρ = n/a"
    title += f"  ·  n = {res.n_used:,} matched proteins"

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=m["dataset_abundance"], y=m["paxdb_abundance"],
        mode="markers",
        marker=dict(color=_dataset_color(dataset_name), size=6, opacity=0.55,
                    line=dict(width=0.3, color="white")),
        text=m["primary_id"].astype(str),
        hovertemplate=("<b>%{text}</b><br>"
                       f"{res.x_label}: %{{x:.3f}}<br>"
                       f"{res.y_label}: %{{y:.3f}}<extra></extra>"),
        name="proteins",
    ))

    # Optional visual trend line (least-squares). This is a display aid only —
    # the reported statistic is the rank-based Spearman ρ, not this OLS fit.
    if trend and len(m) >= 2:
        x = m["dataset_abundance"].to_numpy(dtype=float)
        y = m["paxdb_abundance"].to_numpy(dtype=float)
        slope, intercept = np.polyfit(x, y, 1)
        xs = np.array([x.min(), x.max()])
        fig.add_trace(go.Scatter(
            x=xs, y=slope * xs + intercept, mode="lines",
            line=dict(color=_NAVY, width=1.5, dash="dash"),
            name="trend (OLS)", hoverinfo="skip",
        ))

    fig.update_layout(
        template="plotly_white",
        title=dict(text=title, x=0.5, font=dict(size=14, color=_NAVY)),
        xaxis_title=res.x_label, yaxis_title=res.y_label,
        height=460, margin=dict(l=60, r=20, t=50, b=50),
        font=dict(color=_NAVY, size=12),
        showlegend=True, legend=dict(orientation="h", y=1.02, x=1, xanchor="right"),
    )
    st.plotly_chart(fig, use_container_width=True)


def _run_ora_ui(datasets: dict, annotations) -> None:
    from biastracker.analysis.enrichment import run_ora

    names = list(datasets.keys())
    bg_options = ["All loaded datasets (union)", _HUMAN_PROTEOME_LABEL, _CUSTOM_BG_LABEL] + names
    c1, c2, c3 = st.columns(3)
    query_name = c1.selectbox("Query dataset", names, key="ora_query",
                              help="Proteins in this dataset form the query set.")
    bg_choice  = c2.selectbox("Background universe", bg_options,
                              key="ora_bg",
                              help="The reference set the query is tested against. The whole "
                                   "human proteome tests vs all human proteins (classic ORA); "
                                   "the dataset options isolate method/detection bias; upload "
                                   "your own list to restrict the universe (e.g. a tissue proteome).")
    min_term   = c3.number_input("Min term size", 1, 500, 3, key="ora_min")

    custom_bg_file = None
    if bg_choice == _CUSTOM_BG_LABEL:
        custom_bg_file = st.file_uploader(
            "Background accession list",
            type=["tsv", "csv", "txt", "list"],
            key="ora_bg_file",
            help="One UniProt accession per line, or any tab/comma-separated table "
                 "(headers and extra columns are ignored). The query is tested only "
                 "against proteins in this universe.",
        )

    if st.button("▶  Run ORA", type="primary", key="ora_run"):
        qds = datasets[query_name]
        query_ids = set(qds.table[qds.id_col].dropna().astype(str))
        if bg_choice.startswith("All"):
            background_ids: set[str] = set()
            for ds in datasets.values():
                background_ids |= set(ds.table[ds.id_col].dropna().astype(str))
        elif bg_choice == _CUSTOM_BG_LABEL:
            if custom_bg_file is None:
                st.warning("Upload a background file first, or pick another background.")
                return
            background_ids, ignored = _parse_background_ids(custom_bg_file)
            if not background_ids:
                st.error("No UniProt accessions could be parsed from the uploaded file. "
                         "Expected UniProt accessions (e.g. P12345), one per line or in a column.")
                return
            if ignored:
                examples = ", ".join(ignored[:5]) + ("…" if len(ignored) > 5 else "")
                st.warning(
                    f"{len(ignored):,} entr{'y was' if len(ignored) == 1 else 'ies were'} "
                    f"ignored (not UniProt IDs): {examples}"
                )
            covered = len(query_ids & background_ids)
            st.caption(
                f"Custom background: {len(background_ids):,} UniProt accessions · "
                f"{covered:,}/{len(query_ids):,} query proteins fall within it."
            )
        elif bg_choice == _HUMAN_PROTEOME_LABEL:
            try:
                with st.spinner("Fetching current human proteome from UniProt…"):
                    background_ids = _fetch_human_proteome_ids()
            except Exception as exc:
                st.error(f"Could not fetch human proteome from UniProt: {str(exc)[:200]}")
                return
            if not background_ids:
                st.error("UniProt returned no accessions for the human proteome query.")
                return
            # Query IDs outside the proteome (e.g. non-human contaminants) are
            # dropped by run_ora; keep the universe = proteome ∪ query so those
            # do not silently shrink the background.
            background_ids = background_ids | query_ids
        else:
            bds = datasets[bg_choice]
            background_ids = set(bds.table[bds.id_col].dropna().astype(str)) | query_ids

        if query_ids == background_ids:
            st.warning("Query and background sets are identical — ORA needs a larger "
                       "background than the query. Pick a different background.")
            return
        with st.spinner("Running Fisher's exact tests with FDR correction…"):
            try:
                res = run_ora(query_ids, background_ids, annotations, min_term_size=int(min_term))
            except Exception as exc:
                st.error(str(exc))
                return
        st.session_state["ora_res"]   = res
        st.session_state["ora_label"] = f"{query_name} vs {bg_choice}"

    if "ora_res" not in st.session_state:
        return
    res: pd.DataFrame = st.session_state["ora_res"]
    label = st.session_state.get("ora_label", "ORA")
    st.markdown(f'<div class="bt-section">ORA — {label}</div>', unsafe_allow_html=True)
    if res.empty:
        st.warning("No terms passed the size threshold. Check that annotation IDs match "
                   "your dataset IDs (e.g. UniProt accessions).")
        return

    n_sig = int((res["fdr"] < 0.05).sum())
    st.caption(f"{len(res):,} terms tested · {n_sig:,} significant at FDR < 0.05")

    plot = res.copy()
    plot["signed_score"] = -np.log10(plot["fdr"].clip(lower=1e-300)) * np.where(
        plot["direction"] == "depleted", -1, 1)
    _enrichment_bar(plot, "signed_score", "−log₁₀ FDR  (signed by direction)",
                    pos_label="enriched", neg_label="depleted")

    disp = res.rename(columns={
        "term_name": "Term", "category": "Category", "query_count": "Query hits",
        "background_count": "BG hits", "odds_ratio": "Odds ratio",
        "p_value": "p", "fdr": "FDR", "direction": "Direction",
    })
    keep = ["Term", "Category", "Query hits", "BG hits", "Odds ratio", "p", "FDR", "Direction"]
    disp = disp[[c for c in keep if c in disp.columns]]
    st.dataframe(
        disp.style.format({c: "{:.4g}" for c in ("Odds ratio", "p", "FDR")}),
        use_container_width=True, hide_index=True,
    )
    st.download_button(
        "⬇  Download ORA results CSV",
        data=res.to_csv(index=False).encode(),
        file_name=f"ora_{label.replace(' ', '_')}.csv",
        mime="text/csv", key="ora_dl",
    )


def _run_fgsea_ui(datasets: dict, annotations) -> None:
    from biastracker.analysis.enrichment import (
        DEFAULT_SIG_THRESHOLD, prepare_enrichment_volcano_data, run_fgsea,
    )
    from biastracker.analysis.ranking import (
        CUSTOM, FGSEA_RANKING_METHODS, MEAN_EXPRESSION,
        compute_mean_expression, detect_expression_columns, prepare_fgsea_ranking,
    )

    names = list(datasets.keys())
    c1, c2 = st.columns(2)
    ds_name = c1.selectbox("Dataset", names, key="gsea_ds")
    ds = datasets[ds_name]

    # ── Ranking metric: mean expression (default) or a custom numeric column ──
    method = c2.radio(
        "Ranking metric", list(FGSEA_RANKING_METHODS),
        format_func=lambda m: FGSEA_RANKING_METHODS[m], key="gsea_method",
        help="Mean expression ranks proteins by the row-wise mean of the LFQ / "
             "expression columns. Custom metric ranks by a numeric column you "
             "provide (e.g. a score shipped with a homemade dataset).",
    )

    expression_columns: list[str] | None = None
    custom_col: str | None = None
    ranking_ok = True

    if method == MEAN_EXPRESSION:
        lfq_cols = detect_expression_columns(ds.table)
        if lfq_cols:
            expression_columns = st.multiselect(
                "LFQ / expression columns to average",
                lfq_cols, default=lfq_cols, key="gsea_lfq_cols",
                help="Proteins are ranked by the row-wise mean of these columns.",
            )
            if not expression_columns:
                st.warning("Select at least one expression column.")
                ranking_ok = False
        else:
            # No per-sample columns (e.g. DIA-NN) — fall back to precomputed mean.
            try:
                compute_mean_expression(ds.table)  # validates a usable column exists
                st.caption("No per-sample LFQ columns detected — ranking by the "
                           "precomputed mean expression column.")
            except ValueError as exc:
                st.warning(str(exc))
                ranking_ok = False
    else:  # CUSTOM
        numeric_cols = [c for c in ds.table.columns
                        if pd.api.types.is_numeric_dtype(ds.table[c])]
        if not numeric_cols:
            st.warning("This dataset has no numeric column to use as a custom metric.")
            ranking_ok = False
        else:
            # Preselect the dataset's own expression column for homemade datasets.
            default_idx = 0
            if ds.source_type in {"manual", "standard_csv"} and "expression" in numeric_cols:
                default_idx = numeric_cols.index("expression")
            custom_col = st.selectbox(
                "Custom ranking column", numeric_cols, index=default_idx,
                format_func=lambda c: _FEAT_META.get(c, c), key="gsea_custom_col",
                help="Numeric column to rank proteins by (descending).",
            )

    a1, a2, a3, a4 = st.columns(4)
    min_term = a1.number_input("Min term size", 1, 500, 5, key="gsea_min")
    max_term = a2.number_input("Max term size (0 = none)", 0, 5000, 500, key="gsea_max")
    n_perm   = a3.number_input("Permutations", 100, 10000, 1000, step=100, key="gsea_perm")
    weight   = a4.number_input("Weight", 0.0, 2.0, 1.0, step=0.5, key="gsea_weight")

    if st.button("▶  Run fgsea", type="primary", key="gsea_run", disabled=not ranking_ok):
        with st.spinner(f"Ranking · {int(n_perm)} permutations…"):
            try:
                ranked = prepare_fgsea_ranking(
                    ds.table, id_col=ds.id_col, method=method,
                    expression_columns=expression_columns, custom_col=custom_col,
                )
            except ValueError as exc:
                st.warning(str(exc))
                return
            try:
                res = run_fgsea(
                    ranked, annotations=annotations,
                    min_term_size=int(min_term), max_term_size=(int(max_term) or None),
                    n_permutations=int(n_perm), weight=float(weight), seed=0,
                )
            except Exception as exc:
                st.error(str(exc))
                return
            metric_lbl = ("mean expression" if method == MEAN_EXPRESSION
                          else f"custom metric ({custom_col})")
            st.session_state["gsea_res"]   = res
            st.session_state["gsea_label"] = f"{ds_name} ranked by {metric_lbl}"

    # ── PaxDb abundance-agreement panel (independent of the fGSEA run) ─────────
    _paxdb_agreement_panel(ds, ds_name)

    if "gsea_res" not in st.session_state:
        return
    res: pd.DataFrame = st.session_state["gsea_res"]
    label = st.session_state.get("gsea_label", "fgsea")
    st.markdown(f'<div class="bt-section">fgsea — {label}</div>', unsafe_allow_html=True)
    if res.empty:
        st.warning("No terms passed the size thresholds / ID overlap. Check that "
                   "annotation IDs match your dataset IDs.")
        return

    n_sig = int((res["fdr"] < DEFAULT_SIG_THRESHOLD).sum())
    st.caption(f"{len(res):,} gene sets tested · {n_sig:,} significant at "
               f"FDR < {DEFAULT_SIG_THRESHOLD:g}")

    # Volcano is the default view; the bar chart remains available.
    view = st.radio("Plot", ["Volcano", "Bar chart"], horizontal=True, key="gsea_plot")
    if view == "Volcano":
        vdf = prepare_enrichment_volcano_data(res, sig_threshold=DEFAULT_SIG_THRESHOLD)
        effect_lbl = "NES" if vdf["effect_col"].iloc[0] == "nes" else "Enrichment score (ES)"
        n_labels = st.slider("Terms to label", 0, 20, 8, key="gsea_labels")
        _enrichment_volcano(vdf, DEFAULT_SIG_THRESHOLD, effect_lbl, n_labels=n_labels)
    else:
        _enrichment_bar(res.copy(), "nes", "Normalized Enrichment Score (NES)",
                        pos_label="high-score end", neg_label="low-score end")

    disp = res.rename(columns={
        "term_name": "Term", "category": "Category", "set_size": "Set size",
        "es": "ES", "nes": "NES", "p_value": "p", "fdr": "FDR",
    })
    keep = ["Term", "Category", "Set size", "ES", "NES", "p", "FDR"]
    disp = disp[[c for c in keep if c in disp.columns]]
    st.dataframe(
        disp.style.format({c: "{:.4g}" for c in ("ES", "NES", "p", "FDR")}),
        use_container_width=True, hide_index=True,
    )
    st.download_button(
        "⬇  Download fgsea results CSV",
        data=res.to_csv(index=False).encode(),
        file_name=f"fgsea_{label.replace(' ', '_')}.csv",
        mime="text/csv", key="gsea_dl",
    )


def _paxdb_agreement_panel(ds, ds_name: str) -> None:
    """PaxDb abundance-agreement panel: Spearman ρ + scatter vs the reference."""
    from biastracker.analysis.paxdb import paxdb_abundance_agreement

    paxdb = _paxdb_ppm()
    with st.expander("PaxDb abundance agreement (mean LFQ vs reference)", expanded=False):
        if paxdb.empty:
            st.caption("PaxDb reference table not available.")
            return
        st.caption(
            "Spearman rank correlation between this dataset's mean-LFQ abundance and "
            "the PaxDb reference proteome. It checks whether proteins that are abundant "
            "here are generally abundant in the reference (rank agreement, not equality)."
        )
        if not st.button("▶  Compute PaxDb agreement", key=f"paxdb_run_{ds_name}"):
            return
        try:
            res = paxdb_abundance_agreement(ds.table, id_col=ds.id_col, paxdb_ppm=paxdb)
        except ValueError as exc:
            st.warning(str(exc))
            return
        if res.message:
            st.warning(res.message)
        else:
            st.caption(
                f"ρ = {res.rho:.3f}  (p = {res.p_value:.2g})  ·  "
                f"{res.n_used:,} proteins used  ·  matched "
                f"{res.n_matched:,}/{res.n_input:,} ({100 * res.matched_fraction:.0f}%)"
                + (f"  ·  {res.n_excluded:,} pair(s) excluded" if res.n_excluded else "")
            )
        _paxdb_scatter(res, ds_name)


def tab_enrichment(datasets: dict) -> None:
    st.markdown('<div class="bt-section">Functional Enrichment</div>', unsafe_allow_html=True)
    if not datasets:
        st.info("⬅ Upload at least one dataset from the sidebar.")
        return

    st.markdown("**1 · Choose an annotation / gene-set**")
    source = st.radio(
        "Annotation source",
        ["Built-in", "Upload file"],
        horizontal=True,
        help="Built-in: bundled UniProt-keyed sets (Contaminants DB, HPA "
             "subcellular location). Upload: your own GMT or long table.",
    )

    if source == "Built-in":
        choice = st.selectbox("Built-in set", list(_BUILTIN_ANNOTATIONS), key="builtin_ann")
        rel_path, src_tag = _BUILTIN_ANNOTATIONS[choice]
        ann, err = _load_builtin_annotation(rel_path, src_tag)
        if err:
            st.error(f"Could not load '{choice}': {err[:300]}")
            return
    else:
        ac1, ac2 = st.columns([1, 2])
        fmt_label = ac1.radio(
            "Format", ["GMT", "Long table (CSV/TSV)"],
            help="GMT: one gene set per line (term, description, IDs…). "
                 "Long table: rows of ID↔term pairs.",
        )
        fmt = "GMT" if fmt_label == "GMT" else "long"
        ann_file = ac2.file_uploader(
            "Annotation file", type=["gmt", "csv", "tsv", "txt"], key="ann_file",
        )

        id_col = term_col = term_id_col = category_col = ""
        if fmt == "long":
            lc1, lc2, lc3, lc4 = st.columns(4)
            id_col       = lc1.text_input("ID column", value="primary_id")
            term_col     = lc2.text_input("Term-name column", value="term_name")
            term_id_col  = lc3.text_input("Term-ID column (opt.)", value="")
            category_col = lc4.text_input("Category column (opt.)", value="")

        if ann_file is None:
            st.caption("Upload an annotation file to enable ORA and fgsea. IDs must match "
                       "your dataset IDs (e.g. UniProt accessions).")
            return

        ann, err = _load_annotations(
            ann_file.getvalue(), ann_file.name, fmt,
            id_col, term_col, term_id_col, category_col,
        )
        if err:
            st.error(f"Annotation load error: {err[:300]}")
            return
    n_terms = ann.table[ann.term_col].nunique()
    n_ids   = ann.table[ann.id_col].nunique()
    st.success(f"✓ {n_terms:,} terms · {n_ids:,} unique IDs")

    st.markdown("**2 · Choose analysis**")
    method = st.radio(
        "Method", ["ORA (over-representation)", "fgsea (pre-ranked GSEA)"],
        horizontal=True,
        help="ORA: is a term over-represented in one dataset's proteins vs a background "
             "(Fisher's exact). fgsea: is a term concentrated at the top/bottom of a "
             "dataset ranked by a score (pre-ranked GSEA).",
    )
    if method.startswith("ORA"):
        _run_ora_ui(datasets, ann)
    else:
        _run_fgsea_ui(datasets, ann)


# ═══════════════════════════════════════════════════════════════
#  Tab: Export
# ═══════════════════════════════════════════════════════════════

def tab_export(datasets: dict) -> None:
    if not datasets:
        st.info("⬅ Upload at least one dataset from the sidebar.")
        return

    st.markdown('<div class="bt-section">Download Processed Tables</div>', unsafe_allow_html=True)
    for name, ds in datasets.items():
        st.download_button(
            label=f"⬇  {name} — annotated table (.csv)",
            data=ds.table.to_csv(index=False).encode(),
            file_name=f"{name.replace(' ', '_')}_annotated.csv",
            mime="text/csv",
            key=f"dl_{name}",
        )

    if "cmp_res" in st.session_state:
        st.markdown(
            '<div class="bt-section" style="margin-top:1.5rem;">Comparison Results</div>',
            unsafe_allow_html=True,
        )
        lbl = st.session_state.get("cmp_label", "comparison")
        st.download_button(
            label=f"⬇  {lbl} — statistics (.csv)",
            data=st.session_state["cmp_res"].to_csv(index=False).encode(),
            file_name=f"stats_{lbl.replace(' ', '_')}.csv",
            mime="text/csv",
            key="dl_cmp",
        )


# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════

def main() -> None:
    datasets = render_sidebar()

    # ── Header ────────────────────────────────────────────────────────────────
    logo_path = Path(__file__).parent / "assets" / "logo.png"
    if logo_path.exists():
        hc1, hc2 = st.columns([1, 9])
        hc1.image(str(logo_path), width=85)
        with hc2:
            st.markdown(
                f'<h1 style="margin:0;color:{_NAVY};">'
                f'Bias<span style="color:{_CYAN};">Tracker</span></h1>'
                f'<p style="margin:0;color:#666;font-size:0.9rem;">'
                f'Physicochemical bias analysis for proteomics datasets</p>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            f'<h1 style="color:{_NAVY};">'
            f'Bias<span style="color:{_CYAN};">Tracker</span></h1>'
            f'<p style="margin-top:-0.5rem;color:#666;">'
            f'Physicochemical bias analysis for proteomics datasets</p>',
            unsafe_allow_html=True,
        )

    st.markdown("")

    # ── Tabs ──────────────────────────────────────────────────────────────────
    t1, t2, t3, t4, t5 = st.tabs([
        "🏠  Overview",
        "📊  Distributions",
        "⚖️  Compare",
        "🧬  Enrichment",
        "⬇  Export",
    ])
    with t1:
        tab_overview(datasets)
    with t2:
        tab_distributions(datasets)
    with t3:
        tab_compare(datasets)
    with t4:
        tab_enrichment(datasets)
    with t5:
        tab_export(datasets)


main()
