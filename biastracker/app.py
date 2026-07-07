"""BiasTracker — Streamlit GUI

Run from the biastracker/ directory:
    streamlit run app.py
"""
from __future__ import annotations

import tempfile
from pathlib import Path

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
        return ds, None
    except Exception as exc:
        return None, str(exc)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


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
                type_label = st.selectbox("Type", list(_DS_TYPES), key=f"ds_type_{slot}")
                ds_type = _DS_TYPES[type_label]

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

def tab_overview(datasets: dict) -> None:
    if not datasets:
        st.info("⬅ Upload at least one dataset from the sidebar to get started.")
        return

    from biastracker.analysis.summary import summarize_dataset

    for name, ds in datasets.items():
        st.markdown(f'<div class="bt-section">{name}</div>', unsafe_allow_html=True)

        n_rows   = len(ds.table)
        n_unique = ds.table[ds.id_col].nunique() if ds.id_col in ds.table.columns else "—"
        n_seq    = int(
            (ds.table["sequence"].notna() &
             (ds.table["sequence"].astype(str).str.strip() != "")).sum()
        )
        n_feats  = len(_available_features(ds))

        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(_card_html(f"{n_rows:,}", f"Rows ({ds.level})"), unsafe_allow_html=True)
        c2.markdown(
            _card_html(
                f"{n_unique:,}" if isinstance(n_unique, int) else str(n_unique),
                "Unique IDs",
            ),
            unsafe_allow_html=True,
        )
        c3.markdown(_card_html(f"{n_seq:,}", "With Sequence"), unsafe_allow_html=True)
        c4.markdown(_card_html(str(n_feats), "Features Available"), unsafe_allow_html=True)

        with st.expander("Feature summary table", expanded=False):
            feats = _available_features(ds)
            st.dataframe(
                summarize_dataset(ds, features=feats),
                use_container_width=True,
                hide_index=True,
            )

        st.markdown("")


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
    rug    = c3.checkbox("Rug", value=False)
    logx   = c4.checkbox("Log₁₀", value=False, help="Log₁₀-transform values (drops ≤ 0)")

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
        color = _COLORS[i % len(_COLORS)]
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
    if logx and n_dropped:
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
    t1, t2, t3, t4 = st.tabs([
        "🏠  Overview",
        "📊  Distributions",
        "⚖️  Compare",
        "⬇  Export",
    ])
    with t1:
        tab_overview(datasets)
    with t2:
        tab_distributions(datasets)
    with t3:
        tab_compare(datasets)
    with t4:
        tab_export(datasets)


main()
