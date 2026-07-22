# BiasTracker

## What is BiasTracker?
BiasTracker is a Python library and CLI tool designed to track, analyze, and visualize biases in proteomics and biological datasets. It provides standardized workflows to summarize datasets, compare distributions, and run enrichment analyses.

## Relationship with protperties
BiasTracker relies heavily on the `protperties` package for standardizing, loading, and modeling proteomics data. BiasTracker depends on `protperties` and exclusively interacts with its public API to ensure robust and reproducible data handling.

## Installation in development mode
To install BiasTracker in development mode, first install `protperties`, then install this package:

```bash
cd BiasTracker
python -m pip install -e ../protperties
python -m pip install -e .
```

## First planned workflow
The first planned workflow is to load a proteomics dataset through `protperties`, then use BiasTracker to run a comprehensive statistical summary and identify potential distribution biases in protein expression across different experimental conditions.

## Quickstart with toy data

To verify the installation works end-to-end without needing real data or an internet connection, run the toy workflow:

```bash
biastracker run examples/ghost_proteome_config.yaml
```

**Expected outputs:**
- `examples/results/tables/` (Contains CSV reports)
- `examples/results/figures/` (Contains PNG plots)

## Annotation Sources

BiasTracker can use several annotation sources for enrichment analysis:

- The PANTHER API is used for GO biological process, GO cellular component, GO molecular function, PANTHER GO-slim, protein class, PANTHER pathway, and Reactome pathway annotations.
- The UniProt API (`annotations.uniprot_go`) fetches GO biological-process / cellular-component / molecular-function terms directly for a set of UniProt accessions and returns them as a long-format annotation set.
- The HPA `subcellular_location.tsv.zip` download is used for subcellular location annotations.
- HPA UMAP and CZ Biohub UMAP annotation workflows are intentionally not implemented yet.
- UniProt ID mapping is used when HPA Ensembl gene IDs need to match `primary_id` values from UniProt-based protein tables.

In the app's **Enrichment** tab, functional terms can be fetched live for the
proteins in your loaded datasets before running ORA/fGSEA: choose **PANTHER
(live API)** (pick GO/pathway/protein-class datasets) or **UniProt GO (live
API)** (pick GO aspects). Both annotate the union of accessions across the
loaded datasets, cache the result, and feed straight into ORA and fGSEA. The
existing **Built-in** (Contaminants DB, HPA) and **Upload file** (GMT / long
table) sources remain available.

**Caching & freshness.** Fetched batches are written to an on-disk cache
(`.cache/biastracker/panther/`, `.cache/biastracker/uniprot_go/`), keyed by the
exact accession batch plus the options (organism/datasets, or GO aspects). Cached
entries have a time-to-live (`DEFAULT_ANNOTATION_TTL_DAYS`, 30 days): older
entries are re-fetched automatically so reference data never goes silently out of
date. Tick **Refresh** in the panel to ignore the cache and re-query now
(`max_age_days=0`); the fresh result overwrites the cache. Loaders also accept
`max_age_days=None` to disable expiry entirely.

BiasTracker protein tables usually use UniProt accessions, especially when they come from `protperties`. HPA subcellular annotations use Ensembl gene IDs in the `Gene` column. When combining HPA annotations with `protperties` protein tables, use `map_to_uniprot: true` so the HPA rows are normalized to UniProt accessions before enrichment.

## PANTHER API Example

Use `type: panther_api` to fetch annotations for IDs already loaded in a dataset. The workflow reads `id_col` from the named dataset, calls PANTHER, caches the response, and exports the normalized annotation table.

```yaml
annotations:
  - name: panther_api
    type: panther_api
    dataset: my_protein_dataset
    id_col: primary_id
    organism: 9606
    categories:
      - go_bp
      - go_cc
      - go_mf
      - protein_class
      - panther_pathway
      - reactome_pathway
```

## HPA Subcellular Example

Use `type: hpa_subcellular` with either a local `path` or the public HPA URL. For UniProt-based protein tables, enable mapping from HPA Ensembl IDs to UniProt.

```yaml
annotations:
  - name: hpa_subcellular
    type: hpa_subcellular
    url: https://www.proteinatlas.org/download/tsv/subcellular_location.tsv.zip
    map_to_uniprot: true
    from_ns: Ensembl
    to_ns: UniProtKB
```

For offline or pinned analyses, download the HPA file once and use `path: data/subcellular_location.tsv.zip` instead of `url`.

## API Annotation Workflow Example

The example config at `examples/api_annotations_config.yaml` is designed to run without internet access. It loads a tiny standard protein CSV, uses a precomputed local PANTHER API cache for `type: panther_api`, loads a tiny local HPA subcellular TSV, maps the HPA Ensembl IDs to UniProt accessions from the toy protein table, and runs enrichment for the `case` group.

```bash
biastracker run examples/api_annotations_config.yaml
```

To run the same pattern against the live PANTHER API, remove the toy `cache_dir` or point it at an empty writable cache directory:

```yaml
annotations:
  - name: panther_api
    type: panther_api
    dataset: my_protein_dataset
    id_col: primary_id
    organism: 9606
    cache_dir: .cache/biastracker/panther
    categories:
      - go_bp
      - go_cc
      - go_mf
      - protein_class
      - panther_pathway
      - reactome_pathway
```

## Reproducibility

API responses and normalized annotations are cached. Workflow runs export normalized annotation tables and metadata into `results/annotations/`, so downstream enrichment can be inspected or reused without depending on a live service response. Network tests are optional; the main toy examples avoid network access by using local data and precomputed cache files.

To run the optional live PANTHER, HPA, and UniProt smoke tests:

```bash
BIASTRACKER_RUN_NETWORK_TESTS=1 pytest tests/test_network_smoke.py
```

## Pre-ranked fGSEA-style Enrichment

BiasTracker also supports pre-ranked gene set enrichment for ranked protein
scores. Use `analysis.fgsea` when every protein has a numeric ranking metric,
such as log fold change, test statistic, abundance shift, or another continuous
bias score.

```yaml
analysis:
  fgsea:
    - dataset: my_protein_dataset
      annotation: panther_api
      score_col: log2_fold_change
      id_col: primary_id
      min_term_size: 10
      max_term_size: 500
      n_permutations: 1000
      weight: 1.0
      seed: 1
```

The workflow writes `results/tables/fgsea__*.csv` with enrichment score (`es`),
normalized enrichment score (`nes`), permutation `p_value`, `fdr`, and
`leading_edge` proteins. It also writes `results/figures/fgsea_dotplot__*.png`
when results are non-empty.

### fGSEA in the app (Enrichment → fgsea)

The app narrows the fGSEA ranking to two scientifically meaningful choices
(`biastracker.analysis.ranking.FGSEA_RANKING_METHODS`):

- **Mean expression** (default) — proteins are ranked by the row-wise mean of the
  dataset's LFQ / expression columns. When per-sample `LFQ intensity …` columns
  are present you may choose which to average; otherwise the precomputed
  `mean_lfq` / `expression` column is used. This is the natural abundance
  ranking for most proteomics datasets.
- **Custom metric** — rank by any numeric column you supply. For a homemade
  dataset the `expression` column is preselected.

Ranking handling: values are numerically coerced; missing/non-numeric/infinite
values are dropped; duplicate accessions collapse to their **maximum**; ties keep
input order (deterministic); the final ranking is sorted descending. Rows flagged
`is_contaminant` are **excluded** from the ranking by default (contaminants are
kept ID-only in the dataset for the contaminant ORA, but should not distort the
abundance ranking).

**PaxDb reference abundance.** PaxDb is treated as **just another per-protein
metric**: on load, protein-level datasets get a `paxdb_log10_ppm` column
(`analysis.paxdb.add_paxdb_abundance`) holding the log₁₀ PaxDb ppm matched by
accession (non-positive / unmatched → NaN, contaminants excluded). It therefore
appears in the **Distributions** and **Compare** tabs with the usual descriptive
statistics, exactly like the physicochemical features — there is no separate
correlation panel.

**Volcano plot.** Enrichment results default to a volcano plot — effect size
(NES, or ES as a fallback) on the x-axis versus `−log10` of the chosen
significance statistic on the y-axis, with a significance line at the threshold
and a vertical line at effect = 0. A **significance criterion** selector offers
*nominal p ≤ 0.05*, *FDR ≤ 0.05*, and *FDR ≤ 0.25* (the conventional GSEA
cutoff); it drives the y-axis, the reference line, the point colouring, and the
"N significant" count. The statistic is floored before the log so a value of 0
stays finite, and both p and FDR are preserved in the hover text. The previous
bar chart remains available via the plot selector.

### Significance on the Distributions tab

With two or more datasets loaded, the **Distributions** tab overlays the
significance of the between-dataset difference for the selected property directly
on the plot (toggle: *Sig*). It reuses the same tests as the Compare tab
(`analysis.compare.feature_significance`): **Mann-Whitney U** for two datasets,
**Kruskal-Wallis** for three or more. The annotation shows the test, nominal
`p`, and — for standard panel features — the panel-corrected `FDR` (matching the
Compare tab) plus a significance marker (`**` < 0.05, `*` < 0.1, else `n.s.`);
two-dataset violin plots also get a significance bracket. Features outside the
standard panel (e.g. `paxdb_log10_ppm`) show the nominal `p` only.
