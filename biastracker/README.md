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
