"""Locations of the input data and of all outputs.

DATA_ROOT holds the processed NetMamba+ release, one directory per dataset with
data-train.json, data-valid.json, data-test.json and metadata.json.

OUTPUT_ROOT receives everything the scripts write:
    runs/<dataset>/<run_id>/           runs of the standalone training scripts
    experiments/                       runs and per-flow predictions of the job queue (experiment.py)
    results/                           summary CSVs, per-seed scores, statistics and benchmark files
    tables/, figures/, figures_tiff/   LaTeX table rows and figures of the paper

RESULTS_IN is the directory read by make_tables.py and make_figures.py. It defaults
to OUTPUT_ROOT/results; point it to the results/ folder of this repository to
redraw the tables and figures from the published result files.

Each location can be overridden with an environment variable:
FLOWMAMBA_DATA, FLOWMAMBA_OUTPUT, FLOWMAMBA_RESULTS.
"""
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.environ.get("FLOWMAMBA_DATA", os.path.join(REPO, "data"))
OUTPUT_ROOT = os.environ.get("FLOWMAMBA_OUTPUT", os.path.join(REPO, "outputs"))
RESULTS_OUT = os.path.join(OUTPUT_ROOT, "results")
RESULTS_IN = os.environ.get("FLOWMAMBA_RESULTS", RESULTS_OUT)
