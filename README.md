# FlowMamba-Early

Code and result files for the paper *FlowMamba-Early: A shared Mamba-2 model for traffic classification as packets arrive*.

FlowMamba-Early is one byte-level Mamba-2 model that classifies a flow after 1, 2, 3 and 5 packets. Its causal backbone carries its state from one packet to the next, so a later decision does not recompute the earlier packets. This repository contains the model, the controlled comparison with separately trained single-exit models of the same architecture (FlowMamba-Single-N), the checks for plaintext and header shortcuts, the reference models, the inference benchmark, and the scripts that turn the results into the tables and figures of the paper.

## Requirements

- Linux with an NVIDIA GPU and CUDA 12.6. The experiments used one GeForce RTX 4090 with 24 GB.
- Python 3.11 and the packages in `requirements.txt`:

```bash
pip install -r requirements.txt
```

`mamba-ssm` and `causal-conv1d` build CUDA kernels; follow their installation notes if no wheel matches your setup. Redrawing the tables and figures from the files in `results/` needs only numpy, pandas, matplotlib and pillow.

## Data

The experiments use the processed fine-tuning datasets released with NetMamba+ ([repository](https://github.com/wangtz19/NetMambaPlus), [data folder](https://drive.google.com/drive/folders/1_IMfmpB34SM7nUSzzivd_7wAOnt2zwzE)): ISCXVPN2016, CICIoT2022, USTC-TFC2016, CipherSpectrum and CSTNET-TLS1.3. The data are not redistributed here. Put the five dataset folders under `data/`, or point the environment variable `FLOWMAMBA_DATA` to their location:

```
data/
├── iscxvpn/          data-train.json  data-valid.json  data-test.json  metadata.json
├── ciciot2022/
├── ustctfc2016/
├── cipherspectrum/
└── cstnet-tls1.3/
```

`data/SHA256SUMS` lists the checksums of the 20 files used in the paper:

```bash
cd data && sha256sum -c SHA256SUMS
```

Each sample is one unidirectional flow. Each of its first five packets is a 320-byte record with an 80-byte header slot and a 240-byte payload slot.

## Repository layout

| Path | Content |
|---|---|
| `src/` | All code. The docstring at the top of each file says what it does and how to run it. |
| `scripts/` | Shell scripts that loop the training scripts over datasets and seeds. |
| `tests/` | Unit tests of causality, the masked loss, the byte masks and the input parsing (`python -m pytest tests`, GPU required). |
| `results/` | The result files behind the tables and figures of the paper. |
| `data/` | Place for the datasets, with their checksums. |

Everything the scripts write goes to `outputs/`, or to the directory in `FLOWMAMBA_OUTPUT`.

## Main components

| File | Purpose |
|---|---|
| `src/run_flowmamba_early.py` | FlowMamba-Early, the masked multi-exit loss and the common training recipe |
| `src/run_flowmamba_single_n.py` | FlowMamba-Single-N, the single-exit model of the same architecture |
| `src/experiment.py` | Training jobs with the step-matched, long and eligibility-matched budgets and the byte masks |
| `src/masks.py` | Masks for server names, variable header fields and link-layer trailer bytes |
| `src/incremental.py` | Incremental inference: one Mamba-2 state per flow, updated packet by packet |
| `src/analysis.py` | Paired statistics, Holm correction, connection-clustered bootstrap and the other comparisons |

## Reproducing the paper

Run from the repository root.

```bash
# 1. FlowMamba-Early and FlowMamba-Single-N, seeds 20260921-23, imported into the experiment layout
bash scripts/run_first_seeds.sh

# 2. All other runs of the controlled comparison: 535 jobs; several workers can share the job list
python src/plan.py
python src/worker.py w1

# 3. Scores and statistics
python src/test_metadata.py
python src/aggregate.py
python src/analysis.py

# 4. Reference models (three seeds), screened variants and long training (one seed)
bash scripts/run_reference_models.sh
bash scripts/run_exit_variants.sh
bash scripts/run_long_training.sh

# 5. Byte-content audit and descriptive statistics
python src/sni_audit.py
python src/dataset_stats.py
python src/analyze_capture_overlap.py

# 6. Inference cost on CSTNET-TLS1.3; run the timing part on an otherwise idle GPU
python src/run_deployment_benchmark.py --part accuracy
python src/run_deployment_benchmark.py --part timing

# 7. Tables and figures
python src/make_tables.py
python src/make_figures.py
```

The training runs took about 140 hours of logged run time on one RTX 4090, of which the main comparison with ten seeds took about 52 hours.

To redraw the tables and figures from the published result files only:

```bash
FLOWMAMBA_RESULTS=results python src/make_tables.py
FLOWMAMBA_RESULTS=results python src/make_figures.py
```

## Result files

| Paper item | Result file in `results/` | Produced by |
|---|---|---|
| Dataset table, test flows per exit, Fig 3 | `dataset_stats.json`, `per_seed.csv` | `dataset_stats.py`, `aggregate.py` |
| Main comparison, Figs 4 and 5, S1 to S3 Appendix | `analysis/main_*.csv`, `analysis/test_*.csv`, `per_seed.csv` | `analysis.py` |
| Retraining with the combined mask, S6 Appendix | `analysis/masked*.csv`, `analysis/testmask_*.csv`, `sni_audit.json` | `analysis.py`, `sni_audit.py` |
| Decisions across exits | `analysis/stability.csv` | `analysis.py` |
| Per-class F1 and confusion matrices, Figs 6 and 7 | `analysis/per_class.json`, `analysis/confusion.json` | `analysis.py` |
| All models over the first three seeds | `*_summary.csv` | training scripts |
| Inference cost, Fig 8 | `deployment_benchmark.json`, `incremental_agreement.json` | `run_deployment_benchmark.py` |
| Capture-file and connection overlap, S2 Appendix | `capture_overlap.json` | `analyze_capture_overlap.py` |
| Screened variants, S4 Appendix | `exit_variant_scores.csv` | `exit_variant_scores.py` |
| Training trajectories, S5 Appendix | `long_training/*.csv` | `run_long_training.py` |
| Sources of the margin, S7 Appendix | `analysis/per_exit_checkpoint*.csv`, `analysis/eligibility_matched*.csv`, `analysis/long50*.csv` | `analysis.py` |

`per_seed.csv` has one row per run, exit and evaluation set. The variants are `orig` (common recipe), `stepmatch` (step-matched budget), `long50` (50 epochs), `maskall` (retrained with the combined mask), `maskall_stepmatch`, `elig2`, `elig3` and `elig5` (eligibility-matched FlowMamba-Early), and `orig_at_mask_*` (models of the common recipe evaluated with a mask applied to the validation and test inputs only). Per-flow predictions, about 120 MB, are not included.

## Reproducibility

Training is not bitwise reproducible, because the backward pass of the Mamba-2 kernels accumulates gradients with atomic additions. Repeated runs with the same seed start from the same weights and batches and then drift apart: on ISCXVPN2016 the logged scores of the first epoch still agree to four decimals, while on the larger datasets small differences already appear within the first epoch. Inference is deterministic. The files in `results/` come from our runs, and `make_tables.py` and `make_figures.py` reproduce the tables and figures of the paper from them exactly.

## Acknowledgments

MNB follows the design of MambaNetBurst (Kulatilleke et al., arXiv:2605.11034), and NetMamba-N follows NetMamba (Wang et al., ICNP 2024). The datasets come from the NetMamba+ release (Wang et al., arXiv:2601.21792) and remain subject to the terms of their original providers.

## License

The code is released under the MIT License; see `LICENSE`.
