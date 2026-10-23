# CertGate — certified stability of gating-based explanations

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23588601.svg)](https://doi.org/10.5281/zenodo.23588601)

Supplementary code and results for "Certified Stability of Gating-Based Explanations: Exact Certificates and Fundamental Limits for Takagi–Sugeno–Kang and Mixture-of-Experts Models".

## Package (`certgate/`)
- `models.py` — gated models: Gaussian TSK (product t-norm), Gaussian TSK (minimum t-norm), linear softmax mixture of experts.
- `certify.py` — exact routing certificate (Theorem 1 with Corollaries 3, 4, 6 / Theorem 5), interval and first-order Lipschitz baselines, certified radius by bisection.
- `train.py` — training with load balancing and the scale-invariant certified-stability penalty.
- `attack.py` — projected-gradient flip search and uniform-sampling stability check.
- `witness.py` — independent construction of counterexamples (constructive side of Theorem 1).
- `data.py` — dataset loading, SHA-256 fingerprints, stratified CV, min–max scaling.

## Experiments (script → output in `results/`)
| Paper section | Script | Output |
|---|---|---|
| Main study (3 seeds × 5 folds × 18 datasets) | `run_full.py`, `analyze_full.py` | `results/full/`, `full_stats.txt`, `per_family_tests.csv` |
| Pilot (PGD attacks, witnesses, R sweep) | `run_pilot.py`, `analyze.py`, `verify_witness.py` | `pilot_*.csv`, `pilot_stats.txt` |
| Independent global-optimiser check | `verify_globalopt.py` | `globalopt_check.csv`, `globalopt.log` |
| Sampling budget, top-2, weight bounds | `run_extra.py` | `extra_*.csv` |
| Equal-budget tuning and fixed R = 8 | `run_tuned.py`, `analyze_tuned.py` | `results/tuned/`, `tuned_stats.txt` |
| Ablations | `run_ablation.py` | `ablation.csv` |
| Theorem 12 synthetic experiment | `run_synth.py` | `synth_thm7.csv` |
| Linguistic case study | `casestudy.py` | `casestudy.json`, `casestudy.txt` |
| Numerical checks of the theory | `theory_checks/*.py` | console output |

## Data
Sixteen UCI datasets read from the public mirror github.com/jbrownlee/Datasets (commit d20fcb6402ae34e653d4513b00f39257bb37ed7f) and two datasets bundled with scikit-learn (WDBC, digits). Row counts match the UCI repository; SHA-256 hashes are in `results/datasets_meta.json`. Please cite the original UCI sources.

## Reproduce
```
pip install -r requirements.txt
git clone https://github.com/jbrownlee/Datasets /path/to/jb_datasets
export CERTGATE_DATA=/path/to/jb_datasets
python run_full.py 0 1 && python analyze_full.py
```
CPU only, float64 (JAX x64). The full study takes roughly 10 CPU-hours; scripts write one file per dataset/seed (or fold) and resume after interruption.

## How to cite
Please cite the paper and the archived software: R. Imamguluyev, "CertGate: Certified stability of gating-based explanations (code and results)," Zenodo, v1.0.0, 2026, doi:10.5281/zenodo.23588601.

## License
MIT (see `LICENSE`).
