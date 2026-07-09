# Analysis of Newman's α-Scheme for the Bradley–Terry Model

Code to reproduce the experiments, figures, and tables in **[Paper Title — fill in]** by **[Author(s) — fill in]**.

> 📄 Paper: [arXiv link — fill in once available]

## Overview

This repository studies the convergence behavior of Newman's α-scheme for fitting the Bradley–Terry (BT) model, under both synchronous and asynchronous updates. It contains:

- Synthetic experiments on stochastic block model (SBM) comparison graphs (homogeneous, clustered, near-bipartite regimes)
- Two bipartite structures (a random bipartite graph, and a purely cyclic graph)
- Real-data experiments (vervet monkey dominance hierarchy, ATP tennis, ASSISTments student–problem responses)

All core numerical routines (Jacobian computation, spectral-gap / local-convergence-factor calculation, the synchronous/asynchronous update rules, and MLE fitting) live in a single shared module, `utils.py`, imported by every notebook.

## Repository structure

| File | Description |
|---|---|
| `utils.py` | Shared BT/Newman's-α-scheme machinery: `centering`, `get_jacobian`, `get_original_J`, `get_lcf`, `get_spectral_gap`, `get_spectral_gap_gs`, `get_slope`/`get_slope_average`, `newman_update*`, `newman_fpi`, `sbm_2block`, `get_data`, `get_gap`. See in-file docstrings for details on each function. |
| `convergence_factor_comparison.ipynb` | Convergence factor ρ and predicted ρ̄ as a function of α, swept across the three synthetic SBM regimes (homogeneous / clustered / near-bipartite) and multiple (σ, L) settings. |
| `convergence_history.ipynb` | Iteration-error convergence-history plots (with fitted ρ̄ slope indicators) for the same three SBM regimes. |
| `example_monkey.ipynb` | Vervet monkey dominance-hierarchy experiment (real data). |
| `example_tennis.ipynb` | ATP tennis match-outcome experiment (real data). |
| `example_ASSISTments.ipynb` | ASSISTments student–problem response experiment (real, bipartite data). |
| `example_bipartite.ipynb` | Hand-constructed bipartite non-convergence example. |
| `example_cyclic.ipynb` | Hand-constructed purely-cyclic ("backward cycle") non-convergence example. |
| `data` | Real-world datasets used in the simulation |

## Requirements / Installation

```bash
pip install -r requirements.txt
```

**LaTeX is required for figure rendering.** Every plotting cell sets `rc('text', usetex=True)`, which shells out to a real LaTeX installation (not matplotlib's built-in "mathtext"). You need a **full** TeX distribution — a minimal/scheme-basic install is not sufficient, since it will fail with errors like `LaTeX Error: File 'type1ec.sty' not found`. On Ubuntu/Debian:

```bash
sudo apt-get install texlive-full
```

(A lighter alternative is `texlive-latex-extra` + `texlive-fonts-recommended` + `cm-super`, but `texlive-full` is the safest default.) Some plots additionally require the LaTeX `bm` package (loaded automatically via a `rc('text.latex', preamble=...)` call in the relevant cells) — this is included in any standard TeX Live install.

## Data

Raw data files are **not committed to this repository**, both to respect the original datasets' redistribution terms and to keep the repo lightweight. Each notebook expects its input file(s) in `data/`:

| Notebook | Expected file | Source |
|---|---|---|
| `example_monkey.ipynb` | `data/dominance.data.csv` | Vilette et al. (2020) vervet monkey dominance dataset — see `rankReliability` GitHub repo |
| `example_tennis.ipynb` | `tennis.csv` | Sackmann-derived ATP match data (cleaned version from Dong (2025), *Statistical ranking with dynamic covariates*) |
| `example_ASSISTments.ipynb` | `skill_builder_data.csv` | ASSISTments 2009–2010 skill-builder dataset (corrected version) — [sites.google.com/site/assistmentsdata](https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data/skill-builder-data-2009-2010) |

Download each file from its source above and place it at the path the corresponding notebook expects (or edit the `pd.read_csv(...)` path at the top of the notebook). `example_bipartite.ipynb` and `example_cyclic.ipynb` use only synthetically generated data and require no external files.

## Reproducing the paper's figures and tables

| Paper reference | Notebook | Output file(s) |
|---|---|---|
| Fig. 2 — convergence factor vs. α (SBM regimes) | `convergence_factor_comparison.ipynb` | `{balanced,cluster,bipartite}_{sigma}_{L}.pdf`, `.csv` |
| Fig. 3 — convergence history (SBM regimes) | `convergence_history.ipynb` | `history_{balanced,cluster,bipartite}_{sigma}.pdf` |
| Fig. 4 (a) and (b) — random bipartite example | `example_bipartite.ipynb` | `nonconvergence.pdf`, `nonconvergence_hist.pdf` |
| Fig. 4 (c) — cyclic example | `example_cyclic.ipynb` | `backward_cycle.pdf` |
| Fig. 5 (left panel) — Vervet monkey experiment | `example_monkey.ipynb` | `vervet_rates.pdf`, `vervet.pdf` |
| Fig. 5 (middle panel) — ATP tennis experiment | `example_tennis.ipynb` | `atp_rates.pdf`, `atp.pdf` |
| Fig. 5 (right panel) — ASSISTments experiment | `example_ASSISTments.ipynb` | `math_rates.pdf`, `math.pdf` |


To run a notebook, open it in Jupyter and select **Kernel → Restart & Run All** — every notebook has been verified to run top-to-bottom from a clean kernel with only `utils.py` (and, where applicable, its input data file) present alongside it.

## Notes on reproducibility

- Random seeds are fixed throughout (`np.random.seed`, `np.random.default_rng`) for exact reproducibility of the synthetic experiments.
- `get_lcf` computes the empirical local convergence factor ρ by removing exactly the eigenvalue nearest to 1 (the trivial eigenvalue) from the Jacobian spectrum, rather than using a fixed numerical tolerance — this is important for correctness on real (non-synthetic) data, where the trivial eigenvalue can retain more floating-point drift from exactly 1.0 than a naive tolerance check would allow for.
- `get_slope`/`get_slope_average` return `NaN` (rather than raising an error) for any `(alpha, sync)` combination where the empirical iteration does not converge within the given `maxiter` — this is itself a meaningful result for the non-convergence examples in the paper.

## Citation

If you use this code, please cite:

```bibtex
@article{[citekey],
  title   = {[Paper Title]},
  author  = {[Author(s)]},
  journal = {Journal of Machine Learning Research},
  year    = {2026},
}
```

## License

Code is released under the [MIT License](LICENSE). See [Data](#data) above regarding the licensing terms of the third-party datasets used in this work — those are governed by their original sources, not this repository's license.
