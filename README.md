# SCOUT

SCOUT is a differentiable causal discovery framework for cyclic structural equation models with soft interventions and unknown intervention targets.

This repository includes code for synthetic data generation, SCOUT model training, and baseline comparisons.

## Requirements

The Python dependencies are listed in `requirements.txt`. Use Python 3.9 or newer with Jupyter installed for the notebooks.

The `backShift` baseline uses R through `rpy2`. If you run that baseline, install R and the required R packages separately.

## Installation

From the repository root, install the Python dependencies with:

```bash
pip install -r requirements.txt
```

If you want an isolated environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For the optional R-backed `backShift` baseline:

```bash
R -e 'install.packages(c("clue", "igraph", "matrixcalc", "reshape2", "ggplot2", "mvnmle", "MASS"))'
```

## Usage

Launch Jupyter from the repository root:

```bash
jupyter notebook
```

The main notebooks are:

- `notebooks/compare_baselines.ipynb`: trains SCOUT and compares against available baselines.
- `notebooks/sergio_test.ipynb`: tests SCOUT on cyclic SERGIO-generated data.
- `notebooks/perturb-cite-seq-soft-intervention.ipynb`: runs soft-intervention experiments on perturb-CITE-seq data.

The SERGIO and perturb-CITE-seq notebooks expect their data/source folders to be available next to this repository directory:

```text
SCOUT/
  SCOUT-master/
  SERGIO-master/
  perturb-cite-seq/
```

## Repository Structure

```text
datagen/              Synthetic cyclic SEM and soft-intervention data generation
models/               SCOUT model, invertible residual block, and flow components
baselines/            Baseline implementations, including NODAGS, LLC, and backShift
notebooks/            Example experiments and baseline comparisons
requirements.txt      Python dependency list
```

## Notes

- SCOUT can train with known or unknown intervention targets.
- Synthetic data generation supports linear, nonlinear, and hybrid SEM settings.
- Large external data folders such as `perturb-cite-seq/` are not committed to this repository.
- Baseline notebooks may require additional non-Python setup, especially for R-based methods.
