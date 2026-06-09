# Evaluation of Machine Learning Regional Climate Models

This module builds on the set of evaluation indices and diagnostics recommended by [CORDEX-ML-Bench](https://github.com/WCRP-CORDEX/ml-benchmark/tree/main/evaluation) for assessing Machine Learning-based regional climate downscaling models.

## Project Structure

This repository is structured as a standard Python package:

- `ml_evaluation/` - Core package containing the Python source code.
  - `evaluation/` - Evaluation indices and diagnostics.
    - `indices.py` - Core evaluation indices.
    - `diagnostics.py` - Core evaluation performance diagnostics.
    - `indices_plus.py` - Extended indices.
    - `diagnostics_plus.py` - Extended diagnostics.
  - `format_predictions/` - Utilities for formatting and templates for model predictions.
    - `templates/` - NetCDF template files matching CORDEX-ML-Bench expected schemas.

______________________________________________________________________

## Evaluation Framework

### Indices vs. Diagnostics

- **Indices** quantify particular characteristics or properties of a single simulation `x`, for example, the mean, specific quantiles, or extremes. These indices are always calculated for an individual simulation and often form the foundation for various diagnostics.
- **Diagnostics** compare two simulations, `x0` and `x1`, which correspond to the ground-truth (reference) data and the model predictions, respectively. These diagnostics are designed to evaluate different aspects of model performance. For example, to assess how well a model captures warm extremes, one might use the bias of the TXx index. Some diagnostics rely on specific indices, while others, such as RMSE, PSD, or the Wasserstein distance, do not depend on them.

### Indices Table

Initial list of metrics to validate RCM emulators (adapted from Kendon et al., 2025):

| Category | Index | Input Timescale | Variables | Output Timescale | Performance Diagnostic | Scoreboard Summary |
|-------|-------|-------|-------|-------|-------|-------|
| **Basic performance** | Raw output. | sub-daily, daily | All | as input | Mean Bias, RMSE, Corr over time (F) | Spatial mean |
|  |  |  |  |  | ***QQ plots*** |  |
|   | Wet/dry day frequency | daily | precip | Freq per year or season | Mean Bias, RMSE, Corr over time (F) | Spatial mean |
| **Extremes** | Percentiles (e.g. 95th, 99th) | daily, sub-daily | All | Annual/Seasonal | Bias of single fields (F) | Spatial mean |
|   | Max (min) values | daily | All | Max per year or season | Mean Bias, RMSE, Corr over time (F) | Spatial mean |
|   | Frequency of threshold exceedance | daily | All | Freq per year or season | Mean Bias, RMSE, Corr over time (F) | Spatial mean |
|   |  |  |  |  | ***Fractional Skill Score*** |  |
| **Spatial coherence** | Radially averaged power | daily, sub-daily | All |  |  |  |
| **Temporal coherence** | Autocorrelation | daily | All | Annual/Seasonal | Bias of single fields (F) | Spatial mean |
|   | Wet/dry, hot/cold spell length: max | daily | precip, temp | Max per year or season | Mean Bias, RMSE, Corr over time (F) | Spatial mean |
|   | Power spectral density | daily | All | Annual/Seasonal | PSD plots |  |
|   | ***Diurnal cycle (amplitude, phase)*** | sub-daily | precip, temp |  |  |  |
|   | ***Joint distribution characteristics*** | sub-daily | All |  | ***Wasserstein Distance (2D)*** |  |
| **Multivariate coherence** | Multivariate correlation | daily | temp-precip | Annual/Seasonal | Bias of single fields (F) | Spatial mean |
|  | ***Probability of joint exceedances*** | daily | temp-precip | Annual/Seasonal | Bias of single fields (F) | Spatial mean |
|  | ***Joint distribution*** | daily | temp-precip | Annual/Seasonal | ***Wasserstein Distance (2D)*** |  |
| **Climate change** | ***Long-term trends*** | daily | All | Annual/Seasonal | Bias of single fields (F) | Spatial mean |
| **Case studies** | ***Event based (compound extremes)*** | daily | heat+drought |  |  |  |
| **Object-Oriented** | ***Raw output + object algorithm*** | daily | precip, temp |  | ***SAL Score*** |  |
| **Explainability** | ***Integrated gradients*** | daily | All |  |  |  |

______________________________________________________________________

## Setup & Installation

You can set up and run this project using either **Conda** or **Pixi**.

### Option A: Setup with Conda

To set up the development environment using Conda:

1. Create a conda environment from the provided environment file:
   ```bash
   conda env create -f requirements/environment.yaml
   ```
1. Activate the environment:
   ```bash
   conda activate rcm-emulation
   ```
1. Install the package in editable mode:
   ```bash
   pip install -e .
   ```

### Option B: Setup with Pixi

[Pixi](https://pixi.sh) is a modern package management tool that automatically manages dependencies and tasks within project environments.

1. Install Pixi if you haven't already.
1. Initialize and run tasks directly. Pixi will automatically download and configure all dependencies when you execute any command:
   ```bash
   pixi run -e dev type-check
   ```
1. Alternatively, you can start a shell within the environment:
   ```bash
   pixi shell -e dev
   ```

______________________________________________________________________

## Command Line Interface (CLI)

The repository provides a `ml-eval` CLI executable through `click`. It supports two workflows: a fast-track workflow for computationally inexpensive metrics, and a slow-track workflow for heavy metrics.

### 1. Fast-Track Workflow

Computes **Mean Bias**, **RMSE**, and **Pearson Correlation** on input NetCDF datasets.

```bash
ml-eval fast --x0 <ref.nc> --x1 <pred.nc> --var <variable> --output <prefix_or_dir> [--season <season>]
```

### 2. Slow-Track Workflow

Computes **Power Spectral Density (PSD)**, **Radially Averaged Log Spectral Distance (RALSD)**, and **Spatial Wasserstein Distance**.

```bash
ml-eval slow --x0 <ref.nc> --x1 <pred.nc> --var <variable> --output <prefix_or_dir> [--season <season>]
```

### Docker Integration (Kubernetes ready)

You can run the CLI inside a Docker container without local dependencies. The container is configured with the Pixi environment loaded and the CLI set as the entrypoint.

Build the image:

```bash
docker build -t d4p .
```

Run the container:

```bash
docker run --rm -v $(pwd):/data d4p fast --x0 /data/ref.nc --x1 /data/pred.nc --var tasmax --output /data/eval_out
```

### Call the CLI directly as a module

```bash
pixi run -e dev python -m ml_evaluation.cli fast --x0 <ref.nc> --x1 <pred.nc> --var <variable> --output <prefix_or_dir> [--season <season>]
```

```bash
pixi run -e dev python -m ml_evaluation.cli slow --x0 <ref.nc> --x1 <pred.nc> --var <variable> --output <prefix_or_dir> [--season <season>]
```

### Optional Input Parameters for ml-eval

```bash
--season <season> # Option to filter by standard seasons or months
--cal365 <bool> # Option to align data to 365day calendar
```

______________________________________________________________________

## Prediction Formatting & Templates

This package provides templates to prepare model predictions for CORDEX-ML-Benchmark:

- Ready-to-fill templates (NetCDF files matching the benchmark's expected schema) are located in `ml_evaluation/format_predictions/templates/`.
- Use the files in `ml_evaluation/format_predictions/templates/` directly as a starting point. They contain the correct dimensions, coordinates, and a placeholder field for the requested variable. Replace the values with your model predictions, preserving the coordinate and dimension structures.

______________________________________________________________________

## Development Tasks

The project contains a set of pre-configured development tasks defined in `pixi.toml` under the `dev` environment:

- **QA / Linting**: `pixi run -e dev qa`
- **Type Checking**: `pixi run -e dev type-check`
- **Unit Tests**: `pixi run -e dev unit-tests`
- **Update Templates**: `pixi run -e dev template-update`
- **Build Docs**: `pixi run -e dev docs-build`
- **Docker Build/Run**: `pixi run -e dev docker-build` and `pixi run -e dev docker-run`
- **Bump Version**: `pixi run -e dev bumpversion-[major|minor|patch]`

To run all quality assurance tasks at once:

```bash
pixi run -e dev default
```

______________________________________________________________________

## How to Contribute Code

To collaborate on code development, you will need a GitHub account.

### Link Local Computer and GitHub

To Pull or Push between your local repository and GitHub, set up either:

- **SSH Key**: Follow the [GitHub SSH Guide](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/using-ssh-agent-forwarding).
- **Personal Access Token (PAT)**: Follow the [GitHub PAT Guide](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens).

### Clone the Repository

Clone the repository to your local machine:

```bash
git clone https://github.com/c3s-cordex-ml/ml-evaluation.git
cd ml-evaluation
```

Configure your local git user profile:

```bash
git config user.email "your.email@mail.com"
git config user.name "your github username"
```

### How to Add Indices and Performance Diagnostics

You can contribute via issues, pull requests, or direct commits:

#### 1. Raise an Issue on GitHub

- Click on the **Issues** tab.
- Click **New Issue** and describe your request.

#### 2. Submit a Pull Request (PR)

- Create a new feature branch:

  ```bash
  git branch branch_add_sal_score
  git checkout branch_add_sal_score
  ```

- Changes should be quality checked using Development Tasks above, e.g.

  **QA / Linting**: `pixi run -e dev qa`

- Stage, commit, and push your changes:

  ```bash
  git add .
  git commit -m "Add SAL score computation diagnostic"
  git push --set-upstream origin branch_add_sal_score
  ```

- Go to GitHub.com and click **Compare & pull request** to submit the PR.

______________________________________________________________________

## References

Kendon, E. J., Addison, H., Doury, A., Somot, S., Watson, P. A., Booth, B. B., ... & Scullion, C. (2025). Potential for machine learning emulators to augment regional climate simulations in provision of local climate change information. Bulletin of the American Meteorological Society, BAMS-D.
