# Evaluating machine-learning-based climate downscaling models

This module packages a set of evaluation indices, diagnostics and plotting tools for assessing Machine Learning-based regional climate downscaling models. It expands on the [CORDEX-ML-Bench](https://github.com/WCRP-CORDEX/ml-benchmark/tree/main/evaluation) validation functions and could be used for further CORDEX benchmarking activities.


## Project Structure

This repository is structured as a standard Python package with self-contained environments managed by Pixi or Conda:

- `ml_evaluation/` - Core package containing the Python source code:
  - `chunking.py` - Optimisation of chunks if Dask parallel processing approach is used.
  - `cli.py` - Command-line interface definitions and logic using `click`.
  - `io.py` - Dataset loading and Zarr-writing utilities, plus default Dask cluster parameters.
  - `transformations.py` - Dimension alignment and threshold/unit conversion helpers shared across CLI workflows.
  - `evaluation/` - Core and extended evaluation indices and diagnostics.
    - `indices.py` - Standard evaluation indices (e.g., mean, wet/dry day frequency).
    - `diagnostics.py` - Standard performance diagnostics comparing predictions and reference data (e.g., bias, RMSE, correlation).
- `environment.yaml` - Legacy Conda environment description.
- `tests/` - Unit tests for testing the CLI, diagnostics, and indices.
- `Dockerfile` - Docker configuration to containerize the evaluation environment and CLI.
- `pixi.toml` - Project configuration file for Pixi, specifying Conda and PyPI dependencies, build platforms, and developer tasks.
- `pixi.lock` - Lockfile storing the exact versions and hashes of resolved dependencies, ensuring a reproducible environment.
- `pyproject.toml` - Standard Python packaging metadata, defining dependencies, metadata, and the `ml-eval` CLI entry point.
- `.pre-commit-config.yaml` - Configuration for QA/linting hooks executed before commits.

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
   conda env create -f environment.yaml
   ```
1. Activate the environment:
   ```bash
   conda activate ml-evaluation
   ```
1. Install the package in editable mode:
   ```bash
   pip install -e .
   ```

### Option B: Setup with Pixi

[Pixi](https://pixi.sh) is a modern, fast package management tool built on top of the Conda ecosystem. It automatically manages dependencies (both Conda-forge and PyPI packages), system environments, and development tasks in a reproducible and self-contained manner.

#### 1. Installing Pixi

If you do not have Pixi installed, you can install it using one of the following methods:

**macOS / Linux:**

```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

After installation, restart your terminal or source your shell configuration file (e.g., `source ~/.bashrc` or `source ~/.zshrc`) to make the `pixi` command available in your PATH.

#### 2. Installing Project Dependencies (`pixi install`)

To install the project dependencies, run the following command in the root of the repository:

```bash
pixi install
```

**What does `pixi install` do?**

- It reads the dependency specifications in `pixi.toml`.
- It resolves and downloads all required packages (both Conda and PyPI) for your platform.
- It creates a local, self-contained environment under the `.pixi/` directory in the root of the repository.
- Unlike global or user-wide conda installations, the environment is isolated to this directory, preventing conflicts with other projects.

#### 3. Understanding the Lockfile (`pixi.lock`)

The `pixi.lock` file is an automatically managed lockfile that records the exact versions, build configurations, and cryptographic hashes of all resolved packages (including transitive dependencies) for all supported platforms.

**Why is it important?**

- **Reproducibility:** It ensures that every developer, CI/CD runner, or Docker container running `pixi install` will build the exact same environment down to the binary level.
- **Consistency:** It eliminates the "works on my machine" class of bugs caused by silent package updates.
- **Speed:** Since dependencies are already resolved in the lockfile, installation is extremely fast.

*Note: You should always commit changes to `pixi.lock` to version control when modifying dependencies in `pixi.toml`.*

#### 4. Running Tasks and Entering the Environment

Pixi allows you to run commands and development tasks directly within the environment without manually activating it, or you can enter an interactive shell.

**Running tasks directly:**

```bash
# Run type checking using the dev environment
pixi run -e dev type-check

# Run unit tests
pixi run -e dev unit-tests
```

**Entering the environment shell:**

```bash
# Start a shell with the development environment activated
pixi shell -e dev
```

______________________________________________________________________

## Command Line Interface (CLI)

The repository provides a `ml-eval` CLI executable through `click`. It supports several workflows that partition the diagnostics for ease of computation as summarised below. The diagnostics are applied to input NetCDF datasets (ground truth and prediction) that are assumed to contain a number of different variables. Output is stored as zarr files.

### 1. Fast-Track Workflow

Computes **RMSE**, **RALSD** and also **Bias** for various indices, comparing predictions `x1` against reference `x0`.

General: mean, P01, P99, lag-1 day auto-correlation, and average annual extrema.

Precipitation specific: SDII, P98wet and R01 (average annual no. wet days wrt 1mm).

```bash
ml-eval fast --x0 <ref.nc> --x1 <pred.nc> --var <variable> --output <prefix_or_dir> [--season <season>]
```

### 2. Climate-Change-Signal Workflow (`signal`)

Computes the change of each climatological index (mean/P01/P99 for non-precip variables; R01/SDII/RX1day for precip) between a historical and future period, for a single source (RCM-truth or a model). This workflow is source-agnostic and does not require ground truth - the signal *bias* between two sources (e.g. model vs. RCM-truth) is obtained downstream by comparing two `signal` output bundles.

```bash
ml-eval signal --hist <hist.nc> --fut <fut.nc> --var <variable> --output <prefix_or_dir> [--season <season>]
```

### 3. Slow-Track Workflow

Computes **distribution scores**, **Bias** for various additional indices (including variable specific), and **correlation** metrics for `x1` with respect to `x0`. This is the heavier counterpart to `fast`.

```bash
ml-eval slow --x0 <ref.nc> --x1 <pred.nc> --var <variable> --output <prefix_or_dir> [--season <season>]

```

### 4. Multi-Variate Workflow (`multivar`)

Computes **correlation** (Pearson and Spearman's rank) between two variables for the reference and prediction datasets, along with their biases.

```bash
ml-eval multivar --x0 <ref.nc> --x1 <pred.nc> --var <variable> --var2 <second_variable> --output <prefix_or_dir> [--season <season>]
```

### 5. Probabilistic Workflow (`prob`)

Computes **CRPS**, **Brier Score**, and the **spread-error ratio** between `x0` and `x1`, evaluated across the full ensemble (no `--member-num` selection).

```bash
ml-eval prob --x0 <ref.nc> --x1 <pred.nc> --var <variable> --output <prefix_or_dir> [--season <season>]
```

### Docker Integration (Kubernetes ready)

You can run the CLI inside a Docker container without local dependencies. The container is configured with the Pixi environment loaded and the CLI set as the entrypoint.

Build the image:

```bash
docker build -t ml-evaluation .
```

Run the container:

```bash
docker run --rm -v $(pwd):/data ml-evaluation fast --x0 /data/ref.nc --x1 /data/pred.nc --var tasmax --output /data/eval_out
```

### Call the CLI directly as a module

```bash
pixi run -e dev python -m ml_evaluation.cli fast --x0 <ref.nc> --x1 <pred.nc> --var <variable> --output <prefix_or_dir> [--season <season>]
```

```bash
pixi run -e dev python -m ml_evaluation.cli slow --x0 <ref.nc> --x1 <pred.nc> --var <variable> --output <prefix_or_dir> [--season <season>]
```

### Optional Input Parameters for ml-eval

These optional flags are shared across most `ml-eval` subcommands (see `--help` on each command for exact availability):

```bash
--season <season>          # Filter by standard season (DJF, MAM, JJA, SON) or a single month (e.g. JAN)
--align-leap-days <bool>   # Align datasets to a standard calendar by filling missing leap days
--member-num <int>         # Evaluate a single ensemble member (>=0, selected by position) or the
                            #   ensemble mean (-1 or omitted). Ignored with a warning if no member
                            #   dimension is present. Not available on `prob`.
--force-load <bool>        # Load data fully into memory up front (useful for quick tests on small data)
--use-dask <bool>          # Use a Dask distributed client for parallel processing
```

______________________________________________________________________

## Visualisation Tools

Visualisation tools are being developed (ml_evaluation/viz/) to compare ML model performance using scorecards (hexagon plots and heat maps) and extended analysis figures (such as spatial maps of indices and the ML model bias). These use the output Zarr files from CLI such that the computation part of the assessment need only happen once.

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
