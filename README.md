<h1 align="center"><b>Causal image embedding</b><br>Embeddings and ATE estimation with image covariates</h1>

<p align="center">
  <img src="https://img.shields.io/badge/python-%3E%3D3.12-blue" alt="Python" />
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-526EAF.svg?logo=opensourceinitiative&logoColor=white" alt="License: MIT" /></a>
  <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff" /></a>
  <a href="https://drive.google.com/file/d/169Q7OtMaH4HkOnYp7AAJrf0cbBm0q9Td"><img src="https://img.shields.io/badge/paper-Debiased_Image_Embedding-4285F4?logo=googledrive&logoColor=white" alt="Paper (PDF on Google Drive)" /></a>
  <a href="https://drive.google.com/file/d/1ZecUEefvwy3h-oVJgvl-_YHJxySl0OeH"><img src="https://img.shields.io/badge/slides-Final_Project-34A853?logo=googledrive&logoColor=white" alt="Final project slides (Google Drive)" /></a>
</p>

This repository implements **image embeddings for causal inference** when a single image encodes both information that estimators *should* condition on (image-side covariates) and information they *should not* (post-treatment variation). We use a **debiased architecture** with separate encoders for those two roles, joint reconstruction of the image together with treatment and outcome, and compare **biased**, **naive**, and **debiased** pipelines with regression, IPW, and doubly robust ATE estimators. On **semi-synthetic Fashion-MNIST**, the proposed embedding improves causal effect estimation relative to standard estimators and naive embedding baselines.

## Documentation

| Resource | Description |
|----------|-------------|
| [Paper (PDF)](https://drive.google.com/file/d/169Q7OtMaH4HkOnYp7AAJrf0cbBm0q9Td) | *Debiased Image Embedding* — Google Drive |
| [Final project slides](https://drive.google.com/file/d/1ZecUEefvwy3h-oVJgvl-_YHJxySl0OeH) | Presentation — Google Drive |
| [`conf/`](conf/) | Hydra defaults; `experiment=full` or `experiment=fast` presets |
| [`src/main_experiment.py`](src/main_experiment.py) | Hydra entry: end-to-end experiment |
| [`src/main_analysis.py`](src/main_analysis.py) | Hydra entry: summarize `df_result.pkl` |
| [`tests/`](tests/) | Pytest suite (ATE helpers, autoencoder shapes) |

## Installation

```bash
git clone https://github.com/tatsu432/causal-image-embedding.git
cd causal-image-embedding
curl -LsSf https://astral.sh/uv/install.sh | sh   # optional; or use pip
uv sync
```

For linting, typing, and tests:

```bash
uv sync --extra dev
```

## Run the experiment

Run from the **repository root** so Hydra finds [`conf/`](conf/) and relative paths (`./data`, artifact files) resolve as expected.

**Full settings** (legacy scale, default):

```bash
uv run python src/main_experiment.py
```

**Fast / sanity preset** (small data, few epochs, no plots):

```bash
uv run python src/main_experiment.py experiment=fast
```

Override any key, for example:

```bash
uv run python src/main_experiment.py experiment=full paths.result_pickle=outputs/run.pkl
```

Artifacts default to the repo root: `fashion_mnist_embedding.pt`, `df_result.pkl` (see `paths` in [`conf/config.yaml`](conf/config.yaml)).

After a run, summarize results (pickle path is configurable):

```bash
uv run python src/main_analysis.py
uv run python src/main_analysis.py paths.result_pickle=outputs/run.pkl
```

Download **Fashion-MNIST** under `./data` (or change `paths.data_root`) if it is not already present.

## Dataset (short)

We use a modified [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist): an **icon overlay** on each image encodes post-treatment information (icon type, transparency, position, size). Synthetic **covariates**, **treatment**, **post-treatment** factors, **images** \(V_i\), and **outcomes** \(Y_i\) are simulated as described in the original project notes.

## Docker

```bash
docker build -t causal-image-embedding:local .
docker run --rm causal-image-embedding:local
```

The default image command runs **`pytest`**. Override the command to run an experiment from `/app` (repo root in the image), for example:

```bash
docker run --rm causal-image-embedding:local uv run python src/main_experiment.py experiment=fast
```

First builds can take a while while **PyTorch** and **TensorFlow** wheels download.

## Tests & CI

From the repo root:

```bash
uv sync --extra dev
uv run ruff check src tests
uv run ruff format --check src tests
uv run mypy src
uv run pytest tests/ -q
```

Pushes and pull requests to `main` / `master` run the same checks via GitHub Actions.
