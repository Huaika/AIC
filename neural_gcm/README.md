# AI in Climate Science — NeuralGCM reproduction

Reproduces the official [NeuralGCM](https://github.com/google-research/neuralgcm)
forecasting quick start from the Nature 2024 paper
*"Neural general circulation models for weather and climate"*
([Nature](https://www.nature.com/articles/s41586-024-07744-y),
[arXiv 2311.07222](https://arxiv.org/abs/2311.07222)).

The notebook initializes the published **deterministic 2.8°** model
(`v1/deterministic_2_8_deg.pkl`) from ERA5, rolls out a 4-day forecast, and
compares NeuralGCM against ERA5 — qualitatively (maps) and with a simple
RMSE-vs-lead-time skill check.

## Setup

NeuralGCM / `dinosaur` / JAX are not reliably tested on the devcontainer's
Python 3.13, so we install into a pinned **Python 3.11** virtualenv (created via
[`uv`](https://github.com/astral-sh/uv)) and register it as a Jupyter kernel.
Run all commands below from this `neural_gcm/` directory:

```bash
cd neural_gcm
bash setup_env.sh
```

This creates `.venv`, installs the dependencies in `requirements.txt`
(including the **CUDA** build of JAX for the devcontainer GPU), and registers a
Jupyter kernel named **"Python (NeuralGCM 3.11)"**.

> If the install fails on a JAX version conflict, `setup_env.sh` automatically
> retries by installing `neuralgcm` first and then pinning `jax[cuda12]` to the
> matching version.

## Run

Open `neuralgcm_forecast_demo.ipynb` and select the
**Python (NeuralGCM 3.11)** kernel, then run all cells. Or headless:

```bash
.venv/bin/jupyter nbconvert --to notebook --execute --inplace \
  neuralgcm_forecast_demo.ipynb
```

The model checkpoint and ERA5 data are read anonymously from public Google
Cloud Storage buckets — no GCP credentials are required, but outbound network
access to `storage.googleapis.com` is.

## Notes

- **GPU**: the devcontainer requests `--gpus device=1`. `jax[cuda12]` ships its
  own CUDA wheels, so no system CUDA toolkit is needed. The first cell prints
  `jax.devices()` — expect a `CudaDevice`.
- **GPU memory**: the 2.8° model needs **~16.7 GiB** for encode/unroll, so it
  only runs on the card when ~17 GiB+ is free. The notebook disables JAX's
  default 75% preallocation (`XLA_PYTHON_CLIENT_PREALLOCATE=false`) so a shared
  card isn't blocked up front. If the GPU is busy or too small, run on CPU:
  ```bash
  JAX_PLATFORMS=cpu .venv/bin/jupyter nbconvert --to notebook --execute --inplace \
    neuralgcm_forecast_demo.ipynb
  ```
  Verified end-to-end on the devcontainer GPU (~16.7 GiB peak) — produces the
  ERA5-vs-NeuralGCM maps and an RMSE-vs-lead-time curve that grows with lead time.
- **Heavier reproductions** (full WeatherBench2 skill scores over the 2020 test
  year, ensemble/stochastic models, or multi-decade climate runs) are out of
  scope for this notebook but use the same model API.
