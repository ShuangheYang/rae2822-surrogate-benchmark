# RAE2822 Surrogate Benchmark

**Reproducible MLP and Kriging experiments for aerodynamic prediction from 20 CST coefficients.**

![RAE2822 design space](docs/assets/airfoil_design_space.png)

This project studies how predictive accuracy changes with training-set size in a transonic airfoil design space. It includes MLP architecture screening, exact isotropic Matérn-5/2 Kriging, paired learning curves, test-error diagnostics, and model inference. Both models predict lift and drag coefficients; lift-to-drag ratio is computed from the two outputs.

| Resource | Location |
|---|---|
| Public data, methodology and Dataset Viewer | [Hugging Face dataset](https://huggingface.co/datasets/ShuangheYang/rae2822-cst20-cfl3d-fine) |
| Trained MLP models | [MLP repository](https://huggingface.co/ShuangheYang/rae2822-cst20-mlp-r05) |
| Trained Kriging models | [Kriging repository](https://huggingface.co/ShuangheYang/rae2822-cst20-kriging-r05) |
| Illustrated research report | [PDF](https://huggingface.co/datasets/ShuangheYang/rae2822-cst20-cfl3d-fine/resolve/61bcbdd80989bca9260acba5c99b8fd0293200ee/research/report/R05_MLP_Kriging_report.pdf) |
| Interactive explorer source | [docs/index.html](docs/index.html) |
| Methods and interpretation | [METHODS.md](METHODS.md) |

The benchmark uses the **fine** labels. The dataset also provides paired coarse results for multi-fidelity research. Neither the CFD-generation platform nor solver executables are included. Training and inference require no CFD installation.

## Install

Python 3.10 or later is required. A separate environment keeps the benchmark independent of CFD software:

```bash
conda env create -f environment.yml
conda activate airfoil_surrogate
```

Alternatively, in an existing Python environment:

```bash
python -m pip install -r requirements.txt
```

CPU is supported. For GPU execution, install a PyTorch build compatible with your CUDA driver using [the PyTorch installer](https://pytorch.org/get-started/locally/). Do not copy CUDA package versions from another machine without checking compatibility. Exact Kriging requires dense float64 linear algebra and several GiB of memory at 10,000 points; CPU operation can be slow.

## Predict with a published model

Run commands from the repository root:

```bash
python run.py models
python run.py --device cpu predict --family mlp --input configs/example_input.json
python run.py --device cpu predict --family kriging --input configs/example_input.json
```

`models` downloads only the two representative checkpoints. Downloads are anonymous, pinned to immutable Hugging Face commits and checked against SHA-256 hashes. No account, token or paid service is needed.

Input JSON accepts a single array of 20 coefficients, a batch of such arrays, or `{"cst": [...]}`. The order is **U0–U9, L0–L9**. Output columns are `CL`, `CD`, `L_D`. Coefficients outside the published design box trigger an extrapolation warning. Arbitrary coordinates must first be normalized and fitted to the same CST definition; the predictor does not interpret images or coordinate arrays as CST coefficients.

To use a newly trained model, pass its NPZ path with `--model`. Model files use pickle-free NumPy serialization.

## Inspect the published study without training

```bash
python run.py --run-dir runs/reference reference
python run.py --run-dir runs/reference figures
python run.py --run-dir runs/reference export
```

This downloads the frozen input arrays, predictions, metrics, screening records and the original experiment workbook. It regenerates the scientific figures and exports `runs/reference/results/benchmark_results.xlsx`. The original 17-sheet workbook remains available separately. `REFERENCE_ONLY` prevents training over these published records.

Optional numerical verification of both representative models on all 1,500 test cases:

```bash
python run.py --run-dir runs/reference models
python run.py --run-dir runs/reference --device cpu verify
```

## Reproduce training in a fresh directory

These commands **start real training**. They are separate from the commands above.

```bash
python run.py --run-dir runs/reproduction prepare
python run.py --run-dir runs/reproduction plan
python run.py --run-dir runs/reproduction --device auto mlp-select
python run.py --run-dir runs/reproduction --device auto mlp-curves
python run.py --run-dir runs/reproduction --device auto kriging
python run.py --run-dir runs/reproduction --device auto evaluate
python run.py --run-dir runs/reproduction figures
python run.py --run-dir runs/reproduction export
```

1. `prepare` verifies the frozen 10,000 training, 500 validation and 1,500 test records. Training and validation share no identical CST vectors with the test partition. IDs are local to each LHS pool; use the pool prefix to identify a geometry.
2. `plan` shows the Kriging run schedule without training.
3. `mlp-select` screens 12 configurations with seed 11, confirms the best three with seeds 22 and 33, and selects by validation score. It does not read final test labels.
4. `mlp-curves` trains the selected configuration at eight budgets for all three seeds.
5. `kriging` fits exact isotropic models on the same training subsets. Hyperparameters use training marginal likelihood, with recorded optimizer diagnostics and one predefined likelihood-driven bound expansion. No sparse approximation is substituted.
6. `evaluate` freezes and verifies the trained model list before reading test labels. It records metrics, paired bootstrap intervals, error associations, sensitivity and warm inference timing.
7. `figures` and `export` use those evaluation records; they do not refit models.

All outputs stay inside `--run-dir`. Completed fits are checked and skipped on restart. Interrupted fits restart from their beginning; there is no epoch-level checkpoint resume. Changed inputs, code or configurations require a new run directory. Use `Ctrl+C` to interrupt; for Kriging, a file named `STOP_KRIGING` or `STOP_RESEARCH` inside the run directory also requests a stop. Remove it before resuming.

For a partial Kriging run, append `--max-budget 500 --seeds 11`; completion of the full frozen schedule is required for the unified evaluation. Hyperparameter experiments should use separate run directories and remain validation-driven.

## Numerical checks

```bash
python run.py --device cpu smoke
python src/test_kriging_engine.py
```

These checks use small synthetic arrays, not CFD training data. They cover CPU optimization, scaling, serialization, resume guards, GP mean/variance and marginal likelihood against scikit-learn, analytic kernel gradients, and CPU/GPU agreement when CUDA is available. GitHub Actions runs the CPU checks.

See [VALIDATION.md](VALIDATION.md) for the release checks and their limits.

## Interactive presentation

`docs/` is a self-contained static explorer with an airfoil selector, saved per-case predictions, and switchable learning curves. The interface displays **saved test predictions**, not new server-side inference. It uses no analytics or external JavaScript libraries.

Preview locally:

```bash
python -m http.server 8000 --directory docs
```

Open `http://localhost:8000`. On a public GitHub repository, select **Settings → Pages → Deploy from a branch → main → /docs**. GitHub Pages hosts this static site on the free plan. Python training and inference are run locally; they are not executed by Pages. See [PUBLISHING.md](PUBLISHING.md).

## Repository map

| File or directory | Purpose |
|---|---|
| `run.py` | Unified command entry point; experiment directory and device selection |
| `configs/protocol.json` | Twelve MLP configurations, splits, budgets and selection rules |
| `configs/assets.lock.json` | Public artifact URLs, immutable revisions and hashes |
| `configs/cst_bounds.json` | Feature order, baseline coefficients and input bounds |
| `download_assets.py` | Verified public downloads and split validation |
| `src/mlp_engine.py` | MLP, AdamW/L-BFGS training, scaling and NPZ serialization |
| `src/kriging_engine.py` | Exact isotropic GP fitting, uncertainty and serialization |
| `run_mlp_study.py` | Screening, confirmation, learning curves and resumable records |
| `run_kriging_study.py` | Likelihood fitting, warm starts, bounds and diagnostics |
| `evaluate_study.py` | Frozen test evaluation, metrics, bootstrap and timing |
| `make_figures.py`, `plot_relative_error_learning.py` | Scientific plots, including separate CL/CD relative errors |
| `predict.py`, `verify_pretrained.py` | Inference and released-prediction verification |
| `export_results.py` | Excel metrics, learning curves and case-level predictions |
| `build_explorer.py` | Generate the browser data payload from frozen evidence |
| `src/test_kriging_engine.py`, `tests/smoke.py` | Numerical and portability checks |
| `docs/` | Static interactive explorer and figures |
| `runs/` | Local downloaded/generated files; excluded from Git |

## Evidence and scope

The published experiment shows lower MLP error at 5,000 and 10,000 training samples, while this Kriging pipeline performs better at the smaller measured budgets. The mean curves change order between 2,000 and 5,000 samples. That interval is an observation for this design domain and protocol, not a universal crossover threshold or a comparison with every form of Kriging.

All budgets use 500 extra validation labels. MLP configurations were selected at the full budget, not independently optimized at every smaller budget. The independent LHS evaluation pool was used in earlier research; it is not a historically blind test. Predictions approximate the recorded CFL3D solutions at the fixed flow condition; they are not wind-tunnel measurements or evidence of out-of-domain accuracy.

## License and citation

Repository code is licensed under [MIT](LICENSE). Data, trained weights and downloaded research artifacts are distributed separately through Hugging Face; their terms are governed by those repositories. This code license does not relabel those external assets or grant rights to the private CFD platform.

Use [CITATION.cff](CITATION.cff) to cite this software, and cite the data/model repository revision used in your experiment. Source mappings and adaptation notes are recorded in [PROVENANCE.json](PROVENANCE.json).
