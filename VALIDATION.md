# Release validation

The portable code was checked on Windows with Python 3.11.16, NumPy 2.4.6, SciPy 1.17.1, PyTorch 2.14.0+cu130, Matplotlib 3.11.2, openpyxl 3.1.5 and scikit-learn 1.9.1. These are the validation environment versions, not a requirement to replace an existing CUDA installation.

- Small synthetic CPU runs passed MLP optimization, training-only scaling, NPZ save/load identity and configuration/data/model resume checks.
- Exact GP means, standard deviations and marginal likelihood matched scikit-learn at the tested fixed hyperparameters. Analytic gradients matched centered finite differences. Small CPU/GPU comparisons passed.
- Both released representative models reproduced all 1,500 saved test predictions. The maximum absolute CL/CD differences were 1.11e-16 for MLP and 2.51e-11 for Kriging.
- The complete evaluation program ran on CPU using the published checkpoints for all 48 budget/seed records. Per-target RMSE and MAPE agreed with the released metrics; the largest absolute difference across those metric fields was 7.57e-10. This checks the evaluation pipeline, not a new training reproduction.
- All 15 PDF/PNG figure sets were regenerated from the frozen evidence. The Excel exporter produced 3,000 case/model rows and retained five-digit string IDs.
- The static explorer loaded all 1,500 test cases; case navigation and metric switching were checked in the browser.

## Browser prediction

The unquantized browser assets were exported from the two pinned NPZ checkpoints. `node tests/browser_core.mjs` checks both SHA-256 hashes and all 1,500 released test inputs. Maximum absolute differences from the stored Python predictions were:

| Model | CL | CD | CL/CD |
|---|---:|---:|---:|
| MLP | 1.11e-16 | 3.47e-18 | 2.13e-14 |
| Kriging | 1.15e-11 | 2.02e-11 | 6.60e-8 |

These values quantify agreement between implementations, not error relative to CFD or experiment. The MLP uses the same Tanh network and linear branch; Kriging retains all 10,000 centers. No model reduction was performed.

Coordinate checks cover signed three-column surfaces, reversed Selig contours, Lednicer blocks, translation and unit changes, explicit chord rotation, rank-aware least squares, and the prediction after a fit. Invalid numbers, incomplete surfaces, non-monotone ordering, open endpoints, poor fits and crossing surfaces are rejected. A non-CST outline is also fitted to check the domain warning. The browser UI was checked for direct CST and coordinate prediction, optional Kriging comparison, invalid input, extrapolation blocking/opt-in, copying a saved case, and clearing stale results after edits.

The browser checks run in a separate Node.js CI job. They exercise the same numerical modules as the Web Worker and require only the committed static assets. They do not establish aerodynamic accuracy for arbitrary coordinate files outside the studied design box.

No model was retrained on the CFD dataset during packaging. Reproducing the full architecture search and learning curves requires running the training commands in the README. Windows checks do not establish that every dependency version or hardware combination will give identical training results; the repository also provides a Linux CPU CI workflow.
