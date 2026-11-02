# Methods

## Data and geometry

The training pool contains 10,000 fine-resolution CFD cases from one Latin hypercube design. An independently sampled 2,000-case design supplies 500 validation and 1,500 test cases. The baseline (ID 00000) is excluded. The frozen split and nested training orders are downloaded with the input data. Reusing numeric IDs across pools does not identify the same geometry.

The 20-dimensional feature vector is $\mathbf a=(U_0,\ldots,U_9,L_0,\ldots,L_9)^T$. Both surfaces use

$$y_s(x)=x^{1/2}(1-x)\sum_{i=0}^{9}a_{s,i}\binom{9}{i}x^i(1-x)^{9-i},\qquad 0\le x\le1.$$

Coordinates are normalized by chord. There is no trailing-edge thickness term. The lower coefficients already contain their signs. Bounds are $a_j^0\pm0.1|a_j^0|$. The released CFD labels use CFL3D, SST, Mach 0.73, Reynolds number $6.5\times10^6$, and angle of attack 2.79 degrees (Case 9 corrected computational condition). Detailed geometry, meshes, convergence rules and engineering choices are documented in the [data-generation methodology](https://huggingface.co/datasets/ShuangheYang/rae2822-cst20-cfl3d-fine/blob/main/docs/DATA_GENERATION_EN.md).

## MLP

The selected architecture has hidden widths 64–64–64, Tanh activations, two outputs and a linear branch:

$$\widehat{\mathbf z}(\mathbf x)=f_\theta(\mathbf x)+W_{\rm lin}\mathbf x.$$

Inputs and targets are standardized using only the current training subset. The optional linear branch starts from a regularized least-squares fit. AdamW with cosine learning-rate decay is followed by optional full-batch float64 L-BFGS polishing. A validation checkpoint is chosen using

$$S=\frac13\sum_{q\in\{C_L,C_D,C_L/C_D\}}\frac{\operatorname{RMSE}_{q,\mathrm{val}}}{s_{q,\mathrm{train}}}.$$

The network still has only two outputs. Ratios are derived after inverse scaling. Twelve configurations vary depth, width, activation, linear branch, regularization, log-drag transformation and an auxiliary ratio penalty. The frozen JSON contains every setting. Screening uses seed 11; the best three are confirmed with seeds 22 and 33. The representative learning-curve model uses the prespecified seed 11. It is not selected by final test error.

## Kriging

Each output has its own isotropic Matérn-5/2 covariance in standardized input space:

$$k(r)=\sigma_f^2\left(1+\frac{\sqrt5r}{\ell}+\frac{5r^2}{3\ell^2}\right)\exp\left(-\frac{\sqrt5r}{\ell}\right),\quad r=\|\mathbf x-\mathbf x'\|_2.$$

For a given output, all 20 directions share one length scale; CL and CD have independently fitted kernel parameters. The mean is the training target mean. Signal variance, length scale and nugget are optimized by exact training marginal likelihood. Diagonal jitter addresses Cholesky roundoff and is recorded separately from the fitted nugget.

$$\widehat y(\mathbf x_*)=m+\mathbf k_*^T(K+\sigma_n^2I)^{-1}(\mathbf y-m\mathbf1).$$

The implementation uses Cholesky solves, not an explicit inverse for prediction. It retains every designated training point. Blocked distance construction reduces temporary storage; it does not approximate the kernel. Dense training is approximately $O(n^3)$ time and $O(n^2)$ memory. Predictive means use stored training centers and solved coefficients.

There are two optimizer starts per output. At larger budgets, preceding-budget hyperparameters provide one warm start. One predefined bound expansion is allowed when the training likelihood gradient points outward at a boundary. The expansion and any optimizer termination issues remain in the records. Kernel fitting does not consult validation or test errors. At the full budget, seeds 22 and 33 explicitly reuse the single canonical fit; they are not independent Kriging repetitions.

## Evaluation

For $q\in\{C_L,C_D,C_L/C_D\}$,

$$\operatorname{RMSE}_q=\sqrt{\frac1N\sum_i(\widehat q_i-q_i)^2},\qquad
\operatorname{MAPE}_q=\frac{100\%}{N}\sum_i\left|\frac{\widehat q_i-q_i}{q_i}\right|.$$

The aggregate learning-curve score is

$$E_2=\frac{100\%}{2}\left(\frac{\operatorname{RMSE}_{C_L}}{s_{C_L,\mathrm{test}}}+\frac{\operatorname{RMSE}_{C_D}}{s_{C_D,\mathrm{test}}}\right).$$

The test standard deviations use `ddof=0` and remain the same for all models and budgets. E2 is not MAPE. Per-output RMSE and MAPE, ratio errors and upper-tail errors are retained. The fixed 500 validation labels are additional to the training budget on the horizontal axis.

The RMSE reduction is $G_q=100\%(1-\mathrm{RMSE}_{q,\mathrm{MLP}}/\mathrm{RMSE}_{q,\mathrm{Kriging}})$. Positive values favor MLP. The 95% intervals resample paired test cases 2,000 times with both trained models fixed. They assess test-sample variation; they do not include retraining variation, CFD discretization error or extrapolation uncertainty. Geometry/error associations are exploratory and do not establish causation.

Warm inference timing synchronizes CUDA work, uses already loaded models, includes scaling and output transfer, and reports medians over repeated calls. It excludes model loading, file reading, mesh generation and CFD. Reproduction timings depend on hardware and concurrent workloads.

## Coordinate inputs and browser inference

Browser inputs use the same 20-coefficient ordering, class function, polynomial degree and scaler parameters as the released models. Let $B_{ij}=x_i^{1/2}(1-x_i)\binom{9}{j}x_i^j(1-x_i)^{9-j}$. For each surface, coordinate fitting solves

$$\widehat{\mathbf a}_s=\operatorname*{arg\,min}_{\mathbf a_s}\|B\mathbf a_s-\mathbf y_s\|_2^2.$$

Column-pivoted Householder QR solves this unconstrained least-squares problem. The interface does not clip coefficients into the design box: doing so would change the supplied geometry. Chord translation, rotation (when specified) and a common scale for both axes preserve the shape. Auto-normalization assumes a horizontal, closed airfoil; manual leading/trailing-edge coordinates define the chord of rotated input.

The coordinate adapter requires at least 30 distinct x positions on each surface, endpoint coverage, finite values, monotone surface branches and no surface crossing. It rejects an open leading/trailing edge. Engineering gates require per-surface fit RMSE at most $5\times10^{-4}c$ and maximum deviation at the supplied points at most $2\times10^{-3}c$. The reconstructed geometry must have maximum thickness between 0.04 and 0.25 chord and maximum absolute camber no greater than 0.10 chord. These broad sanity gates do not extend the model's validated domain. The design-box check is separate; outside-box inference requires an explicit exploration choice and has no validated accuracy estimate.

The MLP implementation evaluates the original dense Tanh layers and linear branch in float64. Kriging evaluates the exact predictive mean using all 10,000 stored centers and the already solved coefficients; it does not refit a kernel in the browser. Both inverse-transform CL and CD before computing their ratio. Browser timing is not the GPU deployment timing reported in the study.

The JavaScript implementation and coordinate adapter have numerical parity and malformed-input tests. Model hashes are checked at load time. New inputs have no associated CFD truth: fit errors measure geometric approximation, not aerodynamic prediction error.

## Reproduction and interpretation

This repository adapts the released research algorithms for portable paths, CPU execution and pickle-free checkpoints. Existing model weights and scientific results are not refitted during packaging. Public artifact revisions and hashes are pinned. Exact bitwise training reproduction across different hardware and PyTorch versions is not promised; record the environment and compare numerical predictions.

The learning curves compare a fixed, full-budget-selected MLP pipeline with an isotropic Kriging pipeline. They do not establish that all neural networks outperform all Gaussian processes. ARD kernels, alternate means and sparse/structured GP methods were not evaluated. The same-domain evaluation pool has prior research exposure; new claims of generalization should use a new untouched dataset.

## References

1. Kulfan, B. M. (2008). Universal Parametric Geometry Representation Method. *Journal of Aircraft*, 45(1), 142–158. [doi:10.2514/1.29958](https://doi.org/10.2514/1.29958).
2. Rasmussen, C. E., and Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning*. MIT Press. [Book](https://gaussianprocess.org/gpml/).
3. Loshchilov, I., and Hutter, F. (2019). Decoupled Weight Decay Regularization. *ICLR*. [Paper](https://arxiv.org/abs/1711.05101).
4. Nocedal, J., and Wright, S. J. (2006). *Numerical Optimization*, second edition. Springer. [doi:10.1007/978-0-387-40065-5](https://doi.org/10.1007/978-0-387-40065-5).
5. Efron, B., and Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall. [doi:10.1007/978-1-4899-4541-9](https://doi.org/10.1007/978-1-4899-4541-9).
