# liouville-ep-validation

Validation of the generalized-Liouville entropy-rate estimator described in

> P. Egenlauf, H. A. Kröninger, A. Kung, M. U. Gaimann and M. Klopotek, *Entropy production of active matter systems as indicator for computing performance*, arXiv:2607.29434 (2026).

This is a first step towards applying the same estimator to an electrothermal VO₂ neuristor reservoir.
The code is my own implementation of the formulas of that paper; it is a check of my reading of them, not a reproduction of their results.

## What is shown

**Stage 1 – toy systems with known answers** (`stage1_toys.py`, `figures/stage1_validation.png`, `results/stage1_results.json`).
The estimator is the sample mean of the divergence over N ensemble members, with standard error std/√N.

| test | result |
|---|---|
| A. damped oscillator (div = −γ) | rate exact to 1e-16 (analytic divergence) and 6e-12 (finite-difference divergence); S(10)−S(0) = −3.000000 |
| B. 2D non-normal linear system | cumulative estimate matches tr(A)·t and the Gaussian entropy ½ ln det(2πe Cov) to 6e-9 |
| C. time-dependent damping and forcing | entropy change matches the exact value to 1.3e-6 |
| D. Jacobi identity, ln\|det Φₜ\| vs ∫div F ds | max deviation 8e-7 (Van der Pol, t = 3), 1.3e-5 (Lorenz-63, t = 0.5), 7e-3 (Lorenz-63, t = 1; limited by the conditioning of the finite-difference flow Jacobian) |
| E. N-scaling of the MC error (Van der Pol) | fitted slope −0.50, equal to std/√N |
| F. forward-Euler map vs continuous divergence | difference ∝ dt (log-log slope 1.03) |

**Stage 2 – small swarm** (`swarm_tiny.py`, `stage2_run.py`, `make_stage2_figure.py`, `figures/stage2_swarm.png`).
- Eq. 32 (analytic) vs central finite differences of the force: max deviation 1.2e-6 over 50 generic states; the driver does not contribute to the divergence (≤ 1.2e-6).
- Jacobi identity for a 3-particle system (t = 0.05): ln\|det Φ\| = −9.87741 vs ∫div = −9.87776.
- Time series (n = 20, N = 200, t = 0–10): in both cases Ṡ_sys < 0 and Q̇ > 0. At t = 10, −Ṡ_sys ≈ 23 (undriven) and ≈ 383 (driven); Q̇ ≈ 0.11 and ≈ 224. The signs and the larger magnitude in the driven case match the description of their Fig. 2; I make no numerical comparison.

## Assumptions and limits
- 20 particles, a 200-member ensemble and one near-critically damped parameter set (Ksc = 0.4138, s = 0.0483, Table II; other constants as listed in Table I).
- The paper does not specify initial conditions, homing layout or driver path. These are my assumptions (see the docstring of `swarm_tiny.py`): Gaussian initial ensemble around random homing points, a circular driver path.
- I use the forces of Eq. 10. The force capper (Eq. 15) is not modelled, so the Eq. 32 check is exact. The code counts how often it would act (fraction of member-steps with at least one particle above Fcap = 800): 2e-5 in the undriven run and 0.32 in the driven run. The driven time series therefore describe the uncapped dynamics.
- RK4 time stepping, dt = 0.002; one random seed. No claim of numerical agreement with the paper.

## How to run
Python 3.13 (tested), NumPy and Matplotlib only.
```
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python3 stage1_toys.py           # ~1 min   -> results/stage1_results.json, figures/stage1_validation.png
python3 stage2_run.py            # ~7 min   -> results/swarm_*.json (existing files are reused; delete them to recompute)
python3 make_stage2_figure.py    # seconds  -> figures/stage2_swarm.png
```
Runtimes were measured on one CPU core.

## Next step
Applying the estimator to my VO₂ neuristor model (not included here).

Licence: MIT.
