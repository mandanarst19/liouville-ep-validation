"""
divergence.py -- Monte-Carlo estimator of the phase-space contraction rate
(generalized-Liouville estimator of Egenlauf, Kroninger, Kung, Gaimann & Klopotek, arXiv:2607.29434,
Sec. III.C.1, Eqs. 24-30).

Theory (one paragraph, so every line below can be defended):
  d xi/dt = F(xi, t),  rho_t evolves by  d rho/dt + div(rho F) = 0.
  Along a trajectory:  d ln rho_t(xi(t)) / dt = - div F(xi(t), t)          (Eq. 26)
  Gibbs entropy S(t) = -E_{rho_t}[ln rho_t] = -E_{rho_0}[ln rho_t(xi(t))]
  => dS/dt = E_{rho_0}[ div F(xi(t), t) ]                                   (Eq. 29)
  MC estimate with N initial states drawn from rho_0:
       S_dot_hat(t) = (1/N) * sum_k div F(xi_k(t), t)
  The estimator is the sample MEAN over the N members (the 1/N normalisation of the expectation in Eq. 29);
  stage1_toys.py (test A) checks this normalisation numerically.
  Error of the estimate: std(div)/sqrt(N), independent of the dimension s.

Conventions
  X has shape (N, s): N ensemble members, s state variables.
  F(t, X)   -> (N, s)   vector field
  div(t, X) -> (N,)     divergence  sum_i dF_i/dx_i  (analytic) -- or use divergence_fd
"""
import numpy as np


# ----------------------------------------------------------------- integrators
def euler_step(F, t, X, dt):
    return X + dt * F(t, X)


def rk4_step(F, t, X, dt):
    k1 = F(t, X)
    k2 = F(t + 0.5 * dt, X + 0.5 * dt * k1)
    k3 = F(t + 0.5 * dt, X + 0.5 * dt * k2)
    k4 = F(t + dt, X + dt * k3)
    return X + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


STEPPERS = {"euler": euler_step, "rk4": rk4_step}


# ------------------------------------------------------ divergence evaluators
def divergence_fd(F, t, X, h=1e-6):
    """Central finite-difference divergence: sum_i [F_i(x+h e_i) - F_i(x-h e_i)]/(2h).
    Only the DIAGONAL of the Jacobian is needed, so 2*s evaluations of F."""
    N, s = X.shape
    out = np.zeros(N)
    for i in range(s):
        Xp, Xm = X.copy(), X.copy()
        Xp[:, i] += h
        Xm[:, i] -= h
        out += (F(t, Xp)[:, i] - F(t, Xm)[:, i]) / (2 * h)
    return out


# ------------------------------------------------------------- the estimator
def mc_entropy_rate(F, div, X0, t_end, dt, method="rk4", record_every=1):
    """Integrate the ensemble X0 and record the MC estimate of dS/dt.

    Returns dict with
      t        (n_rec,)       recorded times
      sdot     (n_rec,)       (1/N) sum_k div F(xi_k(t), t)           [the estimator]
      sdot_se  (n_rec,)       std_k(div)/sqrt(N)                       [MC standard error]
      S_minus_S0 (n_rec,)     time integral of sdot (trapezoid)        [= S(t) - S(0)]
      div_all  (n_rec, N)     per-trajectory divergence (for diagnostics)
    """
    step = STEPPERS[method]
    X = np.array(X0, dtype=float, copy=True)
    N = X.shape[0]
    n_steps = int(round(t_end / dt))
    ts, sd, se, dall = [], [], [], []
    t = 0.0
    for n in range(n_steps + 1):
        if n % record_every == 0:
            d = div(t, X)
            ts.append(t)
            sd.append(d.mean())                      # <- the 1/N normalisation
            se.append(d.std(ddof=1) / np.sqrt(N) if N > 1 else np.nan)
            dall.append(d)
        if n < n_steps:
            X = step(F, t, X, dt)
            t += dt
    ts, sd, se = np.array(ts), np.array(sd), np.array(se)
    S = np.concatenate([[0.0], np.cumsum(0.5 * (sd[1:] + sd[:-1]) * np.diff(ts))])
    return dict(t=ts, sdot=sd, sdot_se=se, S_minus_S0=S, div_all=np.array(dall), X_final=X)


# --------------------------------------- independent check: Jacobi's formula
def flow_map(F, X0, t_end, dt, method="rk4"):
    step = STEPPERS[method]
    X = np.array(X0, dtype=float, copy=True)
    t = 0.0
    for _ in range(int(round(t_end / dt))):
        X = step(F, t, X, dt)
        t += dt
    return X


def logdet_flow_jacobian(F, x0, t_end, dt, h=1e-6, method="rk4"):
    """ln |det d xi(t)/d xi(0)| for ONE initial point, by finite differences of the
    flow map. Jacobi/Liouville: this must equal  int_0^t div F(xi(s), s) ds."""
    s = len(x0)
    pts = np.tile(np.asarray(x0, float), (2 * s, 1))
    for i in range(s):
        pts[2 * i, i] += h
        pts[2 * i + 1, i] -= h
    out = flow_map(F, pts, t_end, dt, method)
    J = np.zeros((s, s))
    for i in range(s):
        J[:, i] = (out[2 * i] - out[2 * i + 1]) / (2 * h)
    sign, ld = np.linalg.slogdet(J)
    return ld, sign


def integral_div_along(F, div, x0, t_end, dt, method="rk4"):
    """int_0^t div F(xi(s), s) ds along one trajectory (trapezoid on the RK4 grid)."""
    step = STEPPERS[method]
    X = np.asarray(x0, float)[None, :].copy()
    t, acc = 0.0, 0.0
    d_prev = div(t, X)[0]
    for _ in range(int(round(t_end / dt))):
        X = step(F, t, X, dt)
        t += dt
        d_new = div(t, X)[0]
        acc += 0.5 * (d_prev + d_new) * dt
        d_prev = d_new
    return acc
