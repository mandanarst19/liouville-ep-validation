"""Stage 1 -- validate the MC Liouville estimator on systems with known answers.
Run:  python3 stage1_toys.py      (NumPy + matplotlib only; ~1 min)
Writes results/stage1_results.json and figures/stage1_validation.png (paths relative to this file).
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from divergence import (mc_entropy_rate, divergence_fd, logdet_flow_jacobian,
                        integral_div_along)

HERE = Path(__file__).resolve().parent
(HERE / 'results').mkdir(exist_ok=True); (HERE / 'figures').mkdir(exist_ok=True)
rng = np.random.default_rng(12345)
R = {}          # results collected for the printout / json


# ---------------------------------------------------------------- toy systems
def osc(gam, om):
    F = lambda t, X: np.stack([X[:, 1], -om**2 * X[:, 0] - gam * X[:, 1]], axis=1)
    div = lambda t, X: np.full(len(X), -gam)
    return F, div


def lin(A):
    A = np.asarray(A, float)
    F = lambda t, X: X @ A.T
    div = lambda t, X: np.full(len(X), np.trace(A))
    return F, div


def vdp(mu):
    F = lambda t, X: np.stack([X[:, 1], mu * (1 - X[:, 0]**2) * X[:, 1] - X[:, 0]], axis=1)
    div = lambda t, X: mu * (1 - X[:, 0]**2)
    return F, div


def lorenz(sig=10.0, rho=28.0, bet=8.0 / 3.0):
    def F(t, X):
        x, y, z = X[:, 0], X[:, 1], X[:, 2]
        return np.stack([sig * (y - x), x * (rho - z) - y, x * y - bet * z], axis=1)
    div = lambda t, X: np.full(len(X), -(sig + 1 + bet))
    return F, div


# ---- A. damped linear oscillator: div = -gamma exactly --------------------
gam, om = 0.3, 2.0
F, div = osc(gam, om)
X0 = rng.normal(0, 0.5, size=(1000, 2))
a = mc_entropy_rate(F, div, X0, t_end=10, dt=0.01)
fd = lambda t, X: divergence_fd(F, t, X)
b = mc_entropy_rate(F, fd, X0, t_end=10, dt=0.01)
R["A_osc"] = dict(max_abs_err_analytic=float(np.abs(a["sdot"] + gam).max()),
                  max_abs_err_fd=float(np.abs(b["sdot"] + gam).max()),
                  S_end_est=float(a["S_minus_S0"][-1]), S_end_exact=-gam * 10,
                  sum_over_members_over_exact=float(a["div_all"][-1].sum() / (-gam)))

# ---- B. 2D non-normal linear system + independent Gaussian-entropy check ---
A = np.array([[0.2, 2.0], [-1.5, -0.9]])           # trace = -0.7
F, div = lin(A)
N = 400_000
Sig0 = np.array([[0.30, 0.10], [0.10, 0.20]])
X0 = rng.multivariate_normal([0, 0], Sig0, size=N)
res = mc_entropy_rate(F, div, X0, t_end=3, dt=0.01, record_every=50)
Xt = X0.copy()
S_gauss = []
Hs = lambda C: 0.5 * np.log(np.linalg.det(2 * np.pi * np.e * C))
S0 = Hs(np.cov(X0.T))
from divergence import rk4_step
t = 0.0
for n in range(300):
    Xt = rk4_step(F, t, Xt, 0.01); t += 0.01
    if (n + 1) % 50 == 0:
        S_gauss.append(Hs(np.cov(Xt.T)) - S0)
S_gauss = np.array(S_gauss)
t_rec = res["t"][1:]
R["B_lin2d"] = dict(trace=float(np.trace(A)),
                    S_est=res["S_minus_S0"][1:].tolist(),
                    S_gauss_covariance=S_gauss.tolist(),
                    S_exact=(np.trace(A) * t_rec).tolist(),
                    max_abs_dev_gauss_vs_exact=float(np.abs(S_gauss - np.trace(A) * t_rec).max()))

# ---- C. time-dependent damping + forcing ----------------------------------
g0, om = 0.3, 2.0
gt = lambda t: g0 * (1 + 0.5 * np.sin(2 * t))
Fc = lambda t, X: np.stack([X[:, 1], -om**2 * X[:, 0] - gt(t) * X[:, 1] + 0.5 * np.cos(3 * t)], axis=1)
dc = lambda t, X: np.full(len(X), -gt(t))
X0 = rng.normal(0, 0.5, size=(500, 2))
c = mc_entropy_rate(Fc, dc, X0, t_end=10, dt=0.005)
exact = -g0 * (c["t"] + 0.25 * (1 - np.cos(2 * c["t"])))
R["C_timedep"] = dict(max_abs_dev_S=float(np.abs(c["S_minus_S0"] - exact).max()))

# ---- D. Jacobi's formula on nonlinear systems (per trajectory, no sampling) -
rows = []
Fv, dv = vdp(1.5)
Fl, dl = lorenz()
for name, (FF, dd, T, dt_, pts) in {
        "VdP mu=1.5, t=3": (Fv, dv, 3.0, 0.002, rng.normal(0, 1.0, size=(5, 2))),
        "Lorenz-63, t=0.5": (Fl, dl, 0.5, 0.0005, rng.normal(0, 5.0, size=(5, 3)) + [0, 0, 25]),
        "Lorenz-63, t=1.0": (Fl, dl, 1.0, 0.0005, rng.normal(0, 5.0, size=(5, 3)) + [0, 0, 25])}.items():
    for x0 in pts:
        ld, sg = logdet_flow_jacobian(FF, x0, T, dt_)
        iv = integral_div_along(FF, dd, x0, T, dt_)
        fdd = lambda t, X, FF=FF: divergence_fd(FF, t, X)
        ivf = integral_div_along(FF, fdd, x0, T, dt_)
        rows.append((name, ld, iv, ivf, sg))
R["D_jacobi"] = [dict(system=r[0], logdet_flow=r[1], int_div_analytic=r[2], int_div_fd=r[3], det_sign=r[4])
                 for r in rows]
R["D_jacobi_max_abs_diff"] = {}
for name in sorted(set(r[0] for r in rows)):
    R["D_jacobi_max_abs_diff"][name] = dict(
        analytic=float(max(abs(r[1] - r[2]) for r in rows if r[0] == name)),
        fd=float(max(abs(r[1] - r[3]) for r in rows if r[0] == name)))

# ---- E. N^-1/2 scaling of the MC error (Van der Pol; state-dependent div) ---
Fv, dv = vdp(1.5)
Nbig = 2_000_000
Xb = rng.normal([1.0, 0.5], 0.3, size=(Nbig, 2))
big = mc_entropy_rate(Fv, dv, Xb, t_end=4.0, dt=0.01, record_every=400)
dvals = big["div_all"][-1]                         # div at t=4 for all trajectories
ref, sd_div = dvals.mean(), dvals.std()
Ns = np.array([100, 300, 1000, 3000, 10000, 30000])
rms = []
for n in Ns:
    errs = [dvals[rng.choice(Nbig, n, replace=False)].mean() - ref for _ in range(300)]
    rms.append(np.sqrt(np.mean(np.square(errs))))
rms = np.array(rms)
slope = np.polyfit(np.log(Ns), np.log(rms), 1)[0]
R["E_scaling"] = dict(ref_sdot_t4=float(ref), std_div=float(sd_div), N=Ns.tolist(), rms_err=rms.tolist(),
                      predicted_std_over_sqrtN=(sd_div / np.sqrt(Ns)).tolist(), fitted_slope=float(slope))

# ---- F. forward-Euler map vs continuous divergence -------------------------
A = np.array([[0.2, 2.0], [-1.5, -0.9]])
F, div = lin(A)
tT = 2.0
dts = np.array([0.2, 0.1, 0.05, 0.025, 0.0125])
cont, disc = [], []
for dt_ in dts:
    n = int(round(tT / dt_))
    X0 = rng.normal(0, 0.3, size=(10, 2))
    r = mc_entropy_rate(F, div, X0, tT, dt_, method="euler")
    cont.append(r["S_minus_S0"][-1])                      # sum of dt * tr J  (continuous divergence)
    disc.append(n * np.log(abs(np.linalg.det(np.eye(2) + dt_ * A))))  # true volume change of Euler map
cont, disc = np.array(cont), np.array(disc)
slopeF = np.polyfit(np.log(dts), np.log(np.abs(cont - disc)), 1)[0]
R["F_euler"] = dict(dt=dts.tolist(), continuous_trJ=cont.tolist(), discrete_logdet=disc.tolist(),
                    abs_diff=np.abs(cont - disc).tolist(), slope=float(slopeF), exact=float(np.trace(A) * tT))

json.dump(R, open(HERE / "results" / "stage1_results.json", "w"), indent=1)

# ------------------------------------------------------------------ printout
print("A  damped oscillator (div = -gamma = -0.3)")
print(f"   max|sdot+gamma|  analytic div: {R['A_osc']['max_abs_err_analytic']:.2e}   finite-diff div: {R['A_osc']['max_abs_err_fd']:.2e}")
print(f"   S(10)-S(0): est {R['A_osc']['S_end_est']:.6f}  exact {R['A_osc']['S_end_exact']:.6f}")
print(f"   sum over members / exact = {R['A_osc']['sum_over_members_over_exact']:.1f}   (= N, so the mean needs the 1/N; N = 1000)")
print("B  2D non-normal linear system, tr A = -0.7; independent Gaussian entropy 0.5 ln det(2 pi e Cov)")
print(f"   max|dS_gauss - trA t| = {R['B_lin2d']['max_abs_dev_gauss_vs_exact']:.2e}  (N = 400000)")
print("C  time-dependent damping: max|S_est - S_exact| = %.2e" % R["C_timedep"]["max_abs_dev_S"])
print("D  Jacobi formula, ln|det flow Jacobian| vs int div ds (max over 5 start points)")
for k, v in R["D_jacobi_max_abs_diff"].items():
    print(f"   {k:20s} analytic div: {v['analytic']:.2e}   fd div: {v['fd']:.2e}")
print(f"E  MC error scaling, Van der Pol t=4: fitted slope {slope:.3f} (expected -0.5)")
for n, e, p in zip(Ns, rms, sd_div / np.sqrt(Ns)):
    print(f"   N={n:6d}  rms err {e:.3e}   std/sqrtN {p:.3e}")
print("F  forward Euler: continuous sum(dt trJ) vs true Euler-map ln det; exact = %.3f" % R["F_euler"]["exact"])
for d_, c_, di in zip(dts, cont, disc):
    print(f"   dt={d_:7.4f}  cont {c_:.5f}  discrete {di:.5f}  |diff| {abs(c_-di):.2e}")
print(f"   log-log slope of |diff| vs dt = {slopeF:.2f}")

# -------------------------------------------------------------------- figure
fig, ax = plt.subplots(2, 2, figsize=(10, 7.5))
a0 = ax[0, 0]
a0.plot(res["t"][1:], R["B_lin2d"]["S_est"], "-", label="MC estimator (cumulative)")
a0.plot(res["t"][1:], R["B_lin2d"]["S_gauss_covariance"], "o", ms=5, label=r"$\frac{1}{2}\ln\det(2\pi e\,\mathrm{Cov})$")
a0.plot(res["t"][1:], R["B_lin2d"]["S_exact"], "k--", lw=1, label=r"exact: $\mathrm{tr}A\cdot t$")
a0.set_xlabel("t"); a0.set_ylabel(r"$S(t)-S(0)$"); a0.set_title("B: 2D linear system, entropy change"); a0.legend(fontsize=8)
a1 = ax[0, 1]
for k, nm in enumerate(R["D_jacobi_max_abs_diff"]):
    sub = [r for r in R["D_jacobi"] if r["system"] == nm]
    a1.semilogy(np.full(len(sub), k) + 0.02 * np.arange(len(sub)),
                [max(abs(r["logdet_flow"] - r["int_div_analytic"]), 1e-16) for r in sub], "o")
a1.set_xticks(range(len(R["D_jacobi_max_abs_diff"])))
a1.set_xticklabels([n.replace(", ", "\n") for n in R["D_jacobi_max_abs_diff"]], fontsize=8)
a1.set_ylabel(r"$|\ln\det\Phi_t-\int\mathrm{div}F\,ds|$"); a1.set_title("D: Jacobi identity per trajectory")
a2 = ax[1, 0]
a2.loglog(Ns, rms, "o-", label="rms error (300 subsets)")
a2.loglog(Ns, sd_div / np.sqrt(Ns), "k--", label=r"$\mathrm{std}(\mathrm{div})/\sqrt{N}$")
a2.set_xlabel("N"); a2.set_ylabel(r"error of $\dot S$ estimate"); a2.set_title(f"E: Van der Pol, slope {slope:.2f}"); a2.legend(fontsize=8)
a3 = ax[1, 1]
a3.loglog(dts, np.abs(cont - disc), "o-")
a3.set_xlabel("dt"); a3.set_ylabel(r"$|\sum dt\,\mathrm{tr}J-\sum\ln\det(I+dt\,J)|$")
a3.set_title(f"F: Euler map vs continuous divergence, slope {slopeF:.2f}", fontsize=9)
a3.set_xticks(dts); a3.set_xticklabels([f"{d:g}" for d in dts], rotation=45, fontsize=8)
a3.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()); a3.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
fig.tight_layout(); fig.savefig(HERE / "figures" / "stage1_validation.png", dpi=150)
print("saved results/stage1_results.json, figures/stage1_validation.png")
