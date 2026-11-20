"""Stage 2 runner (small swarm, see swarm_tiny.py for model and assumptions).

    python3 stage2_run.py            # ~5 min on one CPU core; writes results/swarm_*.json
    python3 make_stage2_figure.py    # writes figures/stage2_swarm.png from those JSON files

Tests
  t1  Eq. 32 (analytic) vs central finite differences of the force, 50 generic states (also: the driver does not
      contribute to the divergence).
  t3  Jacobi identity on a 3-particle system: ln|det d xi(t)/d xi(0)| vs the time integral of the Eq. 32 divergence.
  series  n = 20, N = 200 ensemble members, t = 0..10, driven and undriven: system entropy rate (Eq. 32, MC mean and
          standard error), heat flow (Eqs. 34/35), driver power (Eq. 33), first-law bookkeeping.
An existing result file is loaded, not recomputed (delete it to recompute).
"""
import json
import time
from pathlib import Path

import numpy as np

import swarm_tiny as sw
from divergence import integral_div_along, logdet_flow_jacobian, rk4_step

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
RES.mkdir(exist_ok=True)

n, Nens, SEED = 20, 200, 0
DT = 0.002
sysd = sw.make_system(n, SEED)


def cached_json(path, fn):
    if path.exists():
        return json.load(open(path))
    out = fn()
    json.dump(out, open(path, "w"), indent=1)
    return out


def init(N, seed):
    r = np.random.default_rng(seed)
    q = sysd["xh"][None] + 0.1 * r.standard_normal((N, n, 2))
    p = 0.05 * r.standard_normal((N, n, 2))
    return q, p


# ---------------- t1: Eq. 32 vs finite differences of the force (generic states) -------------
def t1():
    r = np.random.default_rng(5)
    q = sysd["xh"][None] + 0.4 * r.standard_normal((50, n, 2))           # wide enough that N_a varies
    p = 0.3 * r.standard_normal((50, n, 2))
    _, _, Na, _, _ = sw.forces(0.3, q, p, sysd, driven=True)
    fd = sw.div_fd(0.3, q, p, sysd, driven=True)
    an = sw.div_eq32(p, Na)
    d1 = sw.div_fd(0.3, q, p, sysd, driven=True)
    d0 = sw.div_fd(0.3, q, p, sysd, driven=False)
    return dict(max_abs_diff_Eq32_vs_FD=float(np.abs(fd - an).max()), mean_div_FD=float(fd.mean()),
                mean_div_Eq32=float(an.mean()), mean_Na=float(Na.mean()), min_Na=int(Na.min()), max_Na=int(Na.max()),
                driver_contributes_to_divergence_max_abs=float(np.abs(d1 - d0).max()))


# ---------------- t3: Jacobi identity on a 3-particle system (short time, no Theta crossing) --------------------
def t3():
    sys3 = dict(n=3, xh=np.array([[0.0, 0.0], [0.5, 0.2], [0.2, 0.55]]), L=2.0, centre=np.array([0.3, 0.3]))
    q0 = np.array([[0.05, 0.02], [0.45, 0.25], [0.22, 0.5]])
    p0 = np.array([[0.1, -0.05], [-0.08, 0.12], [0.04, 0.09]])
    x0 = np.concatenate([q0.ravel(), p0.ravel()])

    def F(t, X):
        q = X[:, :6].reshape(-1, 3, 2); p = X[:, 6:].reshape(-1, 3, 2)
        return np.concatenate([p.reshape(len(X), -1), sw.forces(t, q, p, sys3, driven=False)[0].reshape(len(X), -1)], 1)

    def div(t, X):
        q = X[:, :6].reshape(-1, 3, 2); p = X[:, 6:].reshape(-1, 3, 2)
        return sw.div_eq32(p, sw.forces(t, q, p, sys3, driven=False)[2])

    # distances along the path: make sure no alignment/repulsion boundary (Theta) is crossed
    X = x0[None].copy(); t = 0.0; rmin, rmax = 9.0, 0.0
    for _ in range(25):
        q = X[:, :6].reshape(-1, 3, 2)
        dd = np.sqrt(((q[:, :, None] - q[:, None]) ** 2).sum(-1))[0][np.triu_indices(3, 1)]
        rmin, rmax = min(rmin, dd.min()), max(rmax, dd.max())
        X = rk4_step(F, t, X, 0.002); t += 0.002
    res = {"pair_distance_range_along_path": [float(rmin), float(rmax)],
           "alignment_radius": sw.PAR["ra"], "repulsion_radius": sw.PAR["rr"]}
    for h in (1e-4, 1e-5, 1e-6):
        ld, sg = logdet_flow_jacobian(F, x0, 0.05, 0.002, h=h)
        iv = integral_div_along(F, div, x0, 0.05, 0.002)
        res[f"h={h:g}"] = dict(logdet_flow=float(ld), int_div_Eq32=float(iv), abs_diff=float(abs(ld - iv)), sign=float(sg))
    return res


# ---------------- time series, driven vs undriven ---------------------------------------------------------------
COLS = ["t", "Ssys", "Ssys_se", "Q", "Q_se", "Pgain", "Pnet", "Wdrv", "Na", "speed", "frac_pB<0", "maxF"]


def run_series(driven, t_end=10.0, dt=DT, rec=10, N=Nens, seed=2):
    q, p = init(N, seed)
    steps = int(round(t_end / dt)); t = 0.0
    rows = []
    E0 = sw.energy(0.0, q, p, sysd, driven=driven)
    work_int = np.zeros(N); active_int = np.zeros(N); Q_int = np.zeros(N); prev = None
    hits_tot = 0
    for k in range(steps + 1):
        F, B, Na, Fd, hits = sw.forces(t, q, p, sysd, driven=driven)
        Q, G, Pn, pB = sw.heat_terms(p, B)
        W = sw.driver_power(t, Fd, sysd) if driven else np.zeros(N)
        cur = (W, Pn, Q)
        if prev is not None:
            work_int += 0.5 * (prev[0] + cur[0]) * dt; active_int += 0.5 * (prev[1] + cur[1]) * dt
            Q_int += 0.5 * (prev[2] + cur[2]) * dt
        prev = cur
        hits_tot += int((hits > 0).sum())
        if k % rec == 0:
            S = sw.div_eq32(p, Na)
            ssp = np.sqrt((p ** 2).sum(-1))
            rows.append([t, S.mean(), S.std(ddof=1) / np.sqrt(N), Q.mean(), Q.std(ddof=1) / np.sqrt(N), G.mean(), Pn.mean(),
                         W.mean(), Na.mean(), ssp.mean(), (pB < 0).mean(), np.abs(F).max()])
        if k < steps:
            q, p = sw.rk4(t, q, p, dt, sysd, driven=driven); t += dt
    E1 = sw.energy(t, q, p, sysd, driven=driven)
    first_law = (E1 - E0) - (work_int + active_int)
    return dict(columns=COLS, series=np.array(rows).tolist(), dt=dt, N=N, n=n, driven=driven,
                E_change_mean=float((E1 - E0).mean()), work=float(work_int.mean()), active_net=float(active_int.mean()),
                heat_integral=float(Q_int.mean()), first_law_resid_mean=float(first_law.mean()),
                first_law_resid_rms=float(np.sqrt((first_law ** 2).mean())),
                first_law_resid_over_scale=float(np.sqrt((first_law ** 2).mean()) /
                                                 (np.abs(work_int) + np.abs(active_int) + np.abs(E1 - E0)).mean()),
                frac_steps_with_cap_hit=hits_tot / (N * (steps + 1)))


def series(driven):
    tag = "driven" if driven else "undriven"
    def comp():
        t0 = time.time(); r = run_series(driven); print(f"  series {tag}: {time.time() - t0:.0f} s", flush=True)
        return r
    return cached_json(RES / f"swarm_series_{tag}_dt{DT}.json", comp)


if __name__ == "__main__":
    for name, fn in (("t1_eq32_vs_fd", t1), ("t3_jacobi_3particles", t3)):
        r = cached_json(RES / f"swarm_{name}.json", fn)
        print(name); print(json.dumps(r, indent=1))
    for driven in (False, True):
        r = series(driven)
        s = np.array(r["series"])
        print(f"\n== driven={driven}, dt={r['dt']}: first-law residual rms/scale {r['first_law_resid_over_scale']:.2e}; "
              f"mean dE {r['E_change_mean']:.3f} = work {r['work']:.3f} + active_net {r['active_net']:.3f} "
              f"(heat integral {r['heat_integral']:.3f}); cap-hit fraction {r['frac_steps_with_cap_hit']:.2e}")
        for k in (0, len(s) // 10, len(s) // 4, len(s) // 2, -1):
            print("   " + "  ".join(f"{c}={s[k, i]:.4g}" for i, c in enumerate(COLS[:6])))
