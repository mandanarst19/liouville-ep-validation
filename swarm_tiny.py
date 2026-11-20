"""
swarm_tiny.py -- a small swarm (n = 20 particles, D = 2) used to check my reading of the generalized-Liouville
estimator of Egenlauf, Kroninger, Kung, Gaimann & Klopotek, arXiv:2607.29434:
  Eq. 32 (system entropy rate), Eqs. 9, 34, 35 (heat flow, sign convention) and Eq. 33 (driver work).
This is a validation of the formulas on a small system, NOT a reproduction of the paper's numbers.

Model (m_i = 1), forces as in Eqs. 10a-d, 11, 12:
  F^r  = sum_j Theta(rr-|x_k-x_j|) (x_k-x_j)/|x_k-x_j|^2        (Eq. 10a; the matching potential is V = -K ln(r/rr))
  F^a  = sum_j Theta(ra-|x_k-x_j|) (v_j - v_k)
  F^h  = x_h - x_k
  F^sc = - v (|v| - s)/s                                        (Eq. 10d)
  F^d  = Theta(rd-|x_k-x_d|) (x_k-x_d)/|x_k-x_d|^2
  F    = Kr F^r + Ka F^a + Kh F^h + Ksc F^sc + Kd F^d           (Eq. 12);   B = Ka F^a + Ksc F^sc  (non-Hamiltonian part)
The force capper (Eq. 15) is NOT applied, so that the algebra of Eq. 32 holds exactly in the check against finite
differences; the code counts how often |F| > Fcap would have triggered.

MY ASSUMPTIONS (initial conditions, homing layout and driver path are not specified in the paper):
  n = 20 particles in a box of side L = 16*sqrt(20/200) (same density as n = 200, L = 16); homing points x_h uniform
  in the box (fixed); initial ensemble q = x_h + N(0, 0.1^2), p ~ N(0, 0.05^2) (a proper Gaussian density rho_0);
  driver: circle of radius 1.5 around the box centre, angular speed 1 (a simple stand-in for a chaotic driver);
  regime: near-critically damped, Ksc = 0.4138, s = 0.0483 (Table II); other constants from Table I used as listed
  (Kh = 40, Kr = 40, Ka = 0.2, Kd = 2000, rr = ra = 1, rd = 2, Fcap = 800);
  RK4 time stepping (the forces depend on velocity, so velocity Verlet is not used).
"""
import numpy as np

D = 2
PAR = dict(Kh=40.0, Kr=40.0, Ka=0.2, Ksc=0.4138, s=0.0483, Kd=2000.0, rr=1.0, ra=1.0, rd=2.0, Fcap=800.0)


def make_system(n=20, seed=0, L=None):
    rng = np.random.default_rng(seed)
    L = L if L is not None else 16.0 * np.sqrt(n / 200.0)
    xh = rng.uniform(0, L, size=(n, D))
    return dict(n=n, L=L, xh=xh, centre=np.array([L / 2, L / 2]))


def driver(t, sysd, radius=1.5, omega=1.0):
    c = sysd["centre"]
    x = c + radius * np.array([np.cos(omega * t), np.sin(omega * t)])
    v = radius * omega * np.array([-np.sin(omega * t), np.cos(omega * t)])
    return x, v


def forces(t, q, p, sysd, par=PAR, driven=True):
    """q, p: (N, n, 2).  Returns F (N,n,2), B (N,n,2), Na (N,n), Fd (N,n,2), Fcap_hits (N,)"""
    d = q[:, :, None, :] - q[:, None, :, :]
    r2 = (d ** 2).sum(-1)
    n = q.shape[1]
    off = ~np.eye(n, dtype=bool)[None]
    r2s = np.where(off, r2, 1.0)
    r = np.sqrt(r2s)
    rep = (off & (r < par["rr"]))[..., None] * d / r2s[..., None]
    Fr = rep.sum(2)
    ma = off & (r < par["ra"])
    Na = ma.sum(2)
    dp = p[:, None, :, :] - p[:, :, None, :]          # p_j - p_k   (m = 1)
    Fa = (ma[..., None] * dp).sum(2)
    Fh = sysd["xh"][None] - q
    sp = np.sqrt((p ** 2).sum(-1, keepdims=True))
    Fsc = -p * (sp - par["s"]) / par["s"]
    if driven:
        xd, _ = driver(t, sysd)
        dd = q - xd
        rd2 = (dd ** 2).sum(-1, keepdims=True)
        Fd = (np.sqrt(rd2) < par["rd"]) * dd / rd2
    else:
        Fd = np.zeros_like(q)
    F = par["Kr"] * Fr + par["Ka"] * Fa + par["Kh"] * Fh + par["Ksc"] * Fsc + (par["Kd"] * Fd if driven else 0.0)
    B = par["Ka"] * Fa + par["Ksc"] * Fsc
    hits = (np.sqrt((F ** 2).sum(-1)) > par["Fcap"]).sum(1)
    return F, B, Na, Fd, hits


def rhs(t, q, p, sysd, **kw):
    return p, forces(t, q, p, sysd, **kw)[0]


def rk4(t, q, p, dt, sysd, **kw):
    k1q, k1p = rhs(t, q, p, sysd, **kw)
    k2q, k2p = rhs(t + dt / 2, q + dt / 2 * k1q, p + dt / 2 * k1p, sysd, **kw)
    k3q, k3p = rhs(t + dt / 2, q + dt / 2 * k2q, p + dt / 2 * k2p, sysd, **kw)
    k4q, k4p = rhs(t + dt, q + dt * k3q, p + dt * k3p, sysd, **kw)
    return q + dt / 6 * (k1q + 2 * k2q + 2 * k3q + k4q), p + dt / 6 * (k1p + 2 * k2p + 2 * k3p + k4p)


# ------------------------------------------------------------ the quantities under test
def div_eq32(p, Na, par=PAR):
    """Integrand of Eq. 32 (m=1):  -sum_i [ D Ka N_i + Ksc/s ((D+1)|p_i| - D s) ]   -> (N,)"""
    sp = np.sqrt((p ** 2).sum(-1))
    return -(D * par["Ka"] * Na + par["Ksc"] / par["s"] * ((D + 1) * sp - D * par["s"])).sum(1)


def div_fd(t, q, p, sysd, h=1e-6, **kw):
    """Finite-difference sum_{i,alpha} d(pdot_{i alpha})/d(p_{i alpha}).  (d qdot/dq = 0, Hamiltonian part drops out.)"""
    out = np.zeros(q.shape[0])
    n = q.shape[1]
    for i in range(n):
        for a in range(D):
            pp, pm = p.copy(), p.copy()
            pp[:, i, a] += h
            pm[:, i, a] -= h
            out += (forces(t, q, pp, sysd, **kw)[0][:, i, a] - forces(t, q, pm, sysd, **kw)[0][:, i, a]) / (2 * h)
    return out


def heat_terms(p, B):
    """Eq. 9/34/35 per state:  pB_i = p_i . B_i (m=1);  Qdot = -sum_i pB_i Theta(-pB_i) >= 0.
    Returns Qdot, P_gain = sum_i pB_i Theta(+pB_i), P_net = sum_i pB_i (all (N,))"""
    pB = (p * B).sum(-1)
    Q = -(pB * (pB < 0)).sum(1)
    G = (pB * (pB > 0)).sum(1)
    return Q, G, pB.sum(1), pB


def driver_power(t, Fd, sysd, par=PAR):
    """Eq. 33:  dH/dt|explicit = d V_ext/dt = sum_i Kd F^d_i . xdot_d   (derived: V depends on q - x_d(t), so dV/dt = F^d . xdot_d)"""
    _, vd = driver(t, sysd)
    return par["Kd"] * (Fd * vd).sum((1, 2))


def energy(t, q, p, sysd, par=PAR, driven=True):
    """E = sum p^2/2 + V(Q) + Vext, with V from the FORCES of Eq. 10 (log potential for the 1/r repulsion)."""
    n = q.shape[1]
    Ekin = 0.5 * (p ** 2).sum((1, 2))
    d = q[:, :, None, :] - q[:, None, :, :]
    r = np.sqrt((d ** 2).sum(-1) + np.eye(n)[None])
    iu = np.triu(np.ones((n, n), bool), 1)[None]
    Vrep = (par["Kr"] * (-np.log(np.minimum(r, par["rr"]) / par["rr"])) * iu).sum((1, 2))      # = 0 beyond rr (continuous)
    Vh = 0.5 * par["Kh"] * ((q - sysd["xh"][None]) ** 2).sum((1, 2))
    E = Ekin + Vrep + Vh
    if driven:
        xd, _ = driver(t, sysd)
        rd = np.sqrt(((q - xd) ** 2).sum(-1))
        E = E + (par["Kd"] * (-np.log(np.minimum(rd, par["rd"]) / par["rd"]))).sum(1)
    return E
