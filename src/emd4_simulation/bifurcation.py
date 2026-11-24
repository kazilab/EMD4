"""Equilibria, folds, hysteresis and recovery times of the senescence model.

At fixed exposure a multi-start root search locates equilibria. At zero
exposure the steady-state equations also reduce to a scalar equation for the
total senescent fraction. Stability and recovery times use the Jacobian of the
equilibrating states, including the delayed secretion and SASP states.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq, fsolve

from .model import (IDX, MARKERS, Intervention, Params, hill, initial_state,
                    rhs, simulate, stress_of, trace)

# Cumulative escape and log neighbour expansion are integrators outside the equilibrium subsystem.
EQ_VARS = ["R", "S1", "S2", "C1", "C2", "P"]
EQ_IDX = [IDX[v] for v in EQ_VARS]


@dataclass(frozen=True)
class Equilibrium:
    E: float
    S: float
    S1: float
    S2: float
    P: float
    stable: bool
    lam_max: float          # dominant eigenvalue, real part

    @property
    def recovery_time(self) -> float:
        """Return time in days from the dominant stable eigenvalue."""
        return np.inf if self.lam_max >= 0 else -1.0 / self.lam_max


def _eq_residual(z: np.ndarray, p: Params, iv: Intervention, E: float) -> np.ndarray:
    y = np.zeros(len(IDX))
    for v, val in zip(EQ_VARS, z):
        y[IDX[v]] = val
    for m in MARKERS:                        # marker lags sit at S at equilibrium
        y[IDX[m.key]] = z[1] + z[2]
    d = rhs(0.0, y, p, iv, E, np.inf)
    return d[EQ_IDX]


def _full_jacobian(y: np.ndarray, p: Params, iv: Intervention,
                   E: float, eps: float = 1e-7) -> np.ndarray:
    """Numerical Jacobian of the equilibrating subsystem."""
    n = len(EQ_IDX)
    J = np.zeros((n, n))
    f0 = rhs(0.0, y, p, iv, E, np.inf)[EQ_IDX]
    for j, idx in enumerate(EQ_IDX):
        yp = y.copy()
        h = eps * max(1.0, abs(y[idx]))
        yp[idx] += h
        J[:, j] = (rhs(0.0, yp, p, iv, E, np.inf)[EQ_IDX] - f0) / h
    return J


def equilibria(p: Params, iv: Intervention, E: float,
               n_start: int = 12) -> list[Equilibrium]:
    """Find and classify equilibria at E using a grid of primary and secondary starting fractions.
    """
    R_star = (p.k_E * E + p.r_basal) / p.k_R
    q2 = p.q_sec * iv.q_sec
    found: list[Equilibrium] = []

    grid = np.linspace(0.0, 0.98, n_start)
    for a in grid:
        for b in (0.0, 0.3, 0.6, 0.9):
            s1 = a * (1.0 - b)
            s2 = a * b
            z0 = np.array([R_star, s1, s2, s1, s2,
                           p.k_sasp * (s1 + q2 * s2) / p.d_P])
            z, info, ier, _ = fsolve(_eq_residual, z0, args=(p, iv, E),
                                     full_output=True, xtol=1e-12)
            if ier != 1:
                continue
            S1, S2 = z[1], z[2]
            if S1 < -1e-8 or S2 < -1e-8 or S1 + S2 > 1.0 + 1e-8:
                continue
            S1, S2 = max(S1, 0.0), max(S2, 0.0)
            S = S1 + S2
            if any(abs(S - e.S) < 1e-6 for e in found):
                continue

            y = np.zeros(len(IDX))
            for v, val in zip(EQ_VARS, z):
                y[IDX[v]] = val
            for m in MARKERS:
                y[IDX[m.key]] = S
            lam = np.linalg.eigvals(_full_jacobian(y, p, iv, E))
            lam_max = float(np.max(lam.real))
            found.append(Equilibrium(E, S, S1, S2, float(z[5]),
                                     lam_max < 0, lam_max))

    return sorted(found, key=lambda e: e.S)


def branch(p: Params, iv: Intervention, E_grid: np.ndarray) -> list[list[Equilibrium]]:
    return [equilibria(p, iv, float(E)) for E in E_grid]


def count_stable(p: Params, iv: Intervention, E: float) -> int:
    return sum(1 for e in equilibria(p, iv, E) if e.stable)


# At equilibrium C_i = S_i and P = k_sasp (S1 + q2 S2) / d_P. With total
# removal g and direct induction a, S1 = a(1 − S)/g and S2 = S − S1.
# These relations reduce the zero-exposure problem to one equation in S.

def _zero_exposure_scalar(p: Params, iv: Intervention):
    """Return ``(F, split, g)`` for the scalar reduction at E = 0."""
    R_star = p.r_basal / p.k_R
    a = p.k_S * stress_of(R_star, p)
    gam = (p.gam_immune * iv.immune + p.gam_intrinsic) * iv.senolytic
    g = gam + p.k_esc * iv.escape
    q2 = p.q_sec * iv.q_sec

    def split(S: float) -> tuple[float, float, float]:
        S1 = a * (1.0 - S) / g
        S2 = S - S1
        P = (p.k_sasp / p.d_P) * (S1 + q2 * S2)
        return S1, S2, P

    def F(S: float) -> float:
        _, _, P = split(S)
        H = hill(P * iv.sasp_neut, p.K_P, p.n_P)
        return (1.0 - S) * (a + p.k_p * iv.paracrine * H) / g - S

    return F, split, g


def equilibria_at_zero(p: Params, iv: Intervention | None = None,
                       n_scan: int = 4001) -> list[Equilibrium]:
    """Bracket roots of the scalar zero-exposure equation and classify their stability."""
    iv = iv or Intervention()
    F, split, _ = _zero_exposure_scalar(p, iv)

    def _bisect(lo: float, hi: float) -> float:
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if F(lo) * F(mid) <= 0.0:
                hi = mid
            else:
                lo = mid
        return 0.5 * (lo + hi)

    def _scan(grid: np.ndarray, f: np.ndarray) -> list[float]:
        """Find grid zeros and bracket sign changes."""
        out: list[float] = []
        for i in range(len(grid)):
            if f[i] == 0.0:
                out.append(float(grid[i]))
        for i in range(len(grid) - 1):
            if f[i] * f[i + 1] < 0.0:
                out.append(_bisect(float(grid[i]), float(grid[i + 1])))
        return out

    grid = np.linspace(0.0, 1.0, n_scan)
    f = np.array([F(float(s)) for s in grid])
    roots = _scan(grid, f)

    # A pair of roots near a fold can lie within one grid interval with matching
    # endpoint signs. Refine local minima of |F| to resolve these pairs.
    af = np.abs(f)
    for i in range(1, len(grid) - 1):
        if af[i] > af[i - 1] or af[i] > af[i + 1]:
            continue                              # not a local minimum
        lo, hi = float(grid[i - 1]), float(grid[i + 1])
        if any(lo <= r <= hi for r in roots):
            continue                              # already found here
        sub = np.linspace(lo, hi, 2001)
        fs = np.array([F(float(s)) for s in sub])
        roots.extend(_scan(sub, fs))

    found: list[Equilibrium] = []
    for S in sorted(roots):
        if any(abs(S - e.S) < 1e-9 for e in found):
            continue
        if not (-1e-12 <= S <= 1.0 + 1e-12):
            continue
        S1, S2, P = split(S)
        if S2 < -1e-9:
            continue
        S1, S2 = max(S1, 0.0), max(S2, 0.0)
        y = np.zeros(len(IDX))
        y[IDX["R"]] = p.r_basal / p.k_R
        y[IDX["S1"]], y[IDX["S2"]] = S1, S2
        y[IDX["C1"]], y[IDX["C2"]] = S1, S2
        y[IDX["P"]] = P
        for m in MARKERS:
            y[IDX[m.key]] = S1 + S2
        lam = np.linalg.eigvals(_full_jacobian(y, p, iv, 0.0))
        lam_max = float(np.max(lam.real))
        found.append(Equilibrium(0.0, max(S, 0.0), S1, S2, P,
                                 lam_max < 0.0, lam_max))
    return found


def count_stable_at_zero(p: Params, iv: Intervention | None = None) -> int:
    """Count stable equilibria at zero exposure."""
    return sum(1 for e in equilibria_at_zero(p, iv) if e.stable)


def fold_curve(p: Params, iv: Intervention | None = None,
               n: int = 200001) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the analytic equilibrium curve q*(S) at zero exposure.

    With total removal g and direct induction a, S1 = a(1 − S)/g and S2 = S − S1.
    Inverting the paracrine Hill response gives the secondary secretion ratio
    for each admissible S. Folds are located as interior extrema on a finite S
    grid, whose resolution limits detection of narrow bistable windows.
    """
    iv = iv or Intervention()
    a = p.k_S * stress_of(p.r_basal / p.k_R, p)
    gam = (p.gam_immune * iv.immune + p.gam_intrinsic) * iv.senolytic
    g = gam + p.k_esc * iv.escape

    S = np.linspace(1e-9, 1.0 - 1e-9, n)
    with np.errstate(divide="ignore", invalid="ignore"):
        T = (S * g / (1.0 - S) - a) / (p.k_p * iv.paracrine)
        ok = np.isfinite(T) & (T > 0.0) & (T < 1.0)
        x = np.full_like(S, np.nan)
        x[ok] = p.K_P * (T[ok] / (1.0 - T[ok])) ** (1.0 / p.n_P)
        P = x / iv.sasp_neut
        S1 = a * (1.0 - S) / g
        S2 = S - S1
        q = (P * p.d_P / p.k_sasp - S1) / (S2 * iv.q_sec)
    q[~ok] = np.nan
    q[S2 <= 0.0] = np.nan
    return S, q


def _n_equilibria_at(q_star: np.ndarray, q: float) -> int:
    """Count crossings of the positive equilibrium curve at secretion ratio q."""
    d = q_star - q
    m = np.isfinite(d)
    d = d[m]
    if d.size < 2:
        return 0
    return int(np.sum(d[:-1] * d[1:] < 0.0) + np.sum(d == 0.0))


def q_sec_bistable_intervals(p: Params, iv: Intervention | None = None,
                             lo: float = 0.01, hi: float = 0.99,
                             n: int = 200001) -> list[tuple[float, float]]:
    """Find bistable secondary-secretion intervals intersecting [lo, hi].

    The extrema of q*(S) delimit folds. Count crossings between adjacent folds
    where the number of equilibria is constant. An upper fold within the search
    range marks the disappearance of the low-burden state.
    """
    S, q_star = fold_curve(p, iv, n)
    if np.isfinite(q_star).sum() < 2:
        return []

    # The low branch can be unresolved on the uniform grid near S = a0/(a0 + g),
    # where S2 vanishes. The two-crossing criterion compensates for this and was
    # checked against a derivative-based fold calculation for the reference ensemble.
    need = 2

    qv = q_star[np.isfinite(q_star)]
    folds = [float(qv[i]) for i in range(1, len(qv) - 1)
             if (qv[i] <= qv[i - 1] and qv[i] <= qv[i + 1])
             or (qv[i] >= qv[i - 1] and qv[i] >= qv[i + 1])]
    edges = sorted({lo, hi} | {f for f in folds if lo < f < hi})

    out: list[tuple[float, float]] = []
    for s_, e_ in zip(edges, edges[1:]):
        if _n_equilibria_at(q_star, 0.5 * (s_ + e_)) >= need:
            if out and abs(out[-1][1] - s_) < 1e-12:
                out[-1] = (out[-1][0], e_)      # merge touching pieces
            else:
                out.append((s_, e_))
    return out


def find_fold(p: Params, iv: Intervention,
              E_lo: float = 0.0, E_hi: float = 3.0,
              tol: float = 1e-4) -> float | None:
    """Locate the exposure at which the low stable branch disappears when bracketed."""
    n_lo, n_hi = count_stable(p, iv, E_lo), count_stable(p, iv, E_hi)
    if n_lo == n_hi:
        return None
    while E_hi - E_lo > tol:
        mid = 0.5 * (E_lo + E_hi)
        if count_stable(p, iv, mid) == n_lo:
            E_lo = mid
        else:
            E_hi = mid
    return 0.5 * (E_lo + E_hi)


def bistable_range(p: Params, iv: Intervention,
                   E_max: float = 3.0, n: int = 61) -> tuple[float, float] | None:
    """Exposure interval over which two stable states coexist."""
    Es = np.linspace(0.0, E_max, n)
    bi = [float(E) for E in Es if count_stable(p, iv, float(E)) >= 2]
    return (min(bi), max(bi)) if bi else None


# Hysteresis

def sweep(p: Params, iv: Intervention, E_path: np.ndarray,
          dwell: float = 400.0, y0: np.ndarray | None = None
          ) -> tuple[np.ndarray, np.ndarray]:
    """Carry the end state of each exposure step forward to the next step.

    Return the final senescent fractions and the terminal state. Equilibration
    close to a fold can take longer than the specified dwell time.
    """
    y = initial_state(p) if y0 is None else y0.copy()
    out = np.zeros_like(E_path, dtype=float)
    t = np.linspace(0.0, dwell, 3)
    for i, E in enumerate(E_path):
        ys = simulate(p, iv, float(E), t, t_off=np.inf, y0=y)
        y = ys[:, -1].copy()
        out[i] = y[IDX["S1"]] + y[IDX["S2"]]
    return out, y


def hysteresis_loop(p: Params, iv: Intervention, E_max: float = 1.2,
                    n: int = 121, dwell: float = 400.0) -> dict:
    """Integrate the gap between increasing and decreasing exposure sweeps.

    Both area and width give the integral of the absolute difference in senescent
    fraction over exposure. The jump at the fold limits the accuracy of a coarse
    exposure grid.
    """
    up_E = np.linspace(0.0, E_max, n)
    down_E = up_E[::-1]
    up_S, y_top = sweep(p, iv, up_E, dwell)
    # continue the down-sweep from the top of the up-sweep
    down_S, _ = sweep(p, iv, down_E, dwell, y0=y_top)

    area = float(np.trapezoid(np.abs(down_S[::-1] - up_S), up_E))
    return {"E": up_E, "up": up_S, "down": down_S[::-1],
            "width": area, "area": area, "n_grid": int(n)}


# Normal form

def normal_form(p: Params, iv: Intervention, E_fold: float,
                span: float = 0.12, n: int = 9) -> dict:
    """Fit the local saddle-node relation below the fold.

    Squared half-separation of the two lower equilibria scales with distance in
    dose from the fold. The local coordinate measures displacement from the
    critical state.
    """
    Es, half_gap = [], []
    for E in np.linspace(max(0.0, E_fold - span), E_fold, n)[:-1]:
        eq = equilibria(p, iv, float(E))
        lower = sorted(eq, key=lambda e: e.S)[:2]
        if len(lower) < 2:
            continue
        Es.append(float(E))
        half_gap.append(0.5 * (lower[1].S - lower[0].S))

    if len(Es) < 3:
        return {"ok": False, "reason": "fewer than 3 usable points below the fold"}

    Es = np.asarray(Es)
    g2 = np.asarray(half_gap) ** 2
    dmu = E_fold - Es                      # distance from criticality, in dose
    slope, intercept = np.polyfit(dmu, g2, 1)
    resid = g2 - (slope * dmu + intercept)
    ss = 1.0 - float(np.sum(resid ** 2) / max(np.sum((g2 - g2.mean()) ** 2), 1e-30))
    return {"ok": True, "E": Es, "half_gap": np.asarray(half_gap),
            "slope": float(slope), "intercept": float(intercept),
            "r2": ss,
            "note": "gap^2 linear in (E_fold - E) is the saddle-node signature; "
                    "sigma and the mu(E) map are not separately identifiable "
                    "from equilibrium positions alone"}


# Critical slowing

def recovery_times(p: Params, iv: Intervention,
                   E_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return recovery times on the low branch at exposures where that branch exists."""
    Es, taus = [], []
    for E in E_grid:
        eq = equilibria(p, iv, float(E))
        unstable = [e for e in eq if not e.stable]
        stable = [e for e in eq if e.stable]
        if not unstable or not stable:
            continue                      # past the fold: no low branch left
        # Restrict recovery to the stable branch below the separatrix.
        sep = min(e.S for e in unstable)
        low = [e for e in stable if e.S < sep]
        if not low:
            continue
        Es.append(float(E))
        taus.append(min(low, key=lambda e: e.S).recovery_time)
    return np.asarray(Es), np.asarray(taus)


def approach_grid(E_fold: float, n: int = 14,
                  closest: float = 3e-4) -> np.ndarray:
    """Space exposures geometrically below the fold to resolve critical slowing."""
    frac = np.concatenate([[1.0], np.logspace(np.log10(0.5), np.log10(closest), n - 1)])
    return E_fold * (1.0 - frac)


def containment_threshold(p: Params, iv: Intervention,
                          lo: float = 0.01, hi: float = 1.0) -> float | None:
    """Bisect the onset of bistability in secondary secretion at fixed other parameters.

    The lower endpoint must be monostable and the upper endpoint bistable.
    The ensemble uses q_sec_bistable_intervals to also detect bounded windows.
    """
    from dataclasses import replace

    def bistable_at(q: float) -> bool:
        return count_stable(replace(p, q_sec=q), iv, 0.0) >= 2

    if bistable_at(lo) or not bistable_at(hi):
        return None
    while hi - lo > 1e-4:
        mid = 0.5 * (lo + hi)
        if bistable_at(mid):
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)
