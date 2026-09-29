"""Equilibrium structure: folds, hysteresis, the normal form, critical slowing.

The manuscript proposes a saddle-node bifurcation as a *candidate* model of a
tissue tipping point, and is explicit that whether threshold behaviour occurs is
an empirical question. This module is the apparatus that decides it for a given
parameter set, and -- more importantly -- the apparatus that measures how much
data it would take to tell the two answers apart.

Method note: equilibria are found by multi-start root-finding rather than by
arclength continuation. For a system this small that is both faster to write and
strictly more robust -- continuation can slide past a disconnected branch, a
dense multi-start cannot. At E = 0 specifically there is a stronger option and
it is the one the ensemble uses: :func:`equilibria_at_zero` reduces the system
to a scalar map and brackets EVERY sign change, so the zero-exposure count is an
exhaustive enumeration rather than a search. Any prose describing this module
must say root-finding, not continuation. Stability is then read off the FULL
numerical Jacobian, not the reduced steady-state map, because eliminating the
SASP and competence states would discard exactly the delays that set the
recovery timescale.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq, fsolve

from .model import (IDX, MARKERS, Intervention, Params, hill, initial_state,
                    rhs, simulate, stress_of, trace)

# The subsystem that actually equilibrates. X (cumulative escape) and lnN
# (neighbour expansion) are integrators -- they grow at a constant rate at the
# fixed point and are excluded by construction, not by oversight.
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
        """Time constant of return after a small perturbation, days.

        Diverges as the fold is approached -- this is the 'critical slowing'
        item in the manuscript's EMD4 evidence list, and it is measurable.
        """
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
    """All equilibria at exposure E, classified.

    Multi-start over a grid of (S1, S2); R and the competence/SASP states are
    algebraically slaved to the guess so the starts are already near-feasible.
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


# --- exact enumeration at E = 0 ------------------------------------------
# :func:`equilibria` multi-starts fsolve, which is robust but not a guarantee:
# a root with a narrow basin can be missed by every start. At E = 0 the system
# collapses to a SCALAR root problem, so the roots can be bracketed exhaustively
# on a dense sign-change scan instead of searched for. That matters because the
# ensemble's bistability count is a claim about *how many* stable states exist,
# and a missed root silently understates it.
#
# The reduction: at equilibrium C_i = S_i and P = k_sasp(S1 + q2*S2)/d_P, and
# with g = gam + k_esc and a = k_S*stress(R*),
#
#     S1 = a(1 - S)/g,   S2 = S - S1,   and S solves
#     F(S) = (1 - S)(a + k_p*paracrine*H(P(S)))/g - S = 0.
#
# Verified to agree with :func:`equilibria` on every draw of the default
# seed-4, n = 250 ensemble.

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
    """Every equilibrium at E = 0, by exhaustive bracketing of the scalar map.

    Stability is read from the eigenvalues of the same full Jacobian
    :func:`equilibria` uses, so the two agree by construction where both find a
    root; this one is simply harder to fool.
    """
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
        """Sign changes AND exact zeros on this grid."""
        out: list[float] = []
        for i in range(len(grid)):
            if f[i] == 0.0:                      # root sitting exactly ON a node
                out.append(float(grid[i]))       # (a strict f[i]*f[i+1] < 0 test
        for i in range(len(grid) - 1):           #  steps straight over these)
            if f[i] * f[i + 1] < 0.0:
                out.append(_bisect(float(grid[i]), float(grid[i + 1])))
        return out

    grid = np.linspace(0.0, 1.0, n_scan)
    f = np.array([F(float(s)) for s in grid])
    roots = _scan(grid, f)

    # A sign-change scan is blind to an EVEN number of roots inside one cell,
    # which is exactly the configuration near a fold: two equilibria approach
    # each other and annihilate, so just before they do they sit arbitrarily
    # close together and F dips to zero and back without changing sign. Refine
    # around every interior local minimum of |F| that the coarse pass left
    # unbracketed. This is what makes the enumeration exhaustive in practice
    # rather than only in principle.
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
    """Number of STABLE zero-exposure equilibria. Two or more = bistable."""
    return sum(1 for e in equilibria_at_zero(p, iv) if e.stable)


def fold_curve(p: Params, iv: Intervention | None = None,
               n: int = 200001) -> tuple[np.ndarray, np.ndarray]:
    """The equilibrium curve in the (S, q_sec) plane, computed ANALYTICALLY.

    F(S) increases monotonically with q_sec, so for each S there is at most one
    q_sec making that S an equilibrium -- and it can be written in closed form
    rather than searched for. Setting F = 0 and solving for the Hill argument,

        T  = [S*g/(1 - S) - a] / (k_p*paracrine)      (need 0 < T < 1)
        x  = K_P * (T / (1 - T))^(1/n_P)              (invert the Hill function)
        P  = x / sasp_neut
        q* = [P*d_P/k_sasp - S1] / (S2 * iv.q_sec),   S1 = a(1-S)/g, S2 = S - S1

    This is exact and vectorises, so the whole curve costs one array pass.

    Why it replaces the previous approach. Locating the bistable range by
    scanning q_sec and asking "are there two stable states here?" requires the
    scan to land INSIDE the bistable set. Draw 193 of the default ensemble is
    bistable only for q_sec in (0.104977, 0.105546) -- a window 5.7e-4 wide --
    so an 81-point scan steps straight over it and concludes the draw has no
    threshold at all. Worse, a scalar "distance to fold" built from the local
    maximum of F alone reports the wrong answer above the upper fold, because
    bistability needs the local maximum ABOVE zero *and* the local minimum
    BELOW it, and the earlier version tracked only the first of those.

    Reading the folds off this curve removes the search over q_sec entirely:
    the folds are exactly the interior local extrema of q*(S), found at grid
    precision in S, where the curve is smooth. That resolves windows far
    narrower than any affordable q_sec scan reaches.

    It is NOT an unconditional guarantee. The curve is analytic but its extrema
    are located numerically, so resolution still matters: on the default
    ensemble ``n = 2001`` recovers 180 of 250 draws, ``n = 20001`` recovers
    195, and ``n = 200001`` recovers 198, after which a tenfold refinement
    changes nothing. Describe this as "an analytic equilibrium curve with
    numerically located extrema, independently checked for this ensemble" --
    not as "the folds are computed analytically".
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
    """How many S solve q*(S) = q, i.e. how many equilibria exist at this q."""
    d = q_star - q
    m = np.isfinite(d)
    d = d[m]
    if d.size < 2:
        return 0
    return int(np.sum(d[:-1] * d[1:] < 0.0) + np.sum(d == 0.0))


def q_sec_bistable_intervals(p: Params, iv: Intervention | None = None,
                             lo: float = 0.01, hi: float = 0.99,
                             n: int = 200001) -> list[tuple[float, float]]:
    """q_sec ranges with three equilibria, clipped to [lo, hi].

    The interior local extrema of :func:`fold_curve` ARE the saddle-nodes: a
    local minimum of q*(S) is where the upper pair of equilibria is born, a
    local maximum is where the lower pair dies. Between consecutive folds the
    number of equilibria cannot change, so the bistable set is recovered by
    splitting [lo, hi] at the folds and counting crossings once per piece.

    Counting rather than pairing the extrema matters: most draws have only ONE
    fold inside the search range (the onset, with the upper fold beyond q = 1
    or absent), and an implementation that pairs a minimum with a maximum
    silently returns nothing for those.

    The set is returned as INTERVALS because it is not always a half-line. When
    an interval ends below ``hi`` the draw is re-entrant: bistability
    disappears again at high secondary secretion. That is not a return to
    containment -- see :func:`ensemble.q_sec_bistable_window`.
    """
    S, q_star = fold_curve(p, iv, n)
    if np.isfinite(q_star).sum() < 2:
        return []

    # WHY TWO CROSSINGS, AND WHY THAT IS A CALIBRATION RATHER THAN A PROOF.
    #
    # q*(S) is obtained by dividing through by S2 = S - a0(1-S)/g, which
    # vanishes at S = a0/(a0 + g) -- NOT at a0/g, as an earlier version of this
    # comment claimed. Near that asymptote q* diverges, so on a uniform S grid
    # the lowest equilibrium is often unresolved: it sits between grid points
    # where the curve is nearly vertical. Draw 23's low state, at 3.844e-4,
    # behaves that way, and requiring three crossings drops it.
    #
    # The earlier claim that the low state is NEVER a crossing, and exists at
    # every q_sec, is also false. For a0 > 0 it generally IS a crossing given
    # enough resolution -- draw 193 shows three crossings at q_sec = 0.1052661
    # and one at q_sec = 0.99 -- and the exactly empty state at a0 = 0 is a
    # separate equilibrium that this positive branch omits by construction.
    #
    # So `need = 2` is an empirical calibration for the default ensemble, not a
    # derivation: it compensates for an unresolved low branch near the
    # asymptote. It is validated, not assumed -- the resulting thresholds agree
    # with an independent derivative-based fold calculation on all 198 draws of
    # the seed-4, n = 250 ensemble to within 1.3e-11. Raising the cutoff to
    # three WITHOUT also clustering the grid near a0/(a0 + g) makes matters
    # worse (174-177 of 250), because it removes the compensation and not the
    # cause. A principled version needs a grid clustered at the asymptote or a
    # reparametrisation of the low branch; until then, re-validate this cutoff
    # against the independent calculation whenever the sampler or the S grid
    # changes.
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
    """Exposure at which the low branch is destroyed (the up-sweep fold).

    Bisection on the number of stable equilibria. Returns None if the system is
    monostable across the whole range -- which is a legitimate answer, and the
    one that means 'no tipping point here'.
    """
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


# --- hysteresis -----------------------------------------------------------

def sweep(p: Params, iv: Intervention, E_path: np.ndarray,
          dwell: float = 400.0, y0: np.ndarray | None = None
          ) -> tuple[np.ndarray, np.ndarray]:
    """Quasi-static sweep: hold each exposure long enough to settle, carrying
    the state forward. Returns (S at the end of each dwell, final state).

    The final state is returned so a down-sweep can continue from where the
    up-sweep finished without re-running it; ``hysteresis_loop`` used to
    discard it and then repeat the entire up-sweep to recover it.

    On ``dwell``: the justification used to read "400 days is ~10x the
    clearance timescale", which is the wrong timescale to compare against. Near
    a saddle-node the relevant timescale is the recovery time, and this build's
    own critical-slowing result reports it rising to ~1010 days within 2e-5 of
    the fold -- longer than any dwell used here. What saves the measurement is
    not that the dwell exceeds that time but that the divergence is confined to
    a vanishing neighbourhood of E*, so its contribution to the loop area is
    negligible: the area changes by 1e-5 between a 600-day and a 2400-day
    dwell. The dwell is adequate for the AREA, and would not be adequate for
    resolving the branch position arbitrarily close to the fold.
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
    """Up-sweep then down-sweep. ``width`` is the AREA between them.

    A single-valued dose-response -- however steep -- returns 0. That is the
    measurement that separates a tipping point from a steep sigmoid, and it is
    the reason the withdrawal arm is worth running.

    ``width`` is a misnomer kept for compatibility: the quantity is
    integral |S_down - S_up| dE, in units of senescent fraction x dose, and it
    is also returned as ``area``. It is NOT a width in E.

    On the grid: the integrand has a jump at the fold, where the up-sweep is
    still on the low branch and the down-sweep is on the high one, so the
    trapezoid rule carries an O(h) error there and ``n`` has to be large enough
    for it not to matter. The default used to be 25, and ``run.py`` called this
    with n = 16, which put the reported area 5% above its converged value
    (0.0678 at n = 16, 0.0657 at n = 31, 0.0641 at n = 121, against an analytic
    estimate of S_upper x E_fold = 0.0648). At n = 121 the residual bias is
    below 1%.
    """
    up_E = np.linspace(0.0, E_max, n)
    down_E = up_E[::-1]
    up_S, y_top = sweep(p, iv, up_E, dwell)
    # continue the down-sweep from the top of the up-sweep
    down_S, _ = sweep(p, iv, down_E, dwell, y0=y_top)

    area = float(np.trapezoid(np.abs(down_S[::-1] - up_S), up_E))
    return {"E": up_E, "up": up_S, "down": down_S[::-1],
            "width": area, "area": area, "n_grid": int(n)}


# --- normal form ----------------------------------------------------------

def normal_form(p: Params, iv: Intervention, E_fold: float,
                span: float = 0.12, n: int = 9) -> dict:
    """Local reduction to dx/dt = mu + sigma*x^2 near the fold.

    x is a LOCAL COORDINATE relative to the critical state, not an absolute
    senescent burden -- the manuscript is explicit about this, and it is what
    keeps the negative branch from being read as a negative number of cells.

    Fitted from the collision of the two lower equilibria: below the fold their
    separation scales as sqrt(-mu/sigma), so a regression of (separation/2)^2 on
    (E_fold - E) recovers sigma up to the mu(E) mapping.
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


# --- critical slowing -----------------------------------------------------

def recovery_times(p: Params, iv: Intervention,
                   E_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Low-branch recovery time vs exposure. Diverges at the fold.

    Returned alongside the exposures at which a low branch still exists, so the
    caller can plot the divergence without inventing points past the fold.
    """
    Es, taus = [], []
    for E in E_grid:
        eq = equilibria(p, iv, float(E))
        unstable = [e for e in eq if not e.stable]
        stable = [e for e in eq if e.stable]
        if not unstable or not stable:
            continue                      # past the fold: no low branch left
        # The low branch is the stable state BELOW the separatrix. Taking
        # min(stable) alone silently hops to the upper branch once the fold is
        # passed, and the two happen to share a recovery time here -- which
        # reports the divergence as no divergence at all.
        sep = min(e.S for e in unstable)
        low = [e for e in stable if e.S < sep]
        if not low:
            continue
        Es.append(float(E))
        taus.append(min(low, key=lambda e: e.S).recovery_time)
    return np.asarray(Es), np.asarray(taus)


def approach_grid(E_fold: float, n: int = 14,
                  closest: float = 3e-4) -> np.ndarray:
    """Exposures approaching a fold geometrically.

    A linear grid cannot show a square-root divergence: it spends most of its
    points where nothing is happening and then steps over the interesting
    decade. This clusters where the slowing actually occurs.
    """
    frac = np.concatenate([[1.0], np.logspace(np.log10(0.5), np.log10(closest), n - 1)])
    return E_fold * (1.0 - frac)


def containment_threshold(p: Params, iv: Intervention,
                          lo: float = 0.01, hi: float = 1.0) -> float | None:
    """Critical q_sec: the secondary-cell secretion rate below which the
    paracrine loop cannot self-sustain and senescence stays contained.

    This is the single number that decides whether EMD4 has a tipping point at
    all, and it is a *measurable* quantity -- the ratio of SASP output between
    primary and secondary senescent cells. Reporting it turns the bistability
    question into an experiment.
    
    NOTE -- there are two estimators of q_sec* in this build, and they are not
    the same procedure. This one counts equilibria by root-finding on the
    reduced system (a STRUCTURAL test: how many fixed points exist). The
    ensemble uses ``ensemble.q_sec_critical``, which bisects on
    ``is_bistable`` -- a DYNAMICAL test that integrates a fully senescent
    tissue forward and asks whether it stays senescent. They agree closely at
    the nominal parameters (0.243 structural vs 0.245 ensemble median), which
    is reassuring but is agreement between two methods, not a replication of
    one. Where they could diverge is near-critical draws with very slow
    transients, which the dynamical test may score as resolving simply because
    the probe horizon is finite.
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
