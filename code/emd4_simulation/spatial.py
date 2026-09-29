"""Tier 3: spatial stochastic model of SASP-mediated secondary senescence.

Reimplements the mechanism described by Martin, Schumacher & Chandra
(Aging Cell 22:e13892, 2023; code at github.com/lkmartin90/Senescence_Spread):
ligand release from senescent cells, distance-dependent signal accumulation at
neighbouring cells, Gillespie-simulated induction events, and a delay between
induction and the acquisition of a SASP-producing state.

**Where this departs from the source.** Martin et al. model discrete ligand
binding with Poisson-distributed counts per cell. This implementation carries a
CONTINUOUS ligand field ``L_i`` (an exponential kernel plus a juxtacrine term
at contact range) and converts it to an induction hazard through a soft
threshold, ``h_i = k_ind * max(L_i - theta, 0)``. Only the induction and
clearance EVENTS are stochastic, drawn by the Gillespie algorithm; the binding
step is deterministic in the mean. This is a mean-field approximation of the
binding layer, adequate for the containment question V10 asks -- whether the
front stops -- but it does not reproduce binding-count fluctuations, so single
lesions here are less variable at low ligand than the source model would make
them. An earlier version of this docstring described the binding as
"Poisson-approximated", which the code has never done.

Written from the published description rather than vendored, so the repository
stays self-contained and the assumptions are visible in one place. The check
that matters is `V10`: before any exposure is added, the model must reproduce
the paper's central result -- that propagation is FINITE, because secondary
senescent cells secrete less than primary ones. A model that predicts runaway
senescence in normal tissue has not earned the right to say anything about
exposure.

Spatial layout is a 2-D hexagonal lattice: six equidistant nearest neighbours,
which makes the juxtacrine term unambiguous in a way a square lattice's
four-versus-eight ambiguity does not.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

# state codes
HEALTHY, INDUCED, SENESCENT, CLEARED = 0, 1, 2, 3


@dataclass(frozen=True)
class SpatialParams:
    """Lengths are in cell diameters; times are in days."""
    nx: int = 61
    ny: int = 61

    # --- secretion --------------------------------------------------------
    s_primary: float = 1.00      # ligand output, primary senescent cell
    q_sec: float = 0.50          # secondary output RELATIVE to primary
    lam: float = 2.20            # paracrine decay length, cell diameters
    juxta: float = 0.35          # ADDITIONAL induction hazard from direct contact

    # --- response ---------------------------------------------------------
    k_ind: float = 0.055         # induction hazard per unit bound ligand, /day
    theta: float = 0.60          # binding threshold below which nothing happens
    tau_sasp: float = 5.0        # induction -> SASP-producing, days
    gam: float = 0.020           # clearance of senescent cells, /day

    # --- exposure ---------------------------------------------------------
    k_direct: float = 0.0        # exposure-driven direct induction hazard, /day
    t_exposure: float = 0.0      # exposure applied on [0, t_exposure)
    replace_cleared: bool = True # cleared cells are replaced by healthy ones


def _hex_coords(nx: int, ny: int) -> np.ndarray:
    """Axial-offset hex lattice embedded in the plane."""
    i, j = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    x = i + 0.5 * (j % 2)
    y = j * np.sqrt(3.0) / 2.0
    return np.stack([x.ravel(), y.ravel()], axis=1)


def _boundary_mask(nx: int, ny: int) -> np.ndarray:
    """Cells on the outer ring of the lattice.

    "Reaching the edge" is membership of this ring, not a radius threshold. The
    hex embedding is anisotropic -- row spacing is sqrt(3)/2 of the column
    spacing -- so the lattice is TALLER in index than in distance, and a radius
    criterion of ``0.45 * min(nx, ny)`` (as this module used previously) sits
    at 18.45 on a 41x41 lattice whose half-height is only 17.32. A lesion that
    ran to the top and bottom edges therefore scored as contained. Testing ring
    membership directly removes the geometry from the definition.
    """
    i, j = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    return ((i == 0) | (i == nx - 1) | (j == 0) | (j == ny - 1)).ravel()


@lru_cache(maxsize=4)
def _kernel_cached(nx: int, ny: int, lam: float, juxta: float) -> np.ndarray:
    """Exponential paracrine kernel plus a juxtacrine bonus at contact range.

    Precomputed once as a dense N x N matrix. At 61x61 that is 3721^2 floats
    (~110 MB), which is the practical ceiling for this approach -- larger
    lattices need a cutoff or an FFT convolution, and the exponential kernel
    makes a cutoff at ~6*lam essentially exact.

    Cached on the four parameters it actually depends on. It does NOT depend on
    q_sec, which is what a containment scan varies, so rebuilding it per
    replicate made the scan's cost scale with the replicate count for no
    reason -- and that cost was the argument for running only five replicates,
    which is how the scan's band edges came to be sampling accidents.

    Callers must treat the result as READ-ONLY; it is shared between runs.
    ``simulate_spatial`` only ever reads columns of it.
    """
    coords = _hex_coords(nx, ny)
    d = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
    K = np.exp(-d / lam)
    K[d > 6.0 * lam] = 0.0                  # kernel support cutoff
    np.fill_diagonal(K, 0.0)                # a cell does not induce itself
    K += juxta * ((d > 0.0) & (d <= 1.01))  # six nearest neighbours
    K.setflags(write=False)
    return K


def _kernel(coords: np.ndarray, p: SpatialParams) -> np.ndarray:
    return _kernel_cached(p.nx, p.ny, p.lam, p.juxta)


def simulate_spatial(p: SpatialParams, t_end: float = 300.0,
                     seed: int = 0, seed_cells: int = 1,
                     record_every: float = 5.0) -> dict:
    """Gillespie simulation of senescence spread from a seeded lesion.

    Events are induction (healthy -> induced), maturation (induced ->
    senescent, i.e. SASP-producing) and clearance. Induction hazards are
    recomputed from the current ligand field after every event, which is exact
    for this system -- the field changes only when a cell changes state.
    """
    rng = np.random.default_rng(seed)
    coords = _hex_coords(p.nx, p.ny)
    n = coords.shape[0]
    K = _kernel(coords, p)
    on_boundary = _boundary_mask(p.nx, p.ny)

    state = np.zeros(n, dtype=np.int8)
    is_primary = np.zeros(n, dtype=bool)

    # seed a lesion at the lattice centre
    centre = np.argmin(np.linalg.norm(coords - coords.mean(axis=0), axis=1))
    order = np.argsort(np.linalg.norm(coords - coords[centre], axis=1))

    # The ligand field is maintained INCREMENTALLY: it changes only when a cell
    # starts or stops secreting, and then only by that cell's kernel column.
    # Rebuilding it from scratch each event is an O(n * n_secreting) inner loop
    # and made a single run take ~107 s; this is O(n).
    field = np.zeros(n)

    def start_secreting(idx: int, primary: bool) -> None:
        field[:] += K[:, idx] * (p.s_primary if primary else p.s_primary * p.q_sec)

    def stop_secreting(idx: int, primary: bool) -> None:
        field[:] -= K[:, idx] * (p.s_primary if primary else p.s_primary * p.q_sec)

    # EVER-reached, accumulated as it happens. `reached_edge` below is a
    # snapshot of the FINAL state, so a lesion that touched the boundary and
    # was then cleared scores as contained on it. "Propagation reached the
    # edge by day 300" and "the edge is occupied on day 300" are different
    # outcomes and the escape claim is the first one.
    ever_edge = False

    for idx in order[:seed_cells]:
        state[idx] = SENESCENT
        is_primary[idx] = True
        ever_edge = ever_edge or bool(on_boundary[idx])
        start_secreting(int(idx), True)

    mature_at = np.full(n, np.inf)
    t = 0.0
    trace_t, trace_n, trace_r = [], [], []
    next_record = 0.0
    origin = coords[centre]

    while t < t_end:
        healthy = state == HEALTHY
        drive = np.maximum(field - p.theta, 0.0)
        hz = p.k_ind * drive * healthy
        if t < p.t_exposure:
            hz = hz + p.k_direct * healthy

        h_ind = float(hz.sum())
        n_sen = int((state == SENESCENT).sum())
        h_clr = float(p.gam * n_sen)
        h_tot = h_ind + h_clr

        # the next maturation is deterministic, so race it against the
        # stochastic events rather than folding it into the hazard
        t_mat = float(mature_at.min())
        dt = rng.exponential(1.0 / h_tot) if h_tot > 0 else np.inf
        t_next = min(t + dt, t_mat)
        if not np.isfinite(t_next):
            break

        while next_record <= min(t_next, t_end):
            sen = state == SENESCENT
            trace_t.append(next_record)
            trace_n.append(int(sen.sum()))
            trace_r.append(float(np.linalg.norm(coords[sen] - origin, axis=1).max())
                           if sen.any() else 0.0)
            next_record += record_every

        t = t_next
        if t >= t_end:
            break

        if t == t_mat:                                     # maturation
            idx = int(np.argmin(mature_at))
            state[idx] = SENESCENT
            mature_at[idx] = np.inf
            ever_edge = ever_edge or bool(on_boundary[idx])
            start_secreting(idx, bool(is_primary[idx]))
        elif rng.random() < h_ind / h_tot:                 # induction
            # cumsum + searchsorted rather than rng.choice(n, p=...), which
            # normalises and validates a length-n vector on every event
            c = np.cumsum(hz)
            idx = int(np.searchsorted(c, rng.random() * c[-1]))
            state[idx] = INDUCED
            mature_at[idx] = t + p.tau_sasp
        else:                                              # clearance
            sen_idx = np.flatnonzero(state == SENESCENT)
            idx = int(sen_idx[rng.integers(len(sen_idx))])
            stop_secreting(idx, bool(is_primary[idx]))
            # Replacement matters: without it the lattice depletes and every
            # lesion "resolves" for the trivial reason that the tissue is gone.
            state[idx] = HEALTHY if p.replace_cleared else CLEARED
            is_primary[idx] = False

    sen = state == SENESCENT
    return {
        "t": np.asarray(trace_t),
        "n_senescent": np.asarray(trace_n),
        "max_radius": np.asarray(trace_r),
        "final_state": state,
        "coords": coords,
        "final_n": int(sen.sum()),
        "final_radius": float(np.linalg.norm(coords[sen] - origin, axis=1).max())
                        if sen.any() else 0.0,
        # Edge OCCUPANCY at t_end. Kept for continuity of the older outputs.
        "reached_edge": bool((sen & on_boundary).any()),
        # Edge ever reached at any time. This is the escape outcome.
        "ever_reached_edge": bool(ever_edge),
    }


def containment_scan(q_values: np.ndarray, n_rep: int = 40,
                     t_end: float = 300.0, base: SpatialParams | None = None) -> dict:
    """Escape PROBABILITY as a function of secondary secretion.

    The spatial counterpart of `bifurcation.containment_threshold`. If the two
    agree on where propagation stops, the mean-field tipping point and the
    spatial front are the same phenomenon seen twice -- which is the claim
    `V10` makes and the reason this module exists.

    What this scan does and does not measure
    ----------------------------------------
    At this lattice size the outcome is very nearly BINARY: a seeded lesion
    either dies out (0 cells, radius 0) or grows to the mean-field upper branch
    and fills the lattice to its corner (~1450/1681 cells, radius 26.46).
    Intermediate, genuinely *bounded* lesions occur in only a few per cent of
    replicates. The consequence is that ``size`` and ``radius`` are NOT
    independent readouts -- they are the saturation value multiplied by the
    escape probability, and quoting all three as separate columns reports one
    number three times.

    So the primary output here is ``escaped`` (an estimated probability, with
    ``n_rep`` and ``n_escaped`` so a binomial interval can be formed), and the
    size/radius columns are retained only as diagnostics. ``frac_intermediate``
    reports how often a bounded lesion is seen at all; while it stays near zero
    this lattice cannot speak to finite propagation *extent*, only to
    extinction probability. ``size_given_escape`` is the one size number that
    means something -- the occupied fraction of surviving lesions, which is
    comparable against the mean-field upper branch.
    """
    base = base or SpatialParams()
    from dataclasses import replace

    sizes, radii, escaped = [], [], []
    n_escaped, occ_given_escape, intermediate = [], [], []
    occupied, n_occupied = [], []
    extinct = []
    n_cells = base.nx * base.ny
    for q in q_values:
        s, r, e, occ, inter = [], [], [], [], 0
        o, ext = [], 0
        for rep in range(n_rep):
            out = simulate_spatial(replace(base, q_sec=float(q)),
                                   t_end=t_end, seed=1000 + rep)
            s.append(out["final_n"])
            r.append(out["final_radius"])
            # ESCAPE = the front reached the boundary at ANY time. The former
            # criterion was `reached_edge`, an end-of-run snapshot, which
            # scores a lesion that touched the edge and was then cleared as
            # contained. Both are reported; they are different outcomes.
            e.append(out["ever_reached_edge"])
            o.append(out["reached_edge"])
            if out["ever_reached_edge"]:
                occ.append(out["final_n"] / n_cells)
            # THREE outcomes, not two: extinction, a bounded surviving lesion,
            # and escape. Collapsing the first two into "contained" hides which
            # containment mechanism is operating.
            if out["final_n"] == 0:
                ext += 1
            # "Intermediate" = survived without saturating: a genuinely bounded
            # lesion, which is the outcome this tier was built to observe.
            if 0 < out["final_n"] < 0.5 * n_cells:
                inter += 1
        sizes.append(float(np.mean(s)))
        radii.append(float(np.mean(r)))
        escaped.append(float(np.mean(e)))
        n_escaped.append(int(np.sum(e)))
        occupied.append(float(np.mean(o)))
        n_occupied.append(int(np.sum(o)))
        extinct.append(ext / n_rep)
        occ_given_escape.append(float(np.mean(occ)) if occ else float("nan"))
        intermediate.append(inter / n_rep)
    return {"q": np.asarray(q_values), "size": np.asarray(sizes),
            "radius": np.asarray(radii), "escaped": np.asarray(escaped),
            "n_rep": int(n_rep),
            "n_escaped": np.asarray(n_escaped),
            # End-of-run edge OCCUPANCY, the older and weaker criterion.
            "occupied_at_end": np.asarray(occupied),
            "n_occupied_at_end": np.asarray(n_occupied),
            "frac_extinct": np.asarray(extinct),
            "size_given_escape": np.asarray(occ_given_escape),
            "frac_intermediate": np.asarray(intermediate),
            "n_cells": int(n_cells)}


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Used rather than the normal approximation because the scan's proportions
    sit at 0 and 1 at the ends of the range, where the normal interval has zero
    width and would report a five-replicate accident as a certainty.
    """
    if n <= 0:
        return (float("nan"), float("nan"))
    ph = k / n
    d = 1.0 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return (float(max(0.0, c - h)), float(min(1.0, c + h)))


def escape_crossing(q: np.ndarray, escaped: np.ndarray,
                    level: float = 0.5) -> float | None:
    """Interpolated q at which the escape probability first crosses ``level``.

    Returns None when the scan never crosses, which is a real answer -- a
    containment scan in which nothing escapes has no crossing. Reading the
    crossing off the grid instead (``q[argmax(escaped > level)]``) both snaps
    the answer to the grid spacing and, because ``argmax`` of an all-False
    array is 0, silently reports the SMALLEST q tested when there is no
    crossing at all.
    """
    q = np.asarray(q, dtype=float)
    e = np.asarray(escaped, dtype=float)
    for i in range(1, len(q)):
        if e[i - 1] < level <= e[i]:
            if e[i] == e[i - 1]:
                return float(q[i])
            w = (level - e[i - 1]) / (e[i] - e[i - 1])
            return float(q[i - 1] + w * (q[i] - q[i - 1]))
    return None
