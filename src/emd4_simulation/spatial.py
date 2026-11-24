"""Spatial stochastic model of secondary senescence on a hexagonal cell lattice.

The model adapts the mechanism of Martin, Schumacher and Chandra (Aging Cell
22:e13892, 2023). A continuous ligand field sums distance-dependent secretion
and contact signalling from immediate neighbours. It approximates the mean of
the discrete binding process and omits binding-count fluctuations. Induction
and clearance are stochastic, and secretion begins after a maturation delay.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

# state codes
HEALTHY, INDUCED, SENESCENT, CLEARED = 0, 1, 2, 3


@dataclass(frozen=True)
class SpatialParams:
    """Model parameters in units of cell diameters and days."""
    nx: int = 61
    ny: int = 61

    # Secretion
    s_primary: float = 1.00      # ligand output, primary senescent cell
    q_sec: float = 0.50          # secondary output RELATIVE to primary
    lam: float = 2.20            # paracrine decay length, cell diameters
    juxta: float = 0.35          # ADDITIONAL induction hazard from direct contact

    # Response
    k_ind: float = 0.055         # induction hazard per unit bound ligand, /day
    theta: float = 0.60          # binding threshold below which nothing happens
    tau_sasp: float = 5.0        # induction -> SASP-producing, days
    gam: float = 0.020           # clearance of senescent cells, /day

    # Exposure
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
    """Identify cells on the outer ring of the hexagonal lattice."""
    i, j = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    return ((i == 0) | (i == nx - 1) | (j == 0) | (j == ny - 1)).ravel()


@lru_cache(maxsize=4)
def _kernel_cached(nx: int, ny: int, lam: float, juxta: float) -> np.ndarray:
    """Return the shared, read-only spatial signalling matrix.

    The exponential paracrine kernel has a six-decay-length cutoff and includes
    a contact term for immediate neighbours. Memory scales with the square of
    the number of cells.
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
    """Simulate stochastic induction and clearance with a fixed delay to maturation."""
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

    # Only a change in secretion alters the ligand field.
    field = np.zeros(n)

    def start_secreting(idx: int, primary: bool) -> None:
        field[:] += K[:, idx] * (p.s_primary if primary else p.s_primary * p.q_sec)

    def stop_secreting(idx: int, primary: bool) -> None:
        field[:] -= K[:, idx] * (p.s_primary if primary else p.s_primary * p.q_sec)

    # Record the first boundary contact even if the lesion later clears.
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
            c = np.cumsum(hz)
            idx = int(np.searchsorted(c, rng.random() * c[-1]))
            state[idx] = INDUCED
            mature_at[idx] = t + p.tau_sasp
        else:                                              # clearance
            sen_idx = np.flatnonzero(state == SENESCENT)
            idx = int(sen_idx[rng.integers(len(sen_idx))])
            stop_secreting(idx, bool(is_primary[idx]))
            # Replacement maintains a pool of susceptible cells after clearance.
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
        # Boundary occupancy at the endpoint.
        "reached_edge": bool((sen & on_boundary).any()),
        # Boundary contact at any time.
        "ever_reached_edge": bool(ever_edge),
    }


def containment_scan(q_values: np.ndarray, n_rep: int = 40,
                     t_end: float = 300.0, base: SpatialParams | None = None) -> dict:
    """Estimate the probability that a seeded lesion reaches the lattice boundary.

    For each secondary secretion ratio, the scan records escape at any time,
    boundary occupancy at the endpoint, extinction and final lesion size.
    The intermediate-size category contains surviving lesions occupying less
    than half the lattice. Mean size and radius also reflect how often lesions
    become extinct or fill the lattice.
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
            e.append(out["ever_reached_edge"])
            o.append(out["reached_edge"])
            if out["ever_reached_edge"]:
                occ.append(out["final_n"] / n_cells)
            if out["final_n"] == 0:
                ext += 1
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
            "occupied_at_end": np.asarray(occupied),
            "n_occupied_at_end": np.asarray(n_occupied),
            "frac_extinct": np.asarray(extinct),
            "size_given_escape": np.asarray(occ_given_escape),
            "frac_intermediate": np.asarray(intermediate),
            "n_cells": int(n_cells)}


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion, including zero or complete escape."""
    if n <= 0:
        return (float("nan"), float("nan"))
    ph = k / n
    d = 1.0 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return (float(max(0.0, c - h)), float(min(1.0, c + h)))


def escape_crossing(q: np.ndarray, escaped: np.ndarray,
                    level: float = 0.5) -> float | None:
    """Interpolate the first upward crossing of level, returning None if none is observed."""
    q = np.asarray(q, dtype=float)
    e = np.asarray(escaped, dtype=float)
    for i in range(1, len(q)):
        if e[i - 1] < level <= e[i]:
            if e[i] == e[i - 1]:
                return float(q[i])
            w = (level - e[i - 1]) / (e[i] - e[i - 1])
            return float(q[i - 1] + w * (q[i] - q[i - 1]))
    return None
