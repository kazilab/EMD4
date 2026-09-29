"""Competing mechanisms in physical days; parameters remain assumptions unless fitted.

No fitted RNA value is a cell fraction. The observation layer has separately fitted
intercepts and loadings. Arrest fraction is a latent, closed-population occupancy
(no birth/death competition); it is not the original model's physiological S.
"""
from dataclasses import dataclass, replace
import numpy as np
from scipy.integrate import solve_ivp

CANDIDATES = ('reversible', 'slow_damage', 'durable_arrest', 'persistent_sasp',
              'one_way_paracrine', 'feedback')
STATE_NAMES = ('damage', 'primary_immature', 'primary_mature',
               'secondary_immature', 'secondary_mature', 'secretory_memory', 'protein',
               'secondary_secretory_memory')

@dataclass(frozen=True)
class Parameters:
    induction: float = 0.8
    repair: float = 1.0
    arrest_loss: float = 0.03
    maturation_days: float = 1.5
    secretion: float = 1.0
    protein_clearance: float = 1.0
    feedback_gain: float = 2.0
    functional_fraction: float = 0.5
    response_half: float = 0.25
    secondary_secretion_fraction: float = 1.0


def candidate_parameters(name, *, induction=0.8, persistence=0.03, feedback_gain=2.0):
    if name not in CANDIDATES:
        raise ValueError(f'Unknown mechanism: {name}')
    p = Parameters(induction=induction, feedback_gain=0.0)
    if name == 'reversible':
        return replace(p, arrest_loss=1.0)
    if name == 'slow_damage':
        return replace(p, repair=persistence, arrest_loss=1.0)
    if name == 'durable_arrest':
        return replace(p, arrest_loss=persistence, secretion=0.0)
    if name == 'persistent_sasp':
        return replace(p, arrest_loss=persistence)
    if name == 'one_way_paracrine':
        return replace(p, arrest_loss=persistence, feedback_gain=feedback_gain,
                       secondary_secretion_fraction=0.0)
    return replace(p, arrest_loss=persistence, feedback_gain=feedback_gain)


@dataclass(frozen=True)
class Protocol:
    exposure_days: float
    exposure_level: float = 1.0
    # Times are absolute days after this culture's start. Zero all extracellular
    # protein at each medium replacement; intracellular damage remains.
    medium_changes: tuple[float, ...] = ()
    block_at: float = np.inf
    block_fraction: float = 0.0
    rescue_at: float = np.inf
    rescue_protein: float = 0.0
    donor_source: float = 0.0
    donor_remove_at: float = np.inf


def simulate(times, p=Parameters(), protocol=Protocol(1.0), *,
             structure='matched_clearance', initial_arrest=0.0, initial_protein=0.0,
             rtol=2e-7, atol=1e-9):
    """Split integration at every pulse/medium-change event; return post-event states."""
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or len(times) == 0 or np.any(~np.isfinite(times)) or np.any(times < 0):
        raise ValueError('Times must be a nonempty finite nonnegative vector')
    if structure not in ('matched_clearance', 'lag_memory'):
        raise ValueError('Unknown maturation structure')
    rates = (p.induction, p.repair, p.arrest_loss, p.secretion,
             p.protein_clearance, p.feedback_gain, p.functional_fraction)
    if (not np.isfinite((*rates, p.maturation_days, p.response_half)).all()
            or min(rates) < 0 or p.maturation_days <= 0 or p.response_half <= 0):
        raise ValueError('Rates must be nonnegative and scales positive')
    if not 0 <= p.functional_fraction <= 1:
        raise ValueError('Functional fraction must be between zero and one')
    if not 0 <= p.secondary_secretion_fraction <= 1:
        raise ValueError('Secondary secretion fraction must be between zero and one')
    if not (0 <= initial_arrest <= 1) or initial_protein < 0:
        raise ValueError('Invalid initial occupancy or protein')
    if not 0 <= protocol.block_fraction <= 1:
        raise ValueError('Block fraction must be between zero and one')
    if not np.isfinite((protocol.exposure_level, protocol.donor_source, protocol.rescue_protein)).all():
        raise ValueError('Protocol amounts must be finite')
    if np.isnan((protocol.exposure_days, protocol.block_at, protocol.rescue_at,
                 protocol.donor_remove_at, *protocol.medium_changes)).any():
        raise ValueError('Protocol times cannot be NaN')
    if min(protocol.exposure_days, protocol.exposure_level, protocol.donor_source,
           protocol.rescue_protein, protocol.block_at, protocol.rescue_at,
           protocol.donor_remove_at, *protocol.medium_changes) < 0:
        raise ValueError('Protocol amounts and times must be nonnegative')
    y = np.zeros(8)
    y[2] = initial_arrest
    y[5] = initial_arrest
    y[6] = initial_protein
    end = float(times.max())
    events = sorted({0., end, *[float(x) for x in
                    (protocol.exposure_days, protocol.block_at, protocol.rescue_at,
                     protocol.donor_remove_at, *protocol.medium_changes) if 0 <= x <= end]})
    result = np.empty((len(times), 8))

    def jump(t, state):
        state = state.copy()
        if t in protocol.medium_changes:
            state[6] = 0.0
        if t == protocol.rescue_at:
            state[6] += protocol.rescue_protein
        return state

    y = jump(0., y)
    result[times == 0.] = y
    for left, right in zip(events[:-1], events[1:]):
        midpoint = (left+right)/2
        exposure = protocol.exposure_level if midpoint < protocol.exposure_days else 0.
        gain = p.feedback_gain * (1-protocol.block_fraction if midpoint >= protocol.block_at else 1.)
        donor = protocol.donor_source if midpoint < protocol.donor_remove_at else 0.

        def rhs(t, z):
            d, i1, m1, i2, m2, c, protein, c2 = z
            arrest = i1+m1+i2+m2
            mature = m1+m2
            available = max(0., 1-arrest)
            primary = p.induction*max(d, 0)/(1+max(d, 0))*available
            signal = p.functional_fraction*max(protein, 0)
            secondary = gain*signal**2/(p.response_half**2+signal**2)*available
            k = 1/p.maturation_days
            producer = (m1+p.secondary_secretion_fraction*m2 if structure == 'matched_clearance'
                        else c-(1-p.secondary_secretion_fraction)*c2)
            return (exposure-p.repair*d,
                    primary-(k+p.arrest_loss)*i1, k*i1-p.arrest_loss*m1,
                    secondary-(k+p.arrest_loss)*i2, k*i2-p.arrest_loss*m2,
                    (arrest-c)*k,
                    p.secretion*max(producer, 0)-p.protein_clearance*protein+donor,
                    (i2+m2-c2)*k)

        sol = solve_ivp(rhs, (left, right), y, rtol=rtol, atol=atol, dense_output=True)
        if not sol.success:
            raise RuntimeError(sol.message)
        mask = (times > left) & (times < right)
        if mask.any():
            result[mask] = sol.sol(times[mask]).T
        y = jump(right, sol.y[:, -1])
        result[times == right] = y
    if np.min(result) < -2e-6 or np.max(result[:, 1:5].sum(axis=1)) > 1+2e-6:
        raise RuntimeError('Integration left the physical state domain')
    return result


def observable_drivers(states, structure='matched_clearance'):
    """Arrest, damage and secretory drivers; not marker-calibrated measurements."""
    y = np.asarray(states)
    secretory = y[:, 2]+y[:, 4] if structure == 'matched_clearance' else y[:, 5]
    return np.column_stack((y[:, 1:5].sum(axis=1), y[:, 0], secretory))


def donor_recipient(p=Parameters(), *, scenario='conditioned_medium',
                    structure='matched_clearance', initial_arrest=0.0):
    """GSE210140 timing; virtual interventions extend its unperturbed protocol.

    Equal donor cell number and medium volume assumed, not measured/calibrated.
    Transfer includes only extracellular protein. No donor cells or exposure are
    carried over except in the explicit residual-toxicant control.
    """
    scenarios = ('conditioned_medium', 'control_medium', 'washout', 'block',
                 'rescue', 'continuous_donor', 'donor_removal', 'carryover_control')
    if scenario not in scenarios:
        raise ValueError(f'Unknown transfer scenario: {scenario}')
    donor_times = np.linspace(0, 3+2/24, 150)
    donor = simulate(donor_times, p, Protocol(2/24, medium_changes=(2/24,)),
                     structure=structure, initial_arrest=initial_arrest)
    transferred = 0.5*donor[-1, 6]
    if scenario in ('control_medium', 'carryover_control'):
        # Matched untreated donors can have background arrest/secretion.
        control = simulate(donor_times, p, Protocol(0), structure=structure,
                           initial_arrest=initial_arrest)
        transferred = 0.5*control[-1, 6]
    protocol = Protocol(0)
    if scenario == 'washout':
        protocol = replace(protocol, medium_changes=(3.,))
    if scenario == 'block':
        protocol = replace(protocol, block_at=0., block_fraction=1.)
    if scenario == 'rescue':
        # Partial receptor inhibition, then active ligand add-back. Complete
        # receptor blockade cannot be rescued by more ligand in this model.
        protocol = replace(protocol, block_at=0., block_fraction=0.5,
                           rescue_at=3., rescue_protein=0.5)
    if scenario in ('continuous_donor', 'donor_removal'):
        protocol = replace(protocol, donor_source=transferred,
                           donor_remove_at=3. if scenario == 'donor_removal' else np.inf)
    if scenario == 'carryover_control':
        protocol = Protocol(2/24, exposure_level=0.01)  # assumed 1% residual pulse
    times = np.unique(np.r_[np.linspace(0, 14, 141), 3.])
    recipient = simulate(times, p, protocol, structure=structure,
                         initial_arrest=initial_arrest, initial_protein=transferred)
    return {'time_days': times, 'states': recipient, 'donor_states': donor,
            'donor_time_days': donor_times, 'transferred_protein': float(transferred),
            'protocol': protocol}
