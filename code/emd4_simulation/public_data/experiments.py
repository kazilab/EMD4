"""Explicitly uncalibrated intervention predictions and structural sensitivity."""
from dataclasses import asdict, replace
import numpy as np
from .models import (CANDIDATES, Parameters, Protocol, candidate_parameters,
                     simulate, donor_recipient, observable_drivers)
from .inference import channel_design

SCENARIOS = ('control_medium', 'conditioned_medium', 'washout', 'block', 'rescue',
             'continuous_donor', 'donor_removal', 'carryover_control')


def virtual_transfers():
    results = []
    curves = {}
    for structure in ('matched_clearance', 'lag_memory'):
        for baseline in (0., .02, .1):
            for model in ('persistent_sasp', 'one_way_paracrine', 'feedback'):
                p = candidate_parameters(model)
                for scenario in SCENARIOS:
                    run = donor_recipient(p, scenario=scenario, structure=structure,
                                          initial_arrest=baseline)
                    y, t = run['states'], run['time_days']
                    at3 = int(np.flatnonzero(t == 3)[0])
                    key = f'{model}/{structure}/baseline={baseline}/{scenario}'
                    results.append({'key': key, 'model': model, 'structure': structure,
                                    'baseline_arrest_assumed': baseline, 'scenario': scenario,
                                    'transferred_protein': run['transferred_protein'],
                                    'recipient_secondary_arrest_day3': float(y[at3,3:5].sum()),
                                    'recipient_secondary_arrest_day14': float(y[-1,3:5].sum()),
                                    'recipient_protein_day14': float(y[-1,6]),
                                    'functional_signal_day14': float(p.functional_fraction*y[-1,6]),
                                    'parameters': asdict(p)})
                    if baseline == 0 and structure == 'matched_clearance':
                        curves[key] = {'time_days': t.tolist(), 'states': y.tolist()}
    return {'status': 'uncalibrated_predictions', 'results': results, 'curves': curves,
            'observed_protocol': {'donor_exposure_hours': 2, 'donor_conditioning_hours': 72,
                                  'CM_volume_fraction': .5, 'recipient_exposure_hours': 72},
            'assumptions': ['Equal donor cell number and medium volume; no empirical cell-density scaling.',
                            'Protein abundance and functional fraction are distinct; q is assumed, not fitted.',
                            'Day-14 outcomes, receptor block, rescue, medium washout and continuous donors are virtual extensions.',
                            'The carryover control uses an assumed 1% residual toxicant pulse.',
                            'No migration or tumor-promotion endpoint is inferred from secondary arrest.']}


def design_sensitivity(*, draws=24, seed=9021):
    """Rank future times by expected separation of assumed observable predictions.

    This is a design heuristic, not a validated expected information-gain estimate.
    Summarise over ranges and structures; do not optimise latent S measurement.
    """
    rng = np.random.default_rng(seed)
    times = np.array([1., 2., 4., 7., 14., 28.])
    genes = ['CDKN1A', 'LMNB1', 'IL6', 'GDF15']
    coefficients = np.array([2., 1.3, .4, 7., -1.2, 3., 1.4, 4., .5])
    noise_sd = .15
    separation = []
    for draw in range(draws):
        structure = ('matched_clearance', 'lag_memory')[draw % 2]
        induction = 10**rng.uniform(-.5, .3)
        persistence = 10**rng.uniform(-2.7, -.7)
        gain = 10**rng.uniform(-.3, .6)
        baseline = rng.uniform(0., .1)
        tau = rng.uniform(.5, 4.)
        predictions = []
        for name in CANDIDATES:
            p = replace(candidate_parameters(name, induction=induction, persistence=persistence,
                                              feedback_gain=gain), maturation_days=tau)
            y = simulate(times, p, Protocol(1., medium_changes=(1.,)),
                         structure=structure, initial_arrest=baseline)
            design = channel_design(observable_drivers(y, structure), genes)
            predictions.append(np.einsum('nck,k->nc', design, coefficients))
        # Distance from feedback to its closest nonfeedback competitor. A marker
        # combination matters only if it separates all alternatives in this draw.
        predictions = np.array(predictions)
        delta = (predictions[-1]-predictions[:-1])/noise_sd
        separation.append(np.min(np.mean(delta**2, axis=-1), axis=0))
    scores = np.array(separation)
    return {'seed': seed, 'draws': draws, 'genes': genes, 'assumed_log2_noise_sd': noise_sd,
            'assumed_observation_coefficients': coefficients.tolist(),
            'status': 'assumption_dependent_design_heuristic',
            'times': [{'day_after_induction_start': float(t), 'day_after_withdrawal': float(t-1),
                       'separation_median': float(np.median(scores[:,i])),
                       'separation_lo10': float(np.quantile(scores[:,i], .1)),
                       'separation_hi90': float(np.quantile(scores[:,i], .9))}
                      for i,t in enumerate(times)],
            'limitation': 'Fixed assumed RNA loadings; no nuisance refitting for this heuristic. Actual study design also requires direct arrest/proliferation assays, secreted protein and functional recipient perturbations.'}
