"""Fit measured channels with nuisance loadings and correlated observation errors.

Covariance and observation coefficients use training data only. Late-time samples
are held out together. This is forecasting within a deposited study, not an
independent experiment or proof that a latent biological mechanism is identified.
"""
from dataclasses import asdict
import numpy as np
import pandas as pd
from scipy.linalg import solve_triangular
from scipy.optimize import least_squares
from .adapters import PANELS
from .models import CANDIDATES, Protocol, candidate_parameters, simulate, observable_drivers


def observation_matrix(frame):
    meta = frame[['sample', 'time_days', 'condition', 'replicate']].drop_duplicates().set_index('sample')
    matrix = frame.pivot(index='sample', columns='gene', values='value')
    if matrix.isna().any().any():
        raise ValueError('Incomplete measured panel; do not silently impute missing markers')
    meta = meta.loc[matrix.index]
    return meta, matrix


def noise_covariance(values, groups, shrinkage=0.5, floor=0.05):
    """Pool within-time replicate residuals; shrink off-diagonals for small n."""
    values = np.asarray(values, float)
    if values.ndim != 2 or not 0 <= shrinkage <= 1 or floor <= 0:
        raise ValueError('Invalid covariance arguments')
    groups = np.asarray(groups)
    scatter = np.zeros((values.shape[1], values.shape[1]))
    dof = 0
    for group in np.unique(groups):
        rows = values[groups == group]
        if len(rows) > 1:
            centered = rows-rows.mean(axis=0)
            scatter += centered.T@centered
            dof += len(rows)-1
    if dof == 0:
        raise ValueError('Replicate observations are required to estimate noise')
    cov = scatter/dof
    diag = np.diag(np.maximum(np.diag(cov), floor**2))
    cov = (1-shrinkage)*cov + shrinkage*diag
    # A variance floor is a declared sensitivity assumption, not extra evidence.
    cov += np.diag(np.maximum(floor**2-np.diag(cov), 0))
    return cov


def channel_design(drivers, genes):
    """Prespecified measurement map with fitted intercept and signed loading.

    CDKN1A additionally permits acute damage. No signs are imposed from the
    senescence hypothesis and no gene values are rescaled into fractions.
    """
    n, c = len(drivers), len(genes)
    blocks = []
    for j, gene in enumerate(genes):
        channel = 2 if gene in PANELS['secretory'] else 1 if gene in PANELS['stress'] else 0
        features = [np.ones(n), drivers[:, channel]]
        if gene == 'CDKN1A':
            features.append(drivers[:, 1])
        for feature in features:
            a = np.zeros((n, c))
            a[:, j] = feature
            blocks.append(a)
    return np.stack(blocks, axis=-1)  # n x channels x coefficients


def fit_candidate(frame, name, *, exposure_days, structure='matched_clearance',
                  initial_arrest=0.0, starts=2, seed=4101, max_nfev=100,
                  fixed_gain=None, covariance_shrinkage=0.5, medium_changes=()):
    """Only training observations may be passed here; holdout evaluation is separate."""
    meta, measured = observation_matrix(frame)
    times, values = meta.time_days.to_numpy(), measured.to_numpy()
    genes = measured.columns.tolist()
    covariance = noise_covariance(values, times, shrinkage=covariance_shrinkage)
    inverse_cholesky = solve_triangular(np.linalg.cholesky(covariance), np.eye(len(genes)), lower=True)
    target = (values@inverse_cholesky.T).ravel()
    names = ['induction', 'maturation_days']
    bounds = [(-2.3, 1.3), (-1.0, 1.0)]
    initial = [np.log10(.8), np.log10(1.5)]
    if name != 'reversible':
        names += ['persistence_rate']
        bounds += [(-2.7, 0.0)]
        initial += [np.log10(.03)]
    if name in ('feedback', 'one_way_paracrine') and fixed_gain is None:
        names += ['feedback_gain']
        bounds += [(-1.3, 1.0)]
        initial += [np.log10(2.)]
    lo, hi = np.array(bounds).T

    def decode(theta):
        from dataclasses import replace
        params = dict(zip(names, 10**np.asarray(theta)))
        p = candidate_parameters(name, induction=params['induction'],
                                 persistence=params.get('persistence_rate', .03),
                                 feedback_gain=params.get('feedback_gain', fixed_gain if fixed_gain is not None else 2.))
        return replace(p, maturation_days=params['maturation_days'])

    def evaluate(theta):
        p = decode(theta)
        states = simulate(times, p, Protocol(exposure_days, medium_changes=tuple(medium_changes)), structure=structure,
                          initial_arrest=initial_arrest, rtol=2e-6, atol=2e-8)
        design = channel_design(observable_drivers(states, structure), genes)
        whitened = np.einsum('ij,njk->nik', inverse_cholesky, design).reshape(len(target), -1)
        coefficients, _, rank, singular = np.linalg.lstsq(whitened, target, rcond=1e-9)
        return whitened@coefficients-target, coefficients, p, int(rank), singular

    rng = np.random.default_rng(seed)
    solutions = []
    for start in range(starts):
        x0 = np.array(initial) if start == 0 else rng.uniform(lo, hi)
        solution = least_squares(lambda theta: evaluate(theta)[0], x0, bounds=(lo, hi),
                                 max_nfev=max_nfev, ftol=2e-6, xtol=2e-6, gtol=2e-6,
                                 diff_step=1e-3)
        solutions.append(solution)
    # Converged fits take precedence; failed fits remain explicitly labelled.
    valid = [s for s in solutions if s.success]
    best = min(valid or solutions, key=lambda x: x.cost)
    residual, coefficients, p, rank, singular = evaluate(best.x)
    return {'model': name, 'structure': structure, 'parameters': asdict(p),
            'initial_arrest': initial_arrest, 'exposure_days': float(exposure_days),
            'medium_changes': list(medium_changes),
            'genes': genes, 'coefficients': coefficients.tolist(),
            'covariance': covariance.tolist(), 'covariance_shrinkage': covariance_shrinkage,
            'train_whitened_SSE': float(residual@residual),
            'train_n_observations': int(values.size), 'train_samples': meta.index.tolist(),
            'nonlinear_parameter_count': len(names), 'linear_parameter_count': len(coefficients),
            'linear_rank': rank, 'optimizer_converged': bool(best.success),
            'optimizer_message': str(best.message),
            'boundary_parameters': [n for n, x, a, b in zip(names, best.x, lo, hi)
                                    if min(x-a, b-x) < .015*(b-a)],
            'starts': starts, 'seed': seed, 'fixed_feedback_gain': fixed_gain,
            'kinetic_parameters_identified': False}


def predict(fit, times):
    from .models import Parameters
    states = simulate(times, Parameters(**fit['parameters']),
                      Protocol(fit['exposure_days'], medium_changes=tuple(fit.get('medium_changes',[]))),
                      structure=fit['structure'], initial_arrest=fit['initial_arrest'])
    design = channel_design(observable_drivers(states, fit['structure']), fit['genes'])
    return np.einsum('nck,k->nc', design, fit['coefficients'])


def score_holdout(fit, holdout):
    meta, values = observation_matrix(holdout)
    values = values[fit['genes']]
    prediction = predict(fit, meta.time_days.to_numpy())
    residual = values.to_numpy()-prediction
    w = solve_triangular(np.linalg.cholesky(fit['covariance']), residual.T, lower=True).T
    return {'holdout_whitened_MSE': float(np.mean(w**2)),
            'holdout_RMSE_log2': float(np.sqrt(np.mean(residual**2))),
            'holdout_samples': values.index.tolist(),
            'holdout_predictions': prediction.tolist(), 'holdout_measured': values.to_numpy().tolist()}


def compare_timecourse(frame, *, exposure_days, structures=('matched_clearance','lag_memory'),
                       starts=2, medium_changes=()):
    last_time = float(frame.time_days.max())
    train, holdout = frame[frame.time_days < last_time], frame[frame.time_days == last_time]
    if train.time_days.nunique() < 3:
        raise ValueError('Need at least three training times plus one held-out time')
    fits = []
    for structure in structures:
        for name in CANDIDATES:
            fit = fit_candidate(train, name, exposure_days=exposure_days,
                                structure=structure, starts=starts, medium_changes=medium_changes)
            fit.update(score_holdout(fit, holdout))
            fits.append(fit)
    valid = [f for f in fits if f['optimizer_converged']]
    ranked = sorted(valid, key=lambda f: f['holdout_whitened_MSE'])
    train_meta, measured_train = observation_matrix(train)
    test_meta, measured_test = observation_matrix(holdout)
    covariance = noise_covariance(measured_train.to_numpy(), train_meta.time_days.to_numpy())
    chol = np.linalg.cholesky(covariance)
    last = measured_train[train_meta.time_days == train_meta.time_days.max()].mean().to_numpy()
    linear = np.linalg.lstsq(np.column_stack((np.ones(len(train_meta)), train_meta.time_days)),
                             measured_train.to_numpy(), rcond=None)[0]
    predictions = {'last_observation': np.tile(last, (len(test_meta), 1)),
                   'linear_time': np.column_stack((np.ones(len(test_meta)), test_meta.time_days))@linear}
    baselines = {}
    for name, prediction in predictions.items():
        residual = measured_test.to_numpy()-prediction
        whitened = solve_triangular(chol, residual.T, lower=True).T
        baselines[name] = {'holdout_whitened_MSE': float(np.mean(whitened**2)),
                           'holdout_RMSE_log2': float(np.sqrt(np.mean(residual**2)))}
    return {'holdout_time_days': last_time, 'fits': fits,
            'ranking': [f"{x['model']}:{x['structure']}" for x in ranked],
            'forecast_baselines': baselines,
            'feedback_established': False,
            'interpretation': 'Descriptive within-study late-time prediction. Continuous induction and a flexible RNA observation map cannot establish post-withdrawal persistence, secondary senescence, or bistability.'}


def feedback_profile(train, *, exposure_days, gains=(0., .05, .2, 1., 5., 10.), starts=2):
    fits = [fit_candidate(train, 'feedback', exposure_days=exposure_days,
                          fixed_gain=g, starts=starts) for g in gains]
    best = min(f['train_whitened_SSE'] for f in fits if f['optimizer_converged'])
    return {'grid': [{'gain': g, 'delta_SSE': f['train_whitened_SSE']-best,
                       'converged': f['optimizer_converged'],
                       'boundary_parameters': f['boundary_parameters']} for g, f in zip(gains, fits)],
            'interval': None,
            'interpretation': 'Finite-grid sensitivity with nuisance kinetics and observation coefficients refitted. This is not a confidence interval; covariance is estimated and the mechanism is structurally confounded.'}


def endpoint_contrasts(frame):
    """Descriptive independent-group Welch intervals; do not infer paired samples."""
    from scipy.stats import t
    rows = []
    for treated, control in [('direct_DOX', 'control'), ('DOX_CM', 'control_CM')]:
        for gene in sorted(frame.gene.unique()):
            x = frame[(frame.condition == treated)&(frame.gene == gene)].value.to_numpy()
            y = frame[(frame.condition == control)&(frame.gene == gene)].value.to_numpy()
            vx, vy = np.var(x, ddof=1)/len(x), np.var(y, ddof=1)/len(y)
            se = np.sqrt(vx+vy)
            df = (vx+vy)**2/(vx**2/(len(x)-1)+vy**2/(len(y)-1)) if se else np.inf
            margin = t.ppf(.975, df)*se if se else None
            rows.append({'contrast': f'{treated}_minus_{control}', 'gene': gene,
                         'difference_log2_TPM_plus_1': float(x.mean()-y.mean()),
                         'lo95': float(x.mean()-y.mean()-margin) if margin is not None else None,
                         'hi95': float(x.mean()-y.mean()+margin) if margin is not None else None,
                         'n_treated': len(x), 'n_control': len(y), 'multiplicity_adjusted': False,
                         'interval_status': 'descriptive_Welch' if se else 'unavailable_zero_observed_variance',
                         'all_values_zero': bool(np.all(x == 0) and np.all(y == 0))})
    return rows


def wilson(k, n):
    if n <= 0:
        return [None, None]
    z = 1.95996398454
    d = 1+z*z/n
    c = (k/n+z*z/(2*n))/d
    h = z*np.sqrt((k/n*(1-k/n)+z*z/(4*n))/n)/d
    return [float(c-h), float(c+h)]


def synthetic_benchmark(*, trials=2, seed=723, starts=1):
    """Cross-generator discrimination; no own-truth success passed off as validation."""
    rng = np.random.default_rng(seed)
    times = np.array([0., .5, 1., 2., 4., 7., 14.])
    genes = ['CDKN1A', 'LMNB1', 'IL6', 'GDF15']
    # Fixed observation layer for generation, freely refitted for inference.
    coefficients = np.array([2., 1.3, .4, 7., -1.2, 3., 1.4, 4., .5])
    covariance = .08**2*(.5*np.eye(4)+.5*np.ones((4,4)))
    confusion = {truth: {name: 0 for name in CANDIDATES} for truth in CANDIDATES}
    records = []
    for truth in CANDIDATES:
        for trial in range(trials):
            # Vary rates, baseline and maturation structure to avoid testing only
            # one cherry-picked nominal truth. These are assumed ranges, not priors
            # estimated from the human datasets.
            from dataclasses import replace
            p = candidate_parameters(truth, induction=rng.uniform(.5, 1.2),
                                     persistence=10**rng.uniform(-2.4, -.7),
                                     feedback_gain=rng.uniform(.5, 3.))
            p = replace(p, maturation_days=rng.uniform(.5, 3.))
            structure = ('matched_clearance', 'lag_memory')[trial % 2]
            initial_arrest = rng.uniform(0., .05)
            states = simulate(times, p, Protocol(1.), structure=structure, initial_arrest=initial_arrest)
            mean = np.einsum('nck,k->nc', channel_design(observable_drivers(states, structure), genes), coefficients)
            rows = []
            for rep in range(3):
                random_intercept = rng.normal(0, .035, len(genes))
                y = mean+rng.multivariate_normal(np.zeros(4), covariance, len(times))+random_intercept
                for i, time in enumerate(times):
                    for j, gene in enumerate(genes):
                        rows.append(dict(sample=f's{rep}_{i}', condition='synthetic', replicate=rep,
                                         time_days=time, gene=gene, value=y[i,j]))
            frame = pd.DataFrame(rows)
            result = compare_timecourse(frame, exposure_days=1.,
                                        structures=('matched_clearance',), starts=starts)
            valid = [f for f in result['fits'] if f['optimizer_converged']]
            winner = min(valid, key=lambda f: f['holdout_whitened_MSE'])['model'] if valid else None
            if winner:
                confusion[truth][winner] += 1
            records.append({'truth': truth, 'trial': trial, 'winner': winner,
                            'true_structure': structure, 'true_initial_arrest': initial_arrest,
                            'true_parameters': asdict(p),
                            'candidate_scores': {f['model']: f['holdout_whitened_MSE'] for f in result['fits']},
                            'converged': {f['model']: f['optimizer_converged'] for f in result['fits']}})
    negatives = [r for r in records if r['truth'] != 'feedback' and r['winner'] is not None]
    false = sum(r['winner'] == 'feedback' for r in negatives)
    return {'trials_per_truth': trials, 'seed': seed, 'confusion': confusion, 'records': records,
            'false_feedback_selections': false, 'nonfeedback_trials_with_a_winner': len(negatives),
            'false_selection_fraction': false/len(negatives) if negatives else None,
            'wilson95': wilson(false, len(negatives)),
            'interpretation': 'Synthetic model-selection stress test, conditional on declared generators/noise/ranges. Descriptive winner is not a positive biological claim. Monte Carlo interval is not an empirical false-positive rate.'}
