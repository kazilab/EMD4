"""Reproducible empirical workflow, isolated from manuscript/nominal outputs."""
from __future__ import annotations
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import sys
import numpy as np
import pandas as pd
import scipy
from .sources import DEFAULT_DATA, ROOT, fetch_all, sha256
from .adapters import timecourse, ihh, audit_washout_archive
from .inference import (compare_timecourse, endpoint_contrasts, feedback_profile,
                        synthetic_benchmark, fit_candidate, score_holdout, predict)
from .experiments import virtual_transfers, design_sensitivity


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def plots(out, observations, comparisons, contrasts, transfers, benchmark):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9, 'axes.spines.top':False, 'axes.spines.right':False})
    ras = observations[observations.condition == 'RASOIS']
    comparison = comparisons['RASOIS']
    valid = [f for f in comparison['fits'] if f['optimizer_converged']]
    best_null = min((f for f in valid if f['model'] != 'feedback'), key=lambda x:x['holdout_whitened_MSE'])
    best_feedback = min((f for f in valid if f['model'] == 'feedback'), key=lambda x:x['holdout_whitened_MSE'])
    times = np.linspace(0, ras.time_days.max(), 150)
    null_prediction, feedback_prediction = predict(best_null, times), predict(best_feedback, times)
    fig, axes = plt.subplots(3, 3, figsize=(11,8), sharex=True)
    for j, gene in enumerate(best_null['genes']):
        ax = axes.flat[j]
        frame = ras[ras.gene == gene]
        ax.scatter(frame.time_days, frame.value, color='black', s=14, label='Public RNA observations')
        ax.plot(times, null_prediction[:,j], color='#227b8e', label=best_null['model'])
        ax.plot(times, feedback_prediction[:,j], color='#bd5534', linestyle='--', label='feedback')
        ax.axvspan(4.05, 6.05, color='#dddddd', alpha=.4)
        ax.set_title(gene)
        if j >= 6: ax.set_xlabel('Days from continuous RAS induction')
        if j % 3 == 0: ax.set_ylabel('Normalised log2 expression')
    axes.flat[0].legend(fontsize=7)
    fig.suptitle('GSE144397: observed RNA fits; day 6 held out\nLate-time prediction does not establish post-withdrawal persistence')
    fig.tight_layout()
    for ext in ('png','pdf'): fig.savefig(out/f'rna_timecourse.{ext}', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1,2,figsize=(11,4.5),sharey=True)
    contrast_names = list(dict.fromkeys(x['contrast'] for x in contrasts))
    for ax, name in zip(axes, contrast_names):
        rows = [x for x in contrasts if x['contrast']==name]
        mean = np.array([x['difference_log2_TPM_plus_1'] for x in rows])
        for j, (row, m) in enumerate(zip(rows, mean)):
            if row['lo95'] is None:
                ax.plot(m, j, marker='x', color='gray')
                ax.annotate('zero observed variance', (m,j), xytext=(8,0),
                            textcoords='offset points', fontsize=6, va='center')
            else:
                errors = np.array([[m-row['lo95']], [row['hi95']-m]])
                ax.errorbar([m], [j], xerr=errors, fmt='o', color='#227b8e', capsize=3)
        ax.set_yticks(range(len(rows)), [x['gene'] for x in rows]);ax.axvline(0,color='gray',lw=1)
        ax.set_title(name.replace('_',' '));ax.set_xlabel('Difference in log2(TPM + 1)')
    fig.suptitle('GSE210140: endpoint RNA contrasts, n=5 per group\nDescriptive 95% Welch intervals; no multiplicity correction')
    fig.tight_layout()
    for ext in ('png','pdf'): fig.savefig(out/f'conditioned_medium_RNA.{ext}',dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1,2,figsize=(11,4))
    for key, curve in transfers['curves'].items():
        if not key.startswith('feedback/'):
            continue
        scenario = key.split('/')[-1]
        y=np.array(curve['states']); t=curve['time_days']
        axes[0].plot(t,y[:,3:5].sum(axis=1),label=scenario.replace('_',' '))
        axes[1].plot(t,y[:,6])
    axes[0].set_ylabel('Latent secondary arrest occupancy')
    axes[1].set_ylabel('Extracellular protein (model units)')
    for ax in axes:ax.set_xlabel('Days after medium transfer')
    axes[0].legend(fontsize=6)
    fig.suptitle('Virtual donor–recipient interventions — uncalibrated assumptions')
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(out/f'virtual_transfer.{ext}',dpi=160)
    plt.close(fig)
    names=list(benchmark['confusion'])
    matrix=np.array([[benchmark['confusion'][t][p] for p in names] for t in names])
    fig,ax=plt.subplots(figsize=(7,5))
    im=ax.imshow(matrix,cmap='Blues',vmin=0)
    for i in range(len(names)):
        for j in range(len(names)):ax.text(j,i,str(matrix[i,j]),ha='center',va='center')
    ax.set_xticks(range(len(names)),names,rotation=35,ha='right');ax.set_yticks(range(len(names)),names)
    ax.set_ylabel('Generating mechanism');ax.set_xlabel('Lowest held-out prediction error')
    ax.set_title('Synthetic model confusion; conditional on assumed generators')
    fig.colorbar(im,ax=ax,label='Trials');fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(out/f'model_confusion.{ext}',dpi=160)
    plt.close(fig)


def report(out, summary):
    comparisons=summary['timecourse_comparisons']
    lines=['# EMD4 public-data analysis', '',
           '**Conclusion: this run does not establish EMD4, functional SASP feedback, or bistability.** It adds measured RNA checks, persistent alternatives, held-out predictions and testable intervention predictions. The original simulation and manuscript outputs are unchanged.', '',
           '## Public-data quality and scope', '',
           '- [GSE144397](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE144397): 36 samples, nine predefined genes, analysed separately as RAS, RAF and quiescence trajectories. Submitter-normalised log2 expression is retained. Induction continues throughout observation; there is no withdrawal experiment in these selected arms.',
           '- [GSE210140](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE210140): 20 samples, nine genes, log2(TPM+1). Direct DOX is compared with its control; recipient DOX-conditioned medium is compared with control-conditioned medium. Five samples per group; replicate numbers are not assumed to establish pairing.',
           f"- [GSE222400](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE222400): **excluded**. All {summary['washout_archive_audit']['DE_table_count']} deposited files contain differential-expression tables rather than individual-sample expression, with {summary['washout_archive_audit']['unique_payloads']} distinct uncompressed payloads. `baseMean` is never treated as a sample measurement. Reanalysis of raw reads or a corrected deposit is needed for this withdrawal series.",
           '- [Okamura arsenite study](https://pmc.ncbi.nlm.nih.gov/articles/PMC11701098/): full article and figures cached with provenance. Huh-7 exposure is 72 hours; withdrawal and reseeding are separate protocol clocks. Figures are retained for review; no numerical values or individual replicates have been invented from their images.', '',
           '## Held-out RNA prediction', '',
           'Each culture context is fitted separately. Both final-time replicates are excluded from all kinetic fitting, observation coefficients and noise covariance estimation. The deposited RMA/batch normalisation predates this split and uses the source collection; this is within-study forecasting, not independent external validation. Replicate trajectories can be correlated across time, so no independent-sample hypothesis test is claimed.', '',
           '| Context | Held-out day | Lowest predictive error | Whitened MSE | Feedback MSE | Last-observation MSE |',
           '|---|---:|---|---:|---:|---:|']
    for condition,c in comparisons.items():
        valid=[f for f in c['fits'] if f['optimizer_converged']]
        best=min(valid,key=lambda x:x['holdout_whitened_MSE'])
        feedback=min((f for f in valid if f['model']=='feedback'),key=lambda x:x['holdout_whitened_MSE'])
        lines.append(f"| {condition} | {c['holdout_time_days']:g} | {best['model']} / {best['structure']} | {best['holdout_whitened_MSE']:.3f} | {feedback['holdout_whitened_MSE']:.3f} | {c['forecast_baselines']['last_observation']['holdout_whitened_MSE']:.3f} |")
    lines += ['', 'Smaller error means better prediction of these selected RNA channels. It does not identify a mechanism. Last-observation and linear-time forecasts provide simple training-only benchmarks (both in JSON). Large errors relative to estimated replicate variability and parameter-bound hits are evidence of poor prediction or weak identification, even when a candidate ranks first. Durable arrest and persistent SASP without feedback are **observationally equivalent in this RNA observation model**, because secreted protein is not measured. Rankings between tied candidates are arbitrary. Failed optimisations are excluded from ranking and retained with status in JSON. Parameter bounds, observation rank and all fits are retained.', '',
              'The feedback profile refits nuisance kinetics and observation coefficients at each gain. It is a finite-grid sensitivity analysis, not a confidence interval. Baseline occupancy (0, 0.02, 0.10), maturation/clearance structure and diagonal versus correlated observation noise are also examined. No RNA-derived estimate of `q_sec_critical` is produced: functional fraction and receptor-response scale are confounded.', '',
              '![Observed RNA fits](rna_timecourse.png)', '',
              '## Conditioned-medium evidence', '',
              'The measured result is a transcriptomic contrast after 72 hours of 50% conditioned medium. Endpoint RNA alone does not show recipient senescence, self-sustaining recipient secretion, SASP protein potency, or exclusion of residual toxicant. Intervals are descriptive Welch 95% intervals, without multiplicity correction; no significance screening or post hoc marker selection is used. A channel with zero observed variance receives no interval: all-zero TPM is not evidence of a precisely zero biological effect.', '',
              '![Conditioned-medium RNA](conditioned_medium_RNA.png)', '',
              '## Virtual interventions and structural checks', '',
              'The donor model uses the public 2-hour exposure, wash and 72-hour conditioning schedule. Only extracellular protein transfers. Exposure carryover is an explicit separate control. Recipient simulations include medium washout, receptor inhibition, partial-block ligand add-back, continuous donor input and donor removal. A one-way paracrine alternative induces recipient arrest while secondary cells cannot secrete active factors; persistence of that arrest does not imply a feedback loop. The latter extensions and 14-day outcomes are predictions, with assumed rates and equal donor cell number/medium volume. Complete receptor inhibition cannot be rescued by more ligand in this model.', '',
              'Two secretion structures are compared: a phenomenological lag that retains memory after arrest clearance, and an immature/mature model in which secreting cells undergo the same clearance as arrested cells. Baseline and parameter sweeps expose assumption dependence. Protein abundance and functional activity remain separate.', '',
              '![Virtual interventions](virtual_transfer.png)', '',
              '## Model-confusion benchmark', '']
    b=summary['synthetic_benchmark']
    lines += [f"{b['trials_per_truth']} synthetic trials per generating mechanism, seed {b['seed']}. Feedback was selected in {b['false_feedback_selections']} of {b['nonfeedback_trials_with_a_winner']} nonfeedback trials with a converged winner. The conditional Monte Carlo Wilson interval is {b['wilson95'][0]:.3f}–{b['wilson95'][1]:.3f}. This is not an empirical false-positive rate or the probability that EMD4 is true.", '',
              'Generating conditions vary persistence, induction, baseline and maturation structure. Every candidate fits noisy observables and re-estimates its measurement coefficients. An unseen final time assesses prediction. Misspecified structure and replicate-level offsets are included. Sparse trials and assumed generators limit interpretation; increase `--trials` for precision. No classifier cutoff is chosen using this benchmark.', '',
              '![Model confusion](model_confusion.png)', '',
              '## What this supports for the manuscript', '',
              '- Supported as a modelling capability: persistent nonfeedback mechanisms are compared, public measured RNA is used, protocols are explicit, and feedback is challenged with held-out data and interventions.',
              '- Supported by the imported data: the reported gene-level trajectories and conditioned-medium contrasts, in their own cell systems and protocols.',
              '- Still unestablished: EMD4 in arsenite-exposed Huh-7 with a quantitatively calibrated functional SASP, necessity/sufficiency of feedback, bistability, tumor promotion, and a biological lower bound for `q_sec_critical`.',
              '- Most informative next validation: the same donor/recipient system with proliferation plus multiple senescence assays, secreted protein, toxicant carryover controls, recipient washout, pathway inhibition and an appropriate rescue. The design-time ranking in JSON is conditional on assumed observable loadings, not a power calculation.', '',
              '## Reproducibility', '',
              'See `summary.json`, `observations.csv`, `endpoint_contrasts.csv`, `source_manifest.json`, `washout_archive_audit.json` and `runtime.json`. Every downloaded input is checksum-verified; modified or untracked cache files fail closed. Source code hashes, package versions, seeds and fitted sample IDs are recorded. `--offline` reproduces analysis without accessing the network. Original manuscript tables do not consume these new files.', '']
    (out/'report.md').write_text('\n'.join(lines))


def run(*, data_dir=DEFAULT_DATA, outdir=None, offline=False, trials=10, starts=3):
    if trials < 1 or starts < 1:
        raise ValueError('trials and starts must be positive')
    out=Path(outdir) if outdir else ROOT/'public_data_results'
    # Never write empirical outputs into the existing publication directory.
    publication=ROOT.parent/'figures'/'output'
    if out.resolve() == publication.resolve() or publication.resolve() in out.resolve().parents:
        raise ValueError('Use a separate empirical output directory, not figures/output')
    out.mkdir(parents=True,exist_ok=True)
    paths=fetch_all(data_dir,offline=offline)
    frame, annotation=timecourse(paths)
    hepatocytes, ihh_audit=ihh(paths,annotation['ensembl_to_symbol'])
    observations=pd.concat([frame,hepatocytes],ignore_index=True)
    observations.to_csv(out/'observations.csv',index=False)
    archive_audit=audit_washout_archive(paths['timecourse_expression'])
    save_json(out/'washout_archive_audit.json',archive_audit)
    comparisons={}
    for condition in ('RASOIS','QUIESCENCE','RAFOIS'):
        print(f'Fitting observed RNA: {condition}',flush=True)
        subset=frame[frame.condition==condition]
        comparisons[condition]=compare_timecourse(subset,exposure_days=float(subset.time_days.max()),starts=starts)
    ras=frame[frame.condition=='RASOIS']
    train,holdout=ras[ras.time_days<6],ras[ras.time_days==6]
    print('Profiling feedback and checking baseline/noise sensitivity',flush=True)
    profile=feedback_profile(train,exposure_days=6,starts=starts)
    sensitivity=[]
    for baseline in (0.,.02,.1):
        for shrinkage in (.5,1.):
            for model in ('persistent_sasp','feedback'):
                f=fit_candidate(train,model,exposure_days=6,initial_arrest=baseline,
                                covariance_shrinkage=shrinkage,starts=starts)
                f.update(score_holdout(f,holdout));sensitivity.append(f)
    contrasts=endpoint_contrasts(hepatocytes)
    pd.DataFrame(contrasts).to_csv(out/'endpoint_contrasts.csv',index=False)
    print('Simulating donor–recipient interventions and design sensitivity',flush=True)
    transfers=virtual_transfers()
    design=design_sensitivity()
    print(f'Cross-generator benchmark: {trials} trials per mechanism',flush=True)
    benchmark=synthetic_benchmark(trials=trials,starts=starts)
    summary={'schema_version':1,'status':'empirical_workflow_complete_with_evidence_gaps',
             'EMD4_established':False,'feedback_established':False,
             'q_sec_critical_calibrated':False,'timecourse_annotation':annotation,
             'ihh_audit':ihh_audit,'washout_archive_audit':archive_audit,
             'timecourse_comparisons':comparisons,'feedback_profile':profile,
             'baseline_and_noise_sensitivity':sensitivity,
             'endpoint_contrasts':contrasts,'virtual_transfers':transfers,
             'design_sensitivity':design,'synthetic_benchmark':benchmark}
    save_json(out/'summary.json',summary)
    manifest=Path(data_dir,'manifest.json').read_text()
    (out/'source_manifest.json').write_text(manifest)
    runtime={'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,
             'scipy':scipy.__version__,'pandas':pd.__version__,
             'source_code_sha256':{p.name:sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
             'trials':trials,'starts':starts,'data_dir':str(Path(data_dir).resolve())}
    save_json(out/'runtime.json',runtime)
    plots(out,frame,comparisons,contrasts,transfers,benchmark)
    report(out,summary)
    print(f'Wrote {out / "report.md"}',flush=True)
    return summary
