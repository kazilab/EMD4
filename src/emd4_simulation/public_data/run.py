"""Compare RNA time courses, conditioned medium responses and simulated mechanisms."""
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
    fig.suptitle('GSE144397 RNA trajectories under continuous RAS induction\nDay 6 held out')
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
    fig.suptitle('GSE210140 endpoint RNA contrasts, n=5 per group\nUnadjusted 95% Welch intervals')
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
    ax.set_title('Synthetic model selection by generating mechanism')
    fig.colorbar(im,ax=ax,label='Trials');fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(out/f'model_confusion.{ext}',dpi=160)
    plt.close(fig)


def report(out, summary):
    comparisons = summary["timecourse_comparisons"]
    lines = [
        "# RNA time courses and conditioned medium",
        "",
        "The models predict expression of nine predefined genes in 36 samples from three "
        "continuous-induction or quiescence time courses. Both final-time replicates are "
        "held out from fitting kinetics, marker coefficients and error covariance. The "
        "deposited expression matrix was normalized across the source collection, so "
        "prediction is assessed within study. [GSE144397](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE144397)",
        "",
        "| Context | Held-out day | Best model and structure | Whitened MSE | Feedback MSE | Last-observation MSE |",
        "| --- | ---: | --- | ---: | ---: | ---: |",
    ]
    for condition, comparison in comparisons.items():
        valid = [f for f in comparison["fits"] if f["optimizer_converged"]]
        best = min(valid, key=lambda f: f["holdout_whitened_MSE"])
        feedback = min((f for f in valid if f["model"] == "feedback"),
                       key=lambda f: f["holdout_whitened_MSE"])
        last = comparison["forecast_baselines"]["last_observation"]["holdout_whitened_MSE"]
        lines.append(f"| {condition} | {comparison['holdout_time_days']:g} | "
                     f"{best['model']} / {best['structure']} | {best['holdout_whitened_MSE']:.3f} | "
                     f"{feedback['holdout_whitened_MSE']:.3f} | {last:.3f} |")
    lines.extend([
        "",
        "Smaller mean squared error (MSE) indicates closer prediction. Durable arrest and "
        "persistent secretion without feedback give identical predicted RNA in this observation "
        "model. The gain profile refits nuisance kinetics and marker coefficients at each gain. "
        "Separate fits test sensitivity to baseline occupancy, maturation and noise assumptions. "
        "[Model fits, bounds and forecast baselines](summary.json)",
        "",
        "![Observed RNA fits](rna_timecourse.png)",
        "",
        "## Conditioned-medium RNA",
        "",
        "Each of four groups has five samples. The contrasts compare direct doxorubicin with "
        "its control and 72 hours of 50% conditioned medium from treated donors with control "
        "conditioned medium. The figure shows differences in log2(TPM + 1) with descriptive "
        "95% Welch intervals. The intervals use independent groups and have no multiplicity "
        "adjustment. A cross marks zero observed variance, for which no interval is estimated. "
        "[GSE210140](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE210140) · "
        "[Gene-level contrasts](endpoint_contrasts.csv)",
        "",
        "![Conditioned-medium RNA](conditioned_medium_RNA.png)",
        "",
        "## Simulated donor–recipient interventions",
        "",
        "The donor model uses a two-hour exposure, a wash and 72 hours of conditioning. "
        "Only extracellular protein transfers, with residual exposure modelled as a separate "
        "control. The 14-day recipient responses are predictions under assigned rates and "
        "equal donor cell number and medium volume. One-way paracrine signalling can sustain "
        "recipient arrest without secretion by secondary cells. The model compares matched "
        "cell and secretion clearance with a secretion delay that persists after arrest clears.",
        "",
        "![Simulated recipient responses](virtual_transfer.png)",
        "",
        "## Synthetic model selection",
        "",
    ])
    benchmark = summary["synthetic_benchmark"]
    lo, hi = benchmark["wilson95"]
    lines.extend([
        f"The analysis used {benchmark['trials_per_truth']} trials per generating mechanism "
        f"and seed {benchmark['seed']}. Feedback gave the lowest held-out error in "
        f"{benchmark['false_feedback_selections']} of {benchmark['nonfeedback_trials_with_a_winner']} "
        f"nonfeedback trials with a converged winner (95% Wilson interval {lo:.3f}–{hi:.3f}). "
        "The interval describes simulation sampling under the specified generators and noise model.",
        "",
        "![Synthetic model selection](model_confusion.png)",
        "",
    ])
    (out / "report.md").write_text("\n".join(lines))


def run(*, data_dir=DEFAULT_DATA, outdir=None, offline=False, trials=10, starts=3):
    if trials < 1 or starts < 1:
        raise ValueError('trials and starts must be positive')
    out=Path(outdir) if outdir else ROOT.parents[1]/'outputs'/'public_data'
    publication=ROOT.parents[1]/'outputs'/'reproduced'
    if out.resolve() == publication.resolve() or publication.resolve() in out.resolve().parents:
        raise ValueError('Use a separate empirical output directory, not outputs/reproduced')
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
