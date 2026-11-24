"""Analyse public withdrawal RNA, arsenite assay figures, secretomes and single-cell measurements."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .sources import ROOT, DEFAULT_DATA, sha256
from .extension_sources import fetch_extensions
from .counts import washout_expression, independent_expression
from .secretome import secretome_tables
from .single_cell import analyze_single_cell
from .arsenite import analyze_arsenite
from .validation import external_direction_checks
from .inference import compare_timecourse, predict, fit_candidate, score_holdout
from .run import run, save_json


def complete_plots(out, paths, digitized, arsenite, washout, comparisons, protein_panel, cells, single_cell, external):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    for fig in (2,5,6):
        source=plt.imread(paths[f'arsenite_fig{fig}'])
        picture,ax=plt.subplots(figsize=(8,9))
        ax.imshow(source)
        chosen=digitized[digitized.figure==fig]
        ax.scatter(chosen.x_pixel,chosen.mean_y_pixel,s=75,facecolors='none',edgecolors='#df3838',linewidths=1)
        for _,row in chosen.iterrows():
            ax.plot([row.x_pixel-4,row.x_pixel+4],[row.mean_y_pixel]*2,color='#df3838',lw=.7)
        ax.set_axis_off();ax.set_title(f'Figure {fig}. Digitized group means (red circles)\nReading sensitivity ±2 pixels')
        picture.tight_layout();picture.savefig(out/f'arsenite_digitization_fig{fig}.png',dpi=140);plt.close(picture)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    values=arsenite['SA_beta_gal_control_adjusted']
    for key,label in [('control_percent','Control'),('treated_percent','Arsenite'),('excess_percentage_points','Treated − control')]:
        axes[0].plot([x['post_withdrawal_hours']/24 for x in values],[x[key] for x in values],'-o',label=label)
    axes[0].set(xlabel='Days after withdrawal',ylabel='SA-β-gal positivity / percentage points',title='Published figure estimates')
    axes[0].legend(fontsize=8)
    lum=arsenite['luminescence'];t=np.array(lum['time_since_4_hour_reference_days'])
    axes[1].scatter(t,lum['observed_log_treated_over_control'],c='black',label='Digitized means')
    for name,f in lum['fits'].items():
        times=np.linspace(0,3,100);r=f['apparent_recovery_per_day'];a=f['inhibition_per_day']
        prediction=-a*(times if r<1e-8 else -np.expm1(-r*times)/r)
        axes[1].plot(times,prediction,label=name.replace('_',' '))
    axes[1].set(xlabel='Assay days after 4-hour normalization',ylabel='log(treated / control luminescence)',title='Post-withdrawal luminescence')
    axes[1].legend(fontsize=7);fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(out/f'arsenite_public_evidence.{ext}',dpi=160)
    plt.close(fig)
    for condition,comparison in comparisons.items():
        valid=[f for f in comparison['fits'] if f['optimizer_converged']]
        if not valid:continue
        best=min((f for f in valid if f['model']!='feedback'),key=lambda f:f['holdout_whitened_MSE'])
        feedback=min((f for f in valid if f['model']=='feedback'),key=lambda f:f['holdout_whitened_MSE'])
        subset=washout[washout.condition.isin([condition,'control'])]
        times=np.linspace(0,17,200)
        a,b=predict(best,times),predict(feedback,times)
        fig,axes=plt.subplots(3,3,figsize=(11,8),sharex=True)
        for j,gene in enumerate(best['genes']):
            ax=axes.flat[j];f=subset[subset.gene==gene]
            ax.scatter(f.time_days,f.value,s=13,c='black')
            ax.plot(times,a[:,j],color='#227b8e',label=best['model'])
            ax.plot(times,b[:,j],'--',color='#bd5534',label='feedback')
            ax.axvline(1,color='gray',lw=.8);ax.axvspan(9.05,17.1,color='#ddd',alpha=.4)
            ax.set_title(gene)
            if j>=6:ax.set_xlabel('Days after induction start')
            if j%3==0:ax.set_ylabel('log2(normalized counts + 1)')
        axes.flat[0].legend(fontsize=7)
        fig.suptitle(f'GSE222400 {condition} after a 24-hour pulse\nDay 17 held out (16 days after withdrawal)')
        fig.tight_layout()
        for ext in ('png','pdf'):fig.savefig(out/f'withdrawal_{condition}.{ext}',dpi=150)
        plt.close(fig)
    allowed=['CXCL8','MMP3','SERPINE1','GDF15','IL6']
    table=protein_panel[protein_panel.gene.isin(allowed)].pivot(index='gene',columns='sheet',values='log2_senescent_over_control').reindex(allowed)
    cmap=plt.get_cmap('RdBu_r').copy();cmap.set_bad('#d9d9d9')
    limit=float(np.nanmax(abs(table.to_numpy())))
    fig,ax=plt.subplots(figsize=(11,4));im=ax.imshow(np.ma.masked_invalid(table.to_numpy()),aspect='auto',cmap=cmap,vmin=-limit,vmax=limit)
    ax.set_xticks(range(len(table.columns)),table.columns,rotation=35,ha='right');ax.set_yticks(range(len(table)),table.index)
    ax.set_title('Public secretome measurements vary by context and time\nGray indicates missing measurements and white indicates a near-zero log ratio')
    fig.colorbar(im,ax=ax,label='Author log2(senescent / control)');fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(out/f'public_secretome.{ext}',dpi=160)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    rna=cells[cells.modality=='RNA']
    for state,group in rna.groupby('library'):
        group=group.set_index('feature').sort_index()
        axes[0].plot(range(len(group)),group.nonzero_fraction,'o-',label=state)
    axes[0].set_xticks(range(len(group)),group.index,rotation=40,ha='right');axes[0].set(ylabel='Fraction of retained cells with RNA detected',title='Single-cell marker sparsity');axes[0].legend(fontsize=8)
    pairs=pd.DataFrame(single_cell['paired_RNA_ADT'])
    for state,group in pairs.groupby('library'):
        axes[1].plot(range(len(group)),group.spearman_rho,'o-',label=state)
    axes[1].set_xticks(range(len(group)),group.ADT,rotation=40,ha='right');axes[1].set(ylabel='Within-library Spearman association',title='Paired surface ADT and RNA');axes[1].legend(fontsize=8)
    fig.suptitle('CITE-seq measurements in one biological library per condition')
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(out/f'public_single_cell.{ext}',dpi=160)
    plt.close(fig)
    contrasts=external.pivot(index='gene',columns=['cell','agent'],values='difference_log2_normalized_counts_plus_1')
    limit=float(np.nanmax(abs(contrasts.to_numpy())))
    fig,ax=plt.subplots(figsize=(8,5));im=ax.imshow(contrasts.to_numpy(),cmap='RdBu_r',vmin=-limit,vmax=limit,aspect='auto')
    ax.set_yticks(range(len(contrasts)),contrasts.index);ax.set_xticks(range(len(contrasts.columns)),[' / '.join(c) for c in contrasts.columns],rotation=25,ha='right')
    ax.set_title('RNA response contrasts in GSE235768\nH₂O₂ includes 72 hours of recovery')
    fig.colorbar(im,ax=ax,label='Difference in log2(normalized counts + 1)');fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(out/f'independent_RNA.{ext}',dpi=160)
    plt.close(fig)


def complete_report(out, data):
    lines = [
        "# EMD4 public-data results",
        "",
        "## RNA after withdrawal",
        "",
        "The NCBI sample-count matrix contains 52 WI-38 samples. Each pulse arm has "
        "16 samples and shares two untreated controls. Two replicative-senescence "
        "samples lack a comparable time origin and are excluded. The models use a "
        "one-day induction with medium replacement on days 1, 5, 9, 13 and 17. Both "
        "samples on day 17 (16 days after withdrawal) are held out from normalization "
        "reference estimation, kinetic fitting and error covariance estimation. "
        "The submitter archive provides differential-expression statistics. The "
        "sample-level comparisons use NCBI HISAT2/featureCounts data and median-of-ratios "
        "normalization. Untreated controls were sampled only at baseline. "
        "[GSE222400](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE222400) · "
        "[NCBI processing](https://www.ncbi.nlm.nih.gov/geo/info/rnaseqcounts.html)",
        "",
        "| Pulse | Best model and structure | Whitened MSE | Feedback MSE | Last-observation MSE |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    better_baseline = []
    for condition, comparison in data["withdrawal_comparisons"].items():
        valid = [f for f in comparison["fits"] if f["optimizer_converged"]]
        if not valid:
            lines.append(f"| {condition} | No converged fit | — | — | — |")
            continue
        best = min(valid, key=lambda f: f["holdout_whitened_MSE"])
        feedback = min((f for f in valid if f["model"] == "feedback"),
                       key=lambda f: f["holdout_whitened_MSE"])
        last = comparison["forecast_baselines"]["last_observation"]["holdout_whitened_MSE"]
        if last < feedback["holdout_whitened_MSE"]:
            better_baseline.append(condition)
        lines.append(f"| {condition} | {best['model']} / {best['structure']} | "
                     f"{best['holdout_whitened_MSE']:.2f} | {feedback['holdout_whitened_MSE']:.2f} | {last:.2f} |")
    if better_baseline:
        lines.extend(["", "Carrying forward the last observation gave lower error than the feedback "
                      f"model for {', '.join(better_baseline)}."])
    benchmark = json.loads((out / "summary.json").read_text())["synthetic_benchmark"]
    lines.extend([
        "",
        "Smaller mean squared error (MSE) indicates closer prediction of held-out relative "
        "expression. Ties between mechanisms, parameter-bound hits and the small number "
        "of held-out samples limit mechanistic interpretation.",
        "",
        "![Withdrawal RNA](withdrawal_DXR.png)",
        "",
        f"In the synthetic benchmark, feedback gave the lowest held-out error in "
        f"{benchmark['false_feedback_selections']} of {benchmark['nonfeedback_trials_with_a_winner']} "
        "nonfeedback trials with a converged winner. The RNA time-course, conditioned-medium "
        "and benchmark results are in the [RNA and simulation report](report.md).",
        "",
        "## Arsenite responses in Huh-7 cells",
        "",
    ])
    first, last = data["arsenite"]["SA_beta_gal_control_adjusted"]
    lines.extend([
        f"Digitized SA-β-gal positivity in treated cells was {first['treated_percent']:.1f}% "
        f"at withdrawal and {last['treated_percent']:.1f}% seven days later. Controls changed "
        f"from {first['control_percent']:.1f}% to {last['control_percent']:.1f}%, giving "
        f"treated–control differences of {first['excess_percentage_points']:.1f} and "
        f"{last['excess_percentage_points']:.1f} percentage points. The study reports five "
        "microscopy fields. The ±2-pixel digitization range measures reading uncertainty. "
        "[Okamura et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC11701098/) · "
        "[Digitized values and coordinates](arsenite_digitized.csv)",
        "",
        "After reseeding, luminescence reflects metabolism, viability and growth relative "
        "to controls. Its apparent relaxation rate cannot identify senescent-cell clearance. "
        "Exposure and withdrawal PCR use different reference genes (18S and RPLP1).",
        "",
        "![Arsenite responses](arsenite_public_evidence.png)",
        "",
        "## Secreted proteins and single-cell markers",
        "",
        f"The SASP Atlas contains {data['secretome']['reported_protein_rows']:,} reported "
        f"protein comparisons across eight contexts, including {data['secretome']['panel_rows']} "
        "rows for the marker panel. Values retain source cell type, induction time, soluble "
        "or vesicular fraction, q-value and reported standard deviation. Missing protein "
        "measurements are shown in gray. [SASP Atlas](https://doi.org/10.1371/journal.pbio.3000599) · "
        "[Marker panel](secretome_panel.csv)",
        "",
        "![Secreted protein abundance](public_secretome.png)",
        "",
        "Paired RNA and eight surface-antibody measurements describe within-library "
        "heterogeneity in proliferating and irradiated WI-38 cells. Quality filtering "
        "requires at least 500 detected RNA features and at most 20% mitochondrial RNA. "
        "RNA and antibody counts are normalized separately. Each condition has a single "
        "biological library. [GSE250041](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE250041)",
        "",
        "| Library | Deposited cells | Retained cells |",
        "| --- | ---: | ---: |",
    ])
    for library in data["single_cell"]["libraries"]:
        lines.append(f"| {library['library']} | {library['cells_before_QC']} | {library['cells_after_QC']} |")
    lines.extend([
        "",
        "![Single-cell RNA and surface markers](public_single_cell.png)",
        "",
        "## Response directions across studies",
        "",
        "Predictions fitted to GSE144397 training samples were compared with BJ and HFL1 "
        "fibroblast RNA and with eligible soluble secretome measurements. The independent "
        "RNA study contains 24 samples with three samples per group. H₂O₂ arms include "
        "72 hours of recovery, and bortezomib treatment is continuous. The table gives "
        "the range of agreeing marker directions across the converged source models. "
        "[GSE235768](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE235768)",
        "",
        "| Target | Agreeing directions | Compared markers |",
        "| --- | ---: | ---: |",
    ])
    checks = pd.DataFrame(data["external_validation"]["checks"])
    for target, group in checks.groupby("target"):
        counts = sorted(set(group.informative_genes))
        lines.append(f"| {target} | {group.sign_agreement.min()}–{group.sign_agreement.max()} | "
                     f"{', '.join(map(str, counts))} |")
    lines.extend([
        "",
        "![Independent RNA contrasts](independent_RNA.png)",
        "",
        "These assays measure RNA expression, secreted protein abundance and surface "
        "markers in different cell systems. Functional SASP potency, sustained feedback, "
        "bistability and the biological secretion threshold remain unresolved.",
        "",
        "[Numerical results](complete_summary.json) · [Input records](complete_source_manifest.json) · "
        "[Run settings and hashes](complete_runtime.json)",
        "",
    ])
    (out / "complete_report.md").write_text("\n".join(lines))
    old_draft = out / "manuscript_claims.md"
    if old_draft.is_file() and sha256(old_draft) == "d09981787a814a3886443b8c0de7657368ece40834a4efbadc13337297754a63":
        old_draft.unlink()


def run_complete(*,data_dir=DEFAULT_DATA,outdir=None,offline=False,trials=25,starts=3):
    out=Path(outdir) if outdir else ROOT.parents[1]/'outputs'/'public_data'
    base=run(data_dir=data_dir,outdir=out,offline=offline,trials=trials,starts=starts)
    paths=fetch_extensions(data_dir,offline=offline)
    print('Reading withdrawal and independent-study counts',flush=True)
    washout,count_audit=washout_expression(paths)
    independent,independent_audit=independent_expression(paths)
    washout.to_csv(out/'withdrawal_observations.csv',index=False)
    independent.to_csv(out/'independent_observations.csv',index=False)
    comparisons={}
    for condition in ('SDS','KCl','DXR'):
        print(f'Fitting withdrawal series: {condition}',flush=True)
        subset=washout[washout.condition.isin([condition,'control'])].copy()
        comparisons[condition]=compare_timecourse(subset,exposure_days=1.,starts=starts,
                                                  medium_changes=(1.,5.,9.,13.,17.))
        comparisons[condition]['interpretation']='Within-study prediction of held-out withdrawal counts.'
    print('Importing published arsenite and secretome measurements',flush=True)
    digitized,arsenite=analyze_arsenite(paths)
    digitized.to_csv(out/'arsenite_digitized.csv',index=False)
    proteins,panel,secretome=secretome_tables(paths)
    proteins.to_csv(out/'secretome_comparisons.csv',index=False)
    panel.to_csv(out/'secretome_panel.csv',index=False)
    cells,single_cell=analyze_single_cell(paths)
    cells.to_csv(out/'single_cell_descriptive.csv',index=False)
    external,validation=external_direction_checks(base['timecourse_comparisons']['RASOIS'],independent,panel)
    external.to_csv(out/'independent_contrasts.csv',index=False)
    data={'schema_version':2,'status':'public_data_scope_complete_with_biological_evidence_limits',
          'EMD4_established':False,'feedback_established':False,'q_sec_critical_calibrated':False,
          'withdrawal_count_audit':count_audit,'independent_count_audit':independent_audit,
          'withdrawal_comparisons':comparisons,'arsenite':arsenite,'secretome':secretome,
          'single_cell':single_cell,'external_validation':validation,
          'base_results':'summary.json','raw_alignment_needed':False,
          'quantitative_functional_calibration':'Not identifiable from the imported assays; no such calibration is claimed.',
          'raw_alignment_alternative':'Validated NCBI-generated public sample counts recovered GSE222400.'}
    save_json(out/'complete_summary.json',data)
    (out/'complete_source_manifest.json').write_text(Path(data_dir,'complete_manifest.json').read_text())
    runtime=json.loads((out/'runtime.json').read_text())
    runtime['source_code_sha256']={p.name:sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
    runtime['input_sha256']={key:sha256(path) for key,path in paths.items()}
    runtime['entrypoint']='python3 -m emd4_simulation.public_data complete'
    import PIL
    runtime['pillow']=PIL.__version__
    save_json(out/'complete_runtime.json',runtime)
    complete_plots(out,paths,digitized,arsenite,washout,comparisons,panel,cells,single_cell,external)
    complete_report(out,data)
    print(f'Completed public-data analysis: {out / "complete_report.md"}',flush=True)
    return data
