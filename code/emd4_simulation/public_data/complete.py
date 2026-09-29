"""Complete public-data-only analysis: source evidence, recovery series and validation."""
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
    # Coordinate audit: source unchanged, manually read points marked in a plot.
    for fig in (2,5,6):
        source=plt.imread(paths[f'arsenite_fig{fig}'])
        picture,ax=plt.subplots(figsize=(8,9))
        ax.imshow(source)
        chosen=digitized[digitized.figure==fig]
        ax.scatter(chosen.x_pixel,chosen.mean_y_pixel,s=75,facecolors='none',edgecolors='#df3838',linewidths=1)
        for _,row in chosen.iterrows():
            ax.plot([row.x_pixel-4,row.x_pixel+4],[row.mean_y_pixel]*2,color='#df3838',lw=.7)
        ax.set_axis_off();ax.set_title(f'Figure {fig}: recorded mean coordinates (red circles)\nSource image retained; ±2-pixel reading sensitivity')
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
    axes[1].set(xlabel='Assay days after 4-hour normalization',ylabel='log(treated / control luminescence)',title='Descriptive assay fits; no mechanistic identification')
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
        fig.suptitle(f'GSE222400 / {condition}: recovered NCBI count matrix\n24-hour pulse; day 17 held out (16 days after withdrawal)')
        fig.tight_layout()
        for ext in ('png','pdf'):fig.savefig(out/f'withdrawal_{condition}.{ext}',dpi=150)
        plt.close(fig)
    allowed=['CXCL8','MMP3','SERPINE1','GDF15','IL6']
    table=protein_panel[protein_panel.gene.isin(allowed)].pivot(index='gene',columns='sheet',values='log2_senescent_over_control').reindex(allowed)
    cmap=plt.get_cmap('RdBu_r').copy();cmap.set_bad('#d9d9d9')
    limit=float(np.nanmax(abs(table.to_numpy())))
    fig,ax=plt.subplots(figsize=(11,4));im=ax.imshow(np.ma.masked_invalid(table.to_numpy()),aspect='auto',cmap=cmap,vmin=-limit,vmax=limit)
    ax.set_xticks(range(len(table.columns)),table.columns,rotation=35,ha='right');ax.set_yticks(range(len(table)),table.index)
    ax.set_title('Public secretome measurements vary by context and time\nGray = no imported measurement; white = near-zero log ratio')
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
    fig.suptitle('Descriptive CITE-seq evidence: one biological library per condition')
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(out/f'public_single_cell.{ext}',dpi=160)
    plt.close(fig)
    contrasts=external.pivot(index='gene',columns=['cell','agent'],values='difference_log2_normalized_counts_plus_1')
    limit=float(np.nanmax(abs(contrasts.to_numpy())))
    fig,ax=plt.subplots(figsize=(8,5));im=ax.imshow(contrasts.to_numpy(),cmap='RdBu_r',vmin=-limit,vmax=limit,aspect='auto')
    ax.set_yticks(range(len(contrasts)),contrasts.index);ax.set_xticks(range(len(contrasts.columns)),[' / '.join(c) for c in contrasts.columns],rotation=25,ha='right')
    ax.set_title('Independent public study: GSE235768\nRNA response contrasts; H₂O₂ includes 72-hour recovery')
    fig.colorbar(im,ax=ax,label='Difference in log2(normalized counts + 1)');fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(out/f'independent_RNA.{ext}',dpi=160)
    plt.close(fig)


def complete_report(out, data):
    lines=['# EMD4: completed public-data version','',
           'The public-data implementation is complete for the sources and analyses specified below. **It does not establish a self-sustaining feedback loop, bistability or a biological value of `q_sec_critical`.** Direct functional calibration remains an evidence gap, not a parameter inferred from RNA or protein abundance.','',
           '## What changed in this version','',
           '- Recovered GSE222400 using NCBI-generated sample counts from the public sequencing reads. The original 52 submitter tables remain excluded because they contain differential-expression statistics. No local raw-read pipeline was needed.',
           '- Added quantitative, traceable digitization of matching-system arsenite/Huh-7 figures, with correct assay clocks and reading uncertainty.',
           '- Imported SASP Atlas protein measurements, cell-density normalization records and paired CITE-seq RNA/ADT measurements.',
           '- Added an independent RNA study, with source predictions frozen before comparison and no target kinetic refitting.',
           '- Retained six competing mechanisms, protocol-specific medium replacement, held-out prediction, structural sensitivity and the cross-generator benchmark.','',
           '## Recovered post-withdrawal RNA evidence','',
           '[GSE222400](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE222400) supplies 52 samples in the separate NCBI count matrix. Fifty enter the time-course analysis; the two replicative-senescence samples have no comparable experimental time origin and are excluded. The three exposure arms contain 16 samples each, plus two shared untreated controls. D0 is the end of a 24-hour induction, so D16 is model day 17. Medium replacements are represented at days 1, 5, 9, 13 and 17.','',
           'The [NCBI processing method](https://www.ncbi.nlm.nih.gov/geo/info/rnaseqcounts.html) uses HISAT2/featureCounts and can differ from the authors’ analysis. Exact GeneID mappings come from the pinned GPL17586 annotation. Median-of-ratios normalization learns its reference genes and geometric means from pre-holdout samples only. This measures relative expression, not RNA per cell. No longitudinal untreated controls are available.','',
           '| Pulse | Best mechanistic predictor | Whitened MSE | Feedback MSE | Last-observation MSE |',
           '|---|---|---:|---:|---:|']
    prediction_notes=[]
    for condition,c in data['withdrawal_comparisons'].items():
        valid=[f for f in c['fits'] if f['optimizer_converged']]
        if not valid:
            lines.append(f'| {condition} | No converged fit | — | — | — |');continue
        best=min(valid,key=lambda f:f['holdout_whitened_MSE'])
        fb=min((f for f in valid if f['model']=='feedback'),key=lambda f:f['holdout_whitened_MSE'])
        lines.append(f"| {condition} | {best['model']} / {best['structure']} | {best['holdout_whitened_MSE']:.2f} | {fb['holdout_whitened_MSE']:.2f} | {c['forecast_baselines']['last_observation']['holdout_whitened_MSE']:.2f} |")
        baseline=c['forecast_baselines']['last_observation']['holdout_whitened_MSE']
        if best['model']!='feedback':
            prediction_notes.append(f"{condition}: {best['model'].replace('_',' ')} predicts best among the mechanistic candidates.")
        if baseline<fb['holdout_whitened_MSE']:
            prediction_notes.append(f'{condition}: the last-observation forecast outperforms feedback.')
        elif baseline and (baseline-fb['holdout_whitened_MSE'])/baseline<.05:
            prediction_notes.append(f'{condition}: feedback improves on the last-observation forecast by less than 5%.')
    lines+=['',' '.join(prediction_notes)+' These are descriptive comparisons on a small held-out sample, not significance tests.','',
            'Ranking alone is not mechanism identification. Large forecast errors, parameter-bound hits, observationally equivalent candidates and absent longitudinal controls limit interpretation. The final D16 pair is never used to fit kinetics, observation coefficients or noise covariance.','',
            '![Withdrawal RNA](withdrawal_DXR.png)','',
            '## Quantitative arsenite evidence','',
            '[Okamura et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC11701098/) provides aggregate figures rather than downloadable replicate tables. `arsenite_digitized.csv` records all 38 values, image hashes, axis anchors and point/bar coordinates. Audit overlays show every reading. ±2 pixels is a digitization sensitivity range, not an experimental confidence interval.','']
    ex=data['arsenite']['SA_beta_gal_control_adjusted']
    lines += [f"Approximate SA-β-gal positivity changes from {ex[0]['treated_percent']:.1f}% to {ex[1]['treated_percent']:.1f}% in treated cells, while controls change from {ex[0]['control_percent']:.1f}% to {ex[1]['control_percent']:.1f}%. The control-adjusted excess is approximately {ex[0]['excess_percentage_points']:.1f} versus {ex[1]['excess_percentage_points']:.1f} percentage points at withdrawal and seven days later. These point estimates illustrate why absolute marker decline should not be called recovery. Five microscopy fields are not treated as five independent cultures.",'',
              'The withdrawal proliferation assay uses normalized luminescence after harvesting/reseeding. Constant and relaxing inhibition curves are fitted descriptively to the treated/control signal. The apparent relaxation rate is **not senescent-cell clearance**: metabolism, viability, control growth and shared normalization can contribute. Exposure PCR uses 18S, whereas withdrawal PCR uses RPLP1; those scales are not pooled into a decay rate. Near-zero control bars receive no fold-change estimate.','',
              '![Arsenite evidence](arsenite_public_evidence.png)','',
              '## Secreted proteins and single-cell measurements','',
              f"The [SASP Atlas source tables](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.3000599) provide {data['secretome']['reported_protein_rows']:,} reported protein comparison rows across eight contexts, including {data['secretome']['panel_rows']} rows for the predefined marker panel. Source workbook row/cell addresses and author q-values/SD are retained. No SD is converted to biological SEM and no replicate values are reconstructed. Missing proteins are not set to zero. Soluble and vesicle measurements, cell types and induction times remain separate.",'',
              'The measurements test secretome composition and temporal/context dependence. They do not identify functional fraction, receptor potency or feedback gain. Intracellular-marker proteins found in medium (for example LMNB1) are not equated with their intracellular assays.','',
              '![Public secretome](public_secretome.png)','',
              '[GSE250041](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE250041) pairs RNA and eight surface-antibody measurements in proliferating and irradiated WI-38 libraries. All RNA features contribute to library-size and mitochondrial QC; antibody counts are excluded from RNA QC. RNA and ADT are normalized separately. The main filter requires at least 500 detected RNA features and at most 20% mitochondrial RNA; alternative thresholds are reported.','',
              '| Library | Deposited cells | Retained cells | Biological libraries |','|---|---:|---:|---:|']
    for x in data['single_cell']['libraries']:
        lines.append(f"| {x['library']} | {x['cells_before_QC']} | {x['cells_after_QC']} | 1 |")
    lines+=['','Marker detection fractions, within-library distributions and paired RNA/ADT associations describe heterogeneity. They are not senescent-state sensitivity/specificity estimates. Thousands of cells do not replace biological replication; no treatment p-values, biological confidence intervals or trained senescence classifier are reported. No doublet removal or reproduction of author clusters is claimed.','',
            '![Paired single-cell evidence](public_single_cell.png)','',
            '## Independent-context validation','',
            '[GSE235768](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE235768) adds 24 samples from BJ and HFL1 fibroblasts, with three samples in each treated/control group. H₂O₂ arms include 72 hours of recovery; bortezomib arms use sustained treatment. Raw 3-prime feature counts are aggregated by gene and normalized against the independent study’s controls.','',
            'Each source model’s RNA direction predictions are frozen using GSE144397 training times. Those predictions are compared with the independent-study contrasts and eligible secretome measurements. All source models are shown; no target outcomes select or refit a source model. This tests response-direction consistency across systems, not transferable kinetic rates or feedback necessity. Sign agreement across a few correlated genes has no p-value or biological confidence interval.','',
            '| Independent context | Agreeing directions across source models | Compared markers |',
            '|---|---:|---:|']
    checks=pd.DataFrame(data['external_validation']['checks'])
    for target,group in checks.groupby('target'):
        counts=sorted(set(group.informative_genes))
        lines.append(f"| {target} | {group.sign_agreement.min()}–{group.sign_agreement.max()} | {', '.join(map(str,counts))} |")
    benchmark=json.loads((out/'summary.json').read_text())['synthetic_benchmark']
    lines+=['','These mixed results include disagreements. The comparisons are not evidence that one mechanism explains all cell types, exposures or times.','',
            '![Independent RNA contrasts](independent_RNA.png)','',
            '## Cross-generator uncertainty check','',
            f"The benchmark includes {len(benchmark['records'])} synthetic trials ({benchmark['trials_per_truth']} per generating mechanism). Feedback was selected in {benchmark['false_feedback_selections']} of {benchmark['nonfeedback_trials_with_a_winner']} nonfeedback-generated trials with a converged winner. The conditional Monte Carlo Wilson interval is {benchmark['wilson95'][0]:.3f}–{benchmark['wilson95'][1]:.3f}. This exposes model-selection ambiguity under the specified generators; it is not an empirical false-positive rate or a probability that EMD4 is true. Full records are in `summary.json`.",'',
            'The public [LX-2 arsenite study abstract](https://pubmed.ncbi.nlm.nih.gov/36089002/) reports a conditioned-medium effect on Huh-7 migration. This is relevant functional evidence in a different donor system, but no quantitative raw migration data were identified for this workflow. Migration is not treated as secondary senescence or proof of a feedback loop.','',
            '## Claims that this version can and cannot support','',
            '| Claim | Public-data status |','|---|---|',
            '| Persistent arsenite-associated marker changes | Quantitative aggregate-figure evidence, with assay and replication limits |',
            '| RNA changes after pulse withdrawal | Measured and fitted in independent WI-38 exposure protocols |',
            '| SASP protein abundance varies by context and time | Measured public proteomics |',
            '| RNA and surface-marker heterogeneity | Described in paired single-cell libraries |',
            '| RNA response directions generalize to other systems | Explicit independent comparisons; inspect agreements and failures |',
            '| Functional SASP potency in arsenite-exposed Huh-7 | Not calibrated by these public inputs |',
            '| Self-sustaining feedback, bistability, tumor promotion | Not established by these analyses |',
            '| Biological `q_sec_critical` and its lower edge | Not identifiable from these assays |','',
            'The mathematical threshold calculation in the original simulation is unchanged. New measurements are not pooled across cell systems to manufacture a physiological parameter estimate. Public data support triangulation and falsification; they do not supply every matched perturbation required for a causal feedback claim.','',
            '## Reproduce and review','',
            'From the parent directory, run `python3 -m emd4_simulation.public_data complete --offline`. `complete_summary.json` contains all results; `complete_source_manifest.json` contains pinned inputs; `complete_runtime.json` records code/input hashes and versions. Tables, digitization overlays and figures are standalone artifacts. See `public_data/README.md` for scope and commands. Original manuscript files remain unchanged; `manuscript_claims.md` provides wording tied to the new evidence.', '']
    (out/'complete_report.md').write_text('\n'.join(lines))
    (out/'manuscript_claims.md').write_text('''# Public-data evidence wording for manuscript review

Suggested Methods wording:

We compared six candidate mechanisms using measured RNA observables with gene-specific observation coefficients and correlated measurement errors. Public time-course protocols retained their exposure, withdrawal and medium-replacement times. Final-time samples were held out from fitting. NCBI-generated count data were used for GSE222400 because the submitter tables contained differential-expression statistics rather than sample-level expression. Independent-study RNA and secretome comparisons used frozen source predictions. Single-cell RNA/ADT analyses were descriptive because each condition had one biological library.

Suggested Results wording:

The public data document persistent arsenite-associated marker changes, post-withdrawal RNA trajectories in other exposure systems, and substantial secretome and single-cell heterogeneity. Candidate mechanisms were compared by held-out prediction rather than their ability to reproduce their own latent state. Relative predictive ranking did not establish feedback necessity. Digitized arsenite figures were reported as aggregate estimates with reading uncertainty, without reconstruction of experimental replicates.

Required limitation:

These analyses do not quantitatively calibrate functional SASP potency in arsenite-exposed Huh-7 cells, demonstrate bistability or establish tumor promotion. RNA, secreted protein abundance, surface-antibody counts and latent senescence occupancy are distinct quantities. The reported model threshold remains conditional on mechanistic assumptions and should not be described as an experimentally measured biological threshold.

Numerical findings and source links: complete_report.md. No manuscript DOCX has been edited automatically.
''')


def run_complete(*,data_dir=DEFAULT_DATA,outdir=None,offline=False,trials=25,starts=3):
    out=Path(outdir) if outdir else ROOT/'public_data_results'
    # Recompute base results with the current code before adding new evidence.
    base=run(data_dir=data_dir,outdir=out,offline=offline,trials=trials,starts=starts)
    paths=fetch_extensions(data_dir,offline=offline)
    print('Reading recovered withdrawal counts and independent study',flush=True)
    washout,count_audit=washout_expression(paths)
    independent,independent_audit=independent_expression(paths)
    washout.to_csv(out/'withdrawal_observations.csv',index=False)
    independent.to_csv(out/'independent_observations.csv',index=False)
    comparisons={}
    for condition in ('SDS','KCl','DXR'):
        print(f'Fitting recovered withdrawal series: {condition}',flush=True)
        subset=washout[washout.condition.isin([condition,'control'])].copy()
        comparisons[condition]=compare_timecourse(subset,exposure_days=1.,starts=starts,
                                                  medium_changes=(1.,5.,9.,13.,17.))
        comparisons[condition]['interpretation']='Within-study held-out withdrawal prediction using recovered sample counts; no feedback establishment.'
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
    # Keep the previous report intact as the core analysis, but make the complete
    # report discoverable to readers of the old entry point.
    legacy=out/'report.md'
    legacy.write_text('> Updated public-data version: [complete report](complete_report.md). The original submitter GSE222400 files remain excluded; a separate NCBI count matrix is now analysed.\n\n'+legacy.read_text())
    print(f'Completed public-data analysis: {out / "complete_report.md"}',flush=True)
    return data
