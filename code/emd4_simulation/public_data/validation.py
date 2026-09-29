"""Independent-context checks with frozen source predictions and no target refit."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, t as student_t
from .inference import predict


def independent_contrasts(frame):
    rows=[]
    for (cell,agent,gene),part in frame.groupby(['cell','agent','gene']):
        x=part[part.condition=='treated'].value.to_numpy()
        y=part[part.condition=='control'].value.to_numpy()
        vx=np.var(x,ddof=1)/len(x);vy=np.var(y,ddof=1)/len(y)
        se=np.sqrt(vx+vy)
        df=(vx+vy)**2/(vx*vx/(len(x)-1)+vy*vy/(len(y)-1)) if se else None
        margin=float(student_t.ppf(.975,df)*se) if se else None
        delta=float(x.mean()-y.mean())
        rows.append({'cell':cell,'agent':agent,'gene':gene,'difference_log2_normalized_counts_plus_1':delta,
                     'lo95':delta-margin if margin is not None else None,
                     'hi95':delta+margin if margin is not None else None,
                     'n_treated':len(x),'n_control':len(y),'multiplicity_adjusted':False})
    return pd.DataFrame(rows)


def external_direction_checks(source_comparison, independent, protein_panel):
    contrasts=independent_contrasts(independent)
    checks=[];predictions=[]
    for fit in source_comparison['fits']:
        if not fit['optimizer_converged']:continue
        # Use the latest training time, not source holdout or target outcomes, to
        # define a direction prediction. No target coefficient/threshold fitting.
        end=source_comparison['holdout_time_days']
        time=4. if end==6 else max(0.,end-1)
        response=predict(fit,[0.,time]);delta=response[1]-response[0]
        frozen=dict(zip(fit['genes'],delta))
        predictions.append({'source_model':fit['model'],'structure':fit['structure'],
                            'comparison_days':[0.,time],
                            'predicted_RNA_change':{g:float(x) for g,x in frozen.items()},
                            'target_used_for_fitting':False})
        for (cell,agent),target in contrasts.groupby(['cell','agent']):
            genes=[g for g in target.gene if g in frozen]
            measured=target.set_index('gene').loc[genes,'difference_log2_normalized_counts_plus_1'].to_numpy()
            predicted=np.array([frozen[g] for g in genes])
            informative=(np.abs(predicted)>1e-8)&(np.abs(measured)>1e-8)
            rho=spearmanr(predicted,measured).statistic
            checks.append({'source_model':fit['model'],'structure':fit['structure'],
                           'target':f'GSE235768/{cell}/{agent}', 'assay':'RNA',
                           'genes':genes,'sign_agreement':int(np.sum(np.sign(predicted[informative])==np.sign(measured[informative]))),
                           'informative_genes':int(informative.sum()),
                           'spearman_rho':float(rho) if np.isfinite(rho) else None,
                           'biological_CI':None,'interpretation':'Independent response-direction check across systems, not kinetic prediction validation.'})
        for sheet,group in protein_panel[protein_panel.compartment=='soluble'].groupby('sheet'):
            # Comparing nuclear/structural RNA to protein detected in medium is
            # misleading (e.g. LMNB1); only prespecified secretory panel + GDF15.
            eligible={'IL6','CXCL8','SERPINE1','MMP3','GDF15'}
            group=group[group.gene.isin(eligible)&group.gene.isin(frozen)]
            if group.gene.duplicated().any():raise ValueError('Duplicate protein/gene comparisons')
            if group.empty:continue
            a=np.array([frozen[g] for g in group.gene]);b=group.log2_senescent_over_control.to_numpy()
            informative=(abs(a)>1e-8)&(abs(b)>1e-8)
            checks.append({'source_model':fit['model'],'structure':fit['structure'],
                           'target':'PXD013721/'+sheet,'assay':'secreted_protein',
                           'genes':group.gene.tolist(),'sign_agreement':int(np.sum(np.sign(a[informative])==np.sign(b[informative]))),
                           'informative_genes':int(informative.sum()),'spearman_rho':None,
                           'biological_CI':None,'interpretation':'Cross-assay directional consistency only; absence/detection and study selection affect coverage.'})
    return contrasts,{'source':'GSE144397 RAS training observations only',
                      'frozen_source_predictions':predictions,'checks':checks,
                      'target_refitting':False,'parameter_transfer_claimed':False,
                      'limits':['Target datasets were chosen by assay/protocol availability, not a preregistered selection.',
                                'Nine correlated marker genes are not independent biological replicates.',
                                'Gene-wise Welch intervals are descriptive and not multiplicity adjusted.',
                                'No sign-concordance p-value or probability of EMD4 is reported.',
                                'Cross-platform response directions can generalize without validating feedback kinetics.']}
