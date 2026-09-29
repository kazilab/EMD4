"""Count-table adapters with training-frozen normalization and exact sample joins."""
from __future__ import annotations
import re
import csv
import gzip
import numpy as np
import pandas as pd
from .adapters import GENES, soft_samples, validate_observations


def frozen_median_ratio(counts, reference_samples, *, min_mean=10):
    """Median-of-ratios with reference genes/geomeans learned from training only.

    Zero-containing reference genes are excluded. Sample size factors may use
    that sample's counts against the frozen reference, including held-out samples;
    no held-out value enters feature selection or the reference itself.
    """
    if not counts.columns.is_unique or not counts.index.is_unique:
        raise ValueError('Count identifiers must be unique')
    c=counts.to_numpy(float)
    if not np.isfinite(c).all() or (c<0).any():
        raise ValueError('Counts must be finite and nonnegative')
    if not reference_samples or not set(reference_samples).issubset(counts.columns):
        raise ValueError('Invalid normalization reference samples')
    train=counts.loc[:,reference_samples].to_numpy(float)
    keep=(train>0).all(axis=1)&(train.mean(axis=1)>=min_mean)
    if keep.sum()<10:
        raise ValueError('Too few positive reference genes for normalization')
    geometric=np.exp(np.log(train[keep]).mean(axis=1))
    factors=np.median(c[keep]/geometric[:,None],axis=0)
    if (factors<=0).any() or not np.isfinite(factors).all():
        raise ValueError('Nonpositive library size factor')
    normalized=counts.divide(pd.Series(factors,index=counts.columns),axis='columns')
    return normalized, {'reference_samples':list(reference_samples),
                        'reference_gene_count':int(keep.sum()),
                        'reference_genes':counts.index[keep].astype(str).tolist(),
                        'reference_geometric_means':geometric.tolist(),
                        'size_factors':dict(zip(counts.columns, factors.tolist())),
                        'method':'positive-in-all-reference-genes median-of-ratios; training-frozen reference'}


def washout_samples(paths):
    meta=soft_samples(paths['timecourse_metadata'])
    rows=[]
    for sample,m in meta.items():
        title=m['title'][0]
        match=re.fullmatch(r'(SDS|KCl|DXR)_D(\d+)_(\d+)',title)
        if match:
            condition,day,rep=match.groups()
            rows.append(dict(sample=sample,condition=condition,replicate=int(rep),
                             time_days=1+int(day),post_withdrawal_days=int(day),title=title))
        elif title.startswith('Untreated-control_'):
            rows.append(dict(sample=sample,condition='control',replicate=int(title[-1]),
                             time_days=0.,post_withdrawal_days=None,title=title))
        elif not title.startswith('Replicative Sen_'):
            raise ValueError(f'Unrecognised withdrawal sample title: {title}')
    return pd.DataFrame(rows)


def washout_expression(paths):
    table=pd.read_csv(paths['washout_counts'],sep='\t',index_col=0)
    # GeneID mappings are read from the pinned GPL17586 annotation already
    # used for the marker panel; no external or inferred symbol conversion.
    mapping = {}
    active = False
    columns = None
    with gzip.open(paths['quiescence_metadata'], 'rt') as f:
        for line in f:
            if line.startswith('!platform_table_begin'):
                active = True
                continue
            if line.startswith('!platform_table_end'): break
            if not active: continue
            cells = next(csv.reader([line], delimiter='\t'))
            if columns is None:
                columns = cells
                continue
            row = dict(zip(columns, cells))
            entries = [x.split(' // ') for x in row['gene_assignment'].split(' /// ')]
            symbols = {{'IL8':'CXCL8'}.get(x[1],x[1]) for x in entries if len(x)>1}
            if len(symbols)!=1 or not symbols.intersection(GENES): continue
            symbol = next(iter(symbols))
            for x in entries:
                if len(x)>=5 and x[4].isdigit():
                    if x[4] in mapping and mapping[x[4]]!=symbol:
                        raise ValueError('Conflicting panel GeneID annotation')
                    mapping[x[4]]=symbol
    table.index=table.index.astype(str)
    meta=washout_samples(paths)
    if not set(meta['sample']).issubset(table.columns):
        raise ValueError('NCBI matrix is missing protocol samples')
    excluded_samples=sorted(set(table.columns)-set(meta['sample']))
    table=table.loc[:,meta['sample']]
    training=meta.loc[meta.time_days<17,'sample'].tolist()
    normalized,audit=frozen_median_ratio(table,training)
    symbols=pd.Series(mapping).reindex(normalized.index)
    selected=symbols.isin(GENES)
    panel=normalized.loc[selected].copy()
    panel.index=symbols[selected].to_numpy()
    if not panel.index.is_unique:
        raise ValueError('Multiple NCBI IDs map to a selected gene; inspect before aggregation')
    rows=[]
    for m in meta.to_dict('records'):
        for gene in GENES:
            if gene not in panel.index:continue
            rows.append({**m,'study':'GSE222400_NCBI','gene':gene,
                         'value':float(np.log2(1+panel.loc[gene,m['sample']])),
                         'unit':'log2_size_factor_normalized_count_plus_1',
                         'time_origin':'start_of_24_hour_induction',
                         'exposure':'sham_control' if m['condition']=='control' else '24_hour_pulse',
                         'evidence':'NCBI_reprocessed_withdrawal_RNA'})
    frame=pd.DataFrame(rows);validate_observations(frame)
    audit.update({'sample_count':len(meta),'missing_genes':sorted(set(GENES)-set(frame.gene)),
                  'excluded_replicative_samples':excluded_samples, 'GeneID_to_symbol':mapping,
                  'count_source':'NCBI-generated HISAT2/featureCounts, not submitter DE files',
                  'protocol':{'exposure_days':1.,'medium_changes_days':[1.,5.,9.,13.,17.],
                              'D0_absolute_day':1.,'heldout_absolute_day':17.},
                  'limitations':['No longitudinal untreated controls in the deposit.',
                                 'Relative count normalization does not measure RNA per cell.',
                                 'NCBI minimal processing QC does not certify experimental sample quality.']})
    return frame,audit


def independent_expression(paths):
    """Independent experiment: do not combine with calibration to select genes."""
    table=pd.read_csv(paths['withdrawal_counts'],sep='\t')
    meta=soft_samples(paths['withdrawal_metadata'])
    samples=[]
    for accession,m in meta.items():
        match=re.fullmatch(r'(BJ|HFL1) cells, (Bortezomib|H2O2)-(treated|control), biol rep(\d+)',m['title'][0])
        if not match:raise ValueError(f'Unexpected independent-study title: {m["title"][0]}')
        cell,agent,condition,replicate=match.groups()
        samples.append(dict(sample=accession,column=m['description'][0],cell=cell,
                            agent=agent,condition=condition,replicate=int(replicate)))
    columns=[m['column'] for m in samples]
    if not set(columns).issubset(table):raise ValueError('Count/sample mapping incomplete')
    # Multiple 3-prime transcript features are summed, never treated as replicates.
    counts=table.groupby('gene_id')[columns].sum()
    symbol_map=table[['gene_id','gene_name']].drop_duplicates().set_index('gene_id').gene_name
    if not symbol_map.index.is_unique:raise ValueError('Ambiguous gene annotation')
    controls=[m['column'] for m in samples if m['condition']=='control']
    normalized,audit=frozen_median_ratio(counts,controls)
    selected=symbol_map.isin(GENES)
    panel=normalized.loc[symbol_map.index[selected]]
    panel.index=symbol_map[selected].to_numpy()
    if not panel.index.is_unique:raise ValueError('Ambiguous selected symbols')
    rows=[]
    for m in samples:
        for gene in GENES:
            if gene not in panel.index:continue
            rows.append({**m,'study':'GSE235768','gene':gene,
                         'value':float(np.log2(1+panel.loc[gene,m['column']])),
                         'unit':'log2_size_factor_normalized_count_plus_1',
                         'recovery_hours':72 if m['agent']=='H2O2' else None})
    frame=pd.DataFrame(rows)
    audit.update({'sample_count':len(samples),'missing_genes':sorted(set(GENES)-set(frame.gene)),
                  'role':'independent context validation of RNA response directions; no kinetic refit',
                  'limitation':'H2O2 has a 72-hour recovery endpoint; bortezomib is continuous. Cell types and protocols differ from arsenite Huh-7.'})
    return frame,audit
