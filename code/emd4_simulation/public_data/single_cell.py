"""Descriptive paired RNA/ADT analysis without treating cells as biological replicates."""
from itertools import islice
import gzip
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from .adapters import GENES

SURFACE_RNA={'CD112':('NECTIN2','PVRL2'),'HLA_A_B_C':('HLA-A','HLA-B','HLA-C'),
             'CD44':('CD44',),'CD54':('ICAM1',),'CD26':('DPP4',),
             'CD49a':('ITGA1',),'CD73':('NT5E',),'CD109':('CD109',)}


def stream_selected_counts(matrix_path, features, wanted, *, chunk_lines=150000):
    """Compute per-cell QC from all RNA while retaining only selected features.

    Avoid densifying the full 36k-gene matrix. MTX dimensions/coordinates/counts
    are validated. This importer requires unique, column-major coordinates,
    as supplied by the pinned matrices, so duplicate entries cannot inflate
    detected-feature QC. Modalities remain separate; ADT never inflates RNA QC.
    """
    is_rna=features.modality.eq('Gene Expression').to_numpy()
    is_mito=features.symbol.str.startswith('MT-').to_numpy()&is_rna
    wanted=np.asarray(wanted,int)
    with gzip.open(matrix_path,'rt') as f:
        if next(f).strip()!='%%MatrixMarket matrix coordinate integer general':
            raise ValueError('Unexpected count matrix format')
        line=next(f)
        while line.startswith('%'):line=next(f)
        n_features,n_cells,n_entries=map(int,line.split())
        if n_features!=len(features):raise ValueError('Feature matrix dimension mismatch')
        if len(np.unique(wanted))!=len(wanted) or (wanted<0).any() or (wanted>=n_features).any():
            raise ValueError('Invalid selected feature indices')
        selected=np.zeros((len(wanted),n_cells),dtype=np.int64)
        mapping=np.full(n_features,-1,dtype=int);mapping[wanted]=np.arange(len(wanted))
        total=np.zeros(n_cells);detected=np.zeros(n_cells);mito=np.zeros(n_cells)
        read=0;last_coordinate=-1
        while lines:=list(islice(f,chunk_lines)):
            a=np.fromstring(''.join(lines),sep=' ',dtype=np.int64)
            if len(a)!=3*len(lines):raise ValueError('Malformed MTX entries')
            a=a.reshape(-1,3);r,c,v=a[:,0]-1,a[:,1]-1,a[:,2]
            if (r<0).any() or (r>=n_features).any() or (c<0).any() or (c>=n_cells).any() or (v<=0).any():
                raise ValueError('Invalid count matrix coordinate or value')
            coordinate=c*n_features+r
            if coordinate[0]<=last_coordinate or (np.diff(coordinate)<=0).any():
                raise ValueError('MTX requires unique, column-major coordinates')
            last_coordinate=int(coordinate[-1])
            rna=is_rna[r]
            total+=np.bincount(c[rna],weights=v[rna],minlength=n_cells)
            detected+=np.bincount(c[rna],minlength=n_cells)
            mitochondrial=is_mito[r]
            mito+=np.bincount(c[mitochondrial],weights=v[mitochondrial],minlength=n_cells)
            keep=mapping[r]>=0
            np.add.at(selected,(mapping[r[keep]],c[keep]),v[keep])
            read+=len(lines)
        if read!=n_entries:raise ValueError('MTX entry count mismatch')
    return selected,{'rna_UMI':total,'detected_RNA_features':detected,
                     'mitochondrial_fraction':np.divide(mito,total,out=np.ones_like(total),where=total>0)}


def analyze_single_cell(paths):
    rows=[];pairs=[];audit=[];distributions={}
    for state in ('proliferating','senescent'):
        print(f'Importing paired single-cell measurements: {state}',flush=True)
        prefix='cite_'+state+'_'
        features=pd.read_csv(paths[prefix+'features'],sep='\t',header=None,
                             names=['id','symbol','modality'])
        barcodes=pd.read_csv(paths[prefix+'barcodes'],sep='\t',header=None)[0].tolist()
        if len(set(barcodes))!=len(barcodes):raise ValueError('Duplicate cell barcode within library')
        wanted_symbols=set(GENES)|{x for group in SURFACE_RNA.values() for x in group}
        selected_indices=features.index[features.symbol.isin(wanted_symbols)|features.modality.eq('Antibody Capture')].to_numpy()
        selected,qc=stream_selected_counts(paths[prefix+'matrix'],features,selected_indices)
        if selected.shape[1]!=len(barcodes):raise ValueError('Cell barcode dimension mismatch')
        keep=(qc['detected_RNA_features']>=500)&(qc['mitochondrial_fraction']<=.2)
        if keep.sum()<10:raise ValueError('Too few cells after prespecified QC')
        sub=features.loc[selected_indices].reset_index(drop=True)
        rna=sub.modality.eq('Gene Expression').to_numpy()
        adt=sub.modality.eq('Antibody Capture').to_numpy()
        if sub.loc[rna,'symbol'].duplicated().any() or sub.loc[adt,'id'].duplicated().any():
            raise ValueError('Ambiguous duplicate selected RNA symbol or ADT identifier')
        rna_values=np.log1p(selected[rna][:,keep]/qc['rna_UMI'][keep]*10000)
        adt_log=np.log1p(selected[adt][:,keep])
        adt_centered=adt_log-adt_log.mean(axis=0)
        rna_map={g:v for g,v in zip(sub.loc[rna,'symbol'],rna_values)}
        adt_map={g:v for g,v in zip(sub.loc[adt,'id'],adt_centered)}
        for gene in GENES:
            if gene not in rna_map:continue
            values=rna_map[gene]
            rows.append({'library':state,'modality':'RNA','feature':gene,'cells':int(keep.sum()),
                         'median':float(np.median(values)),'q10':float(np.quantile(values,.1)),
                         'q90':float(np.quantile(values,.9)),
                         'nonzero_fraction':float(np.mean(values>0)),
                         'unit':'log1p_RNA_counts_per_10000','biological_replicates_per_condition':1})
        for marker,values in adt_map.items():
            rows.append({'library':state,'modality':'ADT','feature':marker,'cells':int(keep.sum()),
                         'median':float(np.median(values)),'q10':float(np.quantile(values,.1)),
                         'q90':float(np.quantile(values,.9)), 'nonzero_fraction':None,
                         'unit':'within_cell_centered_log1p_ADT','biological_replicates_per_condition':1})
            genes=[g for g in SURFACE_RNA.get(marker,()) if g in rna_map]
            if genes:
                # HLA antibody binds several proteins; this is explicitly the mean
                # of their transformed RNA channels, not a one-gene correspondence.
                expression=np.mean([rna_map[g] for g in genes],axis=0)
                rho=spearmanr(expression,values).statistic if np.std(expression)>0 and np.std(values)>0 else None
                pairs.append({'library':state,'ADT':marker,'RNA_genes':genes,
                              'spearman_rho':float(rho) if rho is not None else None,
                              'pvalue':None,'inference':'descriptive_within_library_only'})
        audit.append({'library':state,'cells_before_QC':len(barcodes),'cells_after_QC':int(keep.sum()),
                      'cells_stricter_mito10pct':int(((qc['detected_RNA_features']>=500)&(qc['mitochondrial_fraction']<=.1)).sum()),
                      'cells_min200_features':int(((qc['detected_RNA_features']>=200)&(qc['mitochondrial_fraction']<=.2)).sum()),
                      'missing_panel_genes':sorted(set(GENES)-set(rna_map)),
                      'RNA_features':int(features.modality.eq('Gene Expression').sum()),
                      'ADT_features':int(adt.sum()),'biological_replicates':1})
        # Quantile profiles avoid distributing tens of thousands of single-cell
        # values as though they were independent biological replicates.
        distributions[state]={g:np.quantile(v,np.linspace(0,1,101)).tolist() for g,v in rna_map.items() if g in GENES}
    return pd.DataFrame(rows),{'libraries':audit,'paired_RNA_ADT':pairs,'RNA_quantile_profiles':distributions,
             'protocol':'WI-38, 10 Gy irradiation, 10 days recovery; proliferating comparator.',
             'calibrates':'Descriptive marker sparsity and paired RNA/surface-protein association within the deposited libraries.',
             'limits':['One library per condition: cells are not independent biological replicates.',
                       'No treatment p-values, biological confidence intervals or senescence classifier validation.',
                       'No surface-marker threshold is treated as senescent-state truth.',
                       'Selected eight-antibody panel has no isotype-based background correction; centered ADT values are compositional.',
                       'No doublet removal, latent-state discovery or exact reproduction of author clusters is claimed.',
                       'Surface abundance does not measure secreted-protein potency.']}
