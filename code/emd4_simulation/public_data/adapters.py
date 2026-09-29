"""Strict public-data adapters. Keep studies/assays separate and reject invalid inputs."""
from __future__ import annotations
import csv
import gzip
import hashlib
import io
import re
import tarfile
from pathlib import Path
import numpy as np
import pandas as pd

# Fixed before any model fitting. Gene expression is neither protein activity nor
# a validated senescence classifier. Missing/ambiguous mappings are reported.
PANELS = {
    'arrest': ('CDKN1A', 'CDKN2A', 'LMNB1', 'MKI67'),
    'secretory': ('IL6', 'CXCL8', 'SERPINE1', 'MMP3'),
    'stress': ('GDF15',),
}
GENES = tuple(g for panel in PANELS.values() for g in panel)


def soft_samples(path):
    samples = {}
    current = None
    with gzip.open(path, 'rt') as f:
        for line in f:
            if line.startswith('^SAMPLE = '):
                current = line.strip().split(' = ', 1)[1]
                samples[current] = {}
            elif line.startswith('^'):
                current = None
            elif current and line.startswith('!Sample_') and ' = ' in line:
                k, v = line.rstrip('\n').split(' = ', 1)
                samples[current].setdefault(k.removeprefix('!Sample_'), []).append(v)
    return samples


def platform_panel(path):
    """Use the exact GPL annotation shipped with the expression data.

    Reject a probe with multiple gene symbols. Median of retained probes for a
    gene is deterministic and independent of outcomes. Extract Ensembl gene IDs
    from the same annotation to join the separate IHH TPM data; exact IDs only.
    """
    probes, ensembl = {}, {}
    active = False
    columns = None
    with gzip.open(path, 'rt') as f:
        for line in f:
            if line.startswith('!platform_table_begin'):
                active = True
                continue
            if line.startswith('!platform_table_end'):
                break
            if not active:
                continue
            cells = next(csv.reader([line], delimiter='\t'))
            if columns is None:
                columns = cells
                continue
            row = dict(zip(columns, cells))
            assignments = row['gene_assignment'].split(' /// ')
            symbols = {x.split(' // ')[1] for x in assignments if len(x.split(' // ')) > 1}
            if len(symbols) != 1:
                continue
            symbol = next(iter(symbols))
            # Historical symbol explicitly declared; no fuzzy mapping.
            symbol = {'IL8': 'CXCL8'}.get(symbol, symbol)
            if symbol not in GENES:
                continue
            probes[row['ID']] = symbol
            ids = set(re.findall(r'gene:(ENSG\d+)', row.get('mrna_assignment', '')))
            ensembl.setdefault(symbol, set()).update(ids)
    if not probes:
        raise ValueError('No unambiguous panel annotation in supplied GPL table')
    # Ensembl IDs must identify exactly one of the selected gene symbols.
    reverse = {}
    for symbol, ids in ensembl.items():
        for eid in ids:
            reverse.setdefault(eid, set()).add(symbol)
    safe = {eid: next(iter(s)) for eid, s in reverse.items() if len(s) == 1}
    return probes, safe


def timecourse(paths):
    probes, ensembl = platform_panel(paths['quiescence_metadata'])
    meta = soft_samples(paths['quiescence_metadata'])
    raw_rows, active = [], False
    with gzip.open(paths['quiescence_expression'], 'rt') as f:
        for line in f:
            if line.startswith('!series_matrix_table_begin'):
                active = True
                continue
            if line.startswith('!series_matrix_table_end'):
                break
            if active:
                raw_rows.append(line)
    expression = pd.read_csv(io.StringIO(''.join(raw_rows)), sep='\t', index_col=0)
    expression = expression.loc[expression.index.intersection(probes)]
    expression.index = expression.index.map(probes)
    expression = expression.groupby(level=0).median()
    rows = []
    # Matched RAS/quiescence trajectories, with RAF kept as a separate context.
    for accession, m in meta.items():
        title = m['title'][0]
        match = re.fullmatch(r'WI38_(QUIESCENCE|RASOIS)_REP(\d+)_(T0|\d+H)', title)
        raf = re.fullmatch(r'WI38_SENESCENCE_TRANSCRIPTOME_RAF_(T0|\d+H)_REP(\d+)', title)
        if match:
            condition, replicate, time = match.groups()
        elif raf:
            time, replicate = raf.groups()
            condition = 'RAFOIS'
        else:
            continue
        hours = 0 if time == 'T0' else int(time[:-1])
        for gene in GENES:
            if gene in expression.index:
                rows.append(dict(study='GSE144397', sample=accession, title=title,
                                 condition=condition, replicate=int(replicate),
                                 time_days=hours/24, time_origin='start_of_induction',
                                 exposure='continuous', gene=gene,
                                 value=float(expression.loc[gene, accession]),
                                 unit='submitter_normalised_log2_array_expression',
                                 evidence='other_system_RNA'))
    frame = pd.DataFrame(rows)
    validate_observations(frame)
    audit = {'platform': 'GPL17586', 'probe_to_symbol': probes,
             'ensembl_to_symbol': ensembl,
             'missing_genes': sorted(set(GENES)-set(frame.gene)),
             'sample_count': int(frame['sample'].nunique()),
             'normalisation': 'Submitter RMA and batch correction retained; no count normalisation.',
             'limitation': 'Continuous induction; no post-withdrawal observation. Superseries batch correction used all source samples; this is not a fully independent validation cohort.'}
    return frame, audit


def ihh(paths, ensembl):
    table = pd.read_csv(paths['ihh_tpm'], sep='\t', index_col=0)
    table.index = table.index.str.replace(r'\.\d+$', '', regex=True)
    table = table.loc[table.index.intersection(ensembl)]
    if table.empty:
        raise ValueError('No exact annotation matches to IHH TPM')
    table.index = table.index.map(ensembl)
    # Multiple exact Ensembl gene IDs (if present) are summed on the TPM scale,
    # then log2 transformed. Symbol mapping is recorded with all source IDs.
    table = np.log2(1 + table.groupby(level=0).sum())
    conditions = {'w': 'control', 'x': 'direct_DOX', 'y': 'control_CM', 'z': 'DOX_CM'}
    rows = []
    for accession, m in soft_samples(paths['ihh_metadata']).items():
        match = re.search(r'UJ-3100-([wxyz]\d+)', m['title'][0])
        if not match:
            raise ValueError(f'Unrecognised IHH sample: {accession}')
        column = match.group(1)
        if column not in table:
            raise ValueError(f'Missing TPM column: {column}')
        for gene in GENES:
            if gene in table.index:
                rows.append(dict(study='GSE210140', sample=accession, title=m['title'][0],
                                 condition=conditions[column[0]], replicate=int(column[1:]),
                                 time_days=3 if column[0] in 'yz' else 3+2/24,
                                 time_origin='recipient_transfer' if column[0] in 'yz' else 'donor_induction',
                                 exposure='50_percent_CM' if column[0] in 'yz' else '2_hour_pulse',
                                 gene=gene, value=float(table.loc[gene, column]),
                                 unit='log2_TPM_plus_1', evidence='other_system_RNA'))
    frame = pd.DataFrame(rows)
    validate_observations(frame)
    return frame, {'sample_count': int(frame['sample'].nunique()),
                   'missing_genes': sorted(set(GENES)-set(frame.gene)),
                   'limitation': 'Endpoint RNA contrasts only; no protein potency, washout persistence, or feedback-gain calibration.'}


def validate_observations(frame):
    required = {'study', 'sample', 'condition', 'replicate', 'time_days', 'gene', 'value', 'unit'}
    if not required.issubset(frame) or frame.empty:
        raise ValueError('Incomplete or empty observation table')
    if frame.duplicated(['study', 'sample', 'gene']).any():
        raise ValueError('Duplicate sample/gene observation')
    if not np.isfinite(frame[['time_days', 'value']].to_numpy()).all():
        raise ValueError('Nonfinite measured observation')
    if (frame.time_days < 0).any():
        raise ValueError('Negative protocol time')


def audit_washout_archive(path):
    """Inspect, never extract, untrusted tar member paths. DE tables are not RPKM."""
    files = []
    with tarfile.open(path) as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            payload = archive.extractfile(member).read()
            if member.name.endswith('.gz'):
                payload = gzip.decompress(payload)
            header = payload.splitlines()[0].decode('utf-8-sig')
            is_de = all(x in header for x in ('baseMean', 'log2FoldChange', 'padj'))
            files.append({'member': member.name, 'header': header,
                          'sha256_uncompressed': hashlib.sha256(payload).hexdigest(),
                          'differential_expression_table': is_de})
    de = sum(x['differential_expression_table'] for x in files)
    return {'accession': 'GSE222400', 'status': 'excluded_pending_corrected_expression',
            'files': files, 'file_count': len(files), 'DE_table_count': de,
            'unique_payloads': len({x['sha256_uncompressed'] for x in files}),
            'reason': 'Deposited files were checked as supplied. Differential-expression statistics and baseMean cannot be interpreted as individual-sample RPKM. No kinetic fit uses this archive.',
            'protocol_only': {'induction_hours': 24, 'D0_origin': 'end_of_induction',
                              'medium_replacement_days': 4}}
