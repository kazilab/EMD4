"""Declared public inputs and auditable, atomic downloads."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import urllib.request
import re

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / 'data' / 'public'


@dataclass(frozen=True)
class Source:
    key: str
    filename: str
    url: str
    accession: str
    evidence: str
    description: str
    max_bytes: int = 100_000_000


def geo(key, filename, accession, folder, description, limit=100_000_000):
    prefix = accession[:-3] + 'nnn'
    return Source(key, filename,
                  f'https://ftp.ncbi.nlm.nih.gov/geo/series/{prefix}/{accession}/{folder}/{filename}',
                  accession, 'other_system_measurement', description, limit)


SOURCES = [
    geo('ihh_metadata', 'GSE210140_family.soft.gz', 'GSE210140', 'soft',
        'IHH: doxorubicin and conditioned-medium recipients; protocol and sample metadata.'),
    geo('ihh_tpm', 'GSE210140_UJ3100_exp_gene_level_tpm.txt.gz', 'GSE210140', 'suppl',
        'Published Kallisto TPM, 20 samples; RNA abundance, not SASP potency.'),
    geo('ihh_counts', 'GSE210140_UJ3100_exp_gene_level_count.txt.gz', 'GSE210140', 'suppl',
        'Published estimated gene counts; retained for reproducibility.'),
    geo('timecourse_metadata', 'GSE222400_family.soft.gz', 'GSE222400', 'soft',
        'WI-38: 24-hour induction, washout, and subsequent time course.'),
    geo('timecourse_expression', 'GSE222400_RAW.tar', 'GSE222400', 'suppl',
        'Archive described as RPKM in metadata; audit finds DE tables, excluded from expression fitting.'),
    geo('quiescence_metadata', 'GSE144397_family.soft.gz', 'GSE144397', 'soft',
        'WI-38 quiescence/OIS time courses; biological replicate metadata.'),
    geo('quiescence_expression', 'GSE144397_series_matrix.txt.gz', 'GSE144397', 'matrix',
        'Submitter-normalised Affymetrix expression, not RNA-seq counts.'),
    Source('arsenite_article', 'PMC11701098.xml',
           'https://www.ebi.ac.uk/europepmc/webservices/rest/PMC11701098/fullTextXML',
           'PMC11701098', 'matching_system_protocol',
           'Okamura 2024 Huh-7 arsenite protocol, captions and availability statement.'),
]

CATALOG_NOTES = {
    'PXD013721': {'url': 'https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.3000599',
                 'status': 'supporting_public_resource',
                 'use': 'Secretome composition; protein abundance alone cannot calibrate functional q_sec.'},
    'GSE250041': {'url': 'https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE250041',
                 'status': 'optional_future_assay_extension',
                 'use': 'CITE-seq heterogeneity; not calibration of the six existing protein/functional assays.'},
    'PMID36089002': {'url': 'https://pubmed.ncbi.nlm.nih.gov/36089002/',
                    'status': 'published_summary_only',
                    'use': 'Arsenite LX-2 conditioned medium affects Huh-7 migration, not measured secondary-senescence feedback.'},
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def fetch_source(source: Source, data_dir=DEFAULT_DATA, *, offline=False):
    """Verify cached bytes or fetch once; never silently replace a pinned input."""
    raw = Path(data_dir) / 'raw'
    raw.mkdir(parents=True, exist_ok=True)
    dest = raw / source.filename
    record = raw / (source.filename + '.provenance.json')
    if dest.exists() and record.exists():
        info = json.loads(record.read_text())
        if sha256(dest) != info['sha256']:
            raise ValueError(f'Cached checksum mismatch: {dest}')
        if info['url'] != source.url:
            raise ValueError(f'Cached URL changed: {dest}')
        return dest
    if offline:
        raise FileNotFoundError(f'Offline input missing or unverified: {dest}')
    if dest.exists():
        raise ValueError(f'Untracked input already exists: {dest}')
    tmp = dest.with_suffix(dest.suffix + '.part')
    for attempt in range(3):
        try:
            req = urllib.request.Request(source.url, headers={'User-Agent':'EMD4-public-data/1.0'})
            with urllib.request.urlopen(req, timeout=90) as response, tmp.open('wb') as out:
                total = 0
                for block in iter(lambda: response.read(1024*1024), b''):
                    total += len(block)
                    if total > source.max_bytes:
                        raise ValueError(f'Input exceeds declared size limit: {source.key}')
                    out.write(block)
                content_type = response.headers.get('Content-Type','')
            if total == 0:
                raise ValueError(f'Empty download: {source.key}')
            if source.filename.endswith('.gz'):
                with tmp.open('rb') as check:
                    if check.read(2) != b'\x1f\x8b':
                        raise ValueError(f'Expected gzip, received {content_type}: {source.key}')
            info = {**asdict(source), 'sha256':sha256(tmp), 'bytes':total,
                    'retrieved_utc':datetime.now(timezone.utc).isoformat()}
            tmp.replace(dest)
            record.write_text(json.dumps(info, indent=2)+'\n')
            return dest
        except (OSError, ValueError):
            tmp.unlink(missing_ok=True)
            if attempt == 2:
                raise
            time.sleep(attempt+1)


def fetch_all(data_dir=DEFAULT_DATA, *, offline=False):
    # Sequential records keep failures explicit and avoid hammering public servers.
    result = {}
    for source in SOURCES:
        print(f'Input: {source.key}', flush=True)
        result[source.key] = fetch_source(source, data_dir, offline=offline)
    xml = result['arsenite_article'].read_text()
    for figure in (2, 5, 6):
        name = f'ehpm-29-074-g{figure:03d}.jpg'
        match = re.search(r'urn:cdn:(blobs/[^?\s]+/'+re.escape(name)+r')', xml)
        if match is None:
            raise ValueError(f'Published figure URL missing from article XML: {name}')
        source = Source(f'arsenite_fig{figure}', name,
                        'https://cdn.ncbi.nlm.nih.gov/pmc/'+match.group(1),
                        'PMC11701098', 'matching_system_figure',
                        'Published aggregate figure; no inferred individual replicates.')
        result[source.key] = fetch_source(source, data_dir, offline=offline)
    manifest = {'sources':[json.loads(Path(str(p)+'.provenance.json').read_text())
                           for p in result.values()],
                'current_input_declarations':[asdict(s) for s in SOURCES],
                'supporting_resources':CATALOG_NOTES}
    Path(data_dir,'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return result
