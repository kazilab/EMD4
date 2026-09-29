"""Additional public inputs for the complete, public-data-only evidence analysis."""
import json
from dataclasses import asdict
from pathlib import Path
from .sources import Source, geo, fetch_source, DEFAULT_DATA, fetch_all

EXTENSION_SOURCES = [
    Source('atlas_table','pbio.3000599.s007.xlsx',
           'https://journals.plos.org/plosbiology/article/file?type=supplementary&id=10.1371/journal.pbio.3000599.s007',
           'PXD013721','secreted_protein_measurement','SASP Atlas S1 protein comparisons; abundance, not potency.'),
    Source('atlas_culture','pbio.3000599.s012.xlsx',
           'https://journals.plos.org/plosbiology/article/file?type=supplementary&id=10.1371/journal.pbio.3000599.s012',
           'PXD013721','protocol','SASP Atlas S6 cell-density normalization details.'),
    Source('washout_counts','GSE222400_raw_counts_GRCh38.p13_NCBI.tsv.gz',
           'https://www.ncbi.nlm.nih.gov/geo/download/?type=rnaseq_counts&acc=GSE222400&format=file&file=GSE222400_raw_counts_GRCh38.p13_NCBI.tsv.gz',
           'GSE222400','NCBI_reprocessed_raw_counts','Independent NCBI HISAT2/featureCounts processing of public sequencing reads; distinct from excluded submitter DE tables.'),
    geo('cite_metadata','GSE250041_family.soft.gz','GSE250041','soft','CITE-seq sample metadata.'),
    geo('withdrawal_metadata','GSE235768_family.soft.gz','GSE235768','soft','Independent H2O2 recovery and bortezomib models.'),
    geo('withdrawal_expression','GSE235768_normalized_counts_table.txt.gz','GSE235768','suppl','Submitter EDASeq-normalized sample counts; retained for source inspection, not used in fits.'),
    geo('withdrawal_counts','GSE235768_raw_counts_table.txt.gz','GSE235768','suppl','Raw 3-prime feature counts; gene aggregation and training-frozen normalization.'),
]
for condition in ('Proliferating','Senescent'):
    for kind in ('features.tsv','barcodes.tsv','matrix.mtx'):
        EXTENSION_SOURCES.append(geo('cite_'+condition.lower()+'_'+kind.split('.')[0],
                                    f'GSE250041_{condition}_{kind}.gz','GSE250041','suppl',
                                    'Paired RNA/ADT count matrix component.',300_000_000))


def fetch_extensions(data_dir=DEFAULT_DATA, *, offline=False, base_paths=None):
    paths = dict(base_paths or fetch_all(data_dir,offline=offline))
    for source in EXTENSION_SOURCES:
        print(f'Public extension input: {source.key}',flush=True)
        paths[source.key]=fetch_source(source,data_dir,offline=offline)
    manifest=json.loads(Path(data_dir,'manifest.json').read_text())
    manifest['extension_sources']=[json.loads(Path(str(paths[s.key])+'.provenance.json').read_text())
                                   for s in EXTENSION_SOURCES]
    manifest['current_extension_input_declarations']=[asdict(s) for s in EXTENSION_SOURCES]
    # This complete manifest supersedes the earlier resource-only catalog status.
    manifest['supporting_resources']['PXD013721']['status']='imported_secretome_comparisons'
    manifest['supporting_resources']['GSE250041']['status']='imported_descriptive_single_cell_RNA_ADT'
    Path(data_dir,'complete_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return paths
