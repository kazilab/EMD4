"""Read-only source-table import; no editing or generation of the source workbook.

The small OOXML value reader is part of the reproducible data pipeline, not a
workbook authoring tool. Formula cells require cached values; formulas are never
executed. Cell coordinates are retained to audit every imported statistic.
"""
import posixpath
import re
import zipfile
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from .adapters import GENES

NS={'m':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
REL='http://schemas.openxmlformats.org/officeDocument/2006/relationships'


def workbook_values(path):
    with zipfile.ZipFile(path) as z:
        if sum(x.file_size for x in z.infolist())>200_000_000:
            raise ValueError('Unexpected workbook expansion size')
        strings=[]
        if 'xl/sharedStrings.xml' in z.namelist():
            strings=[''.join(x.itertext()) for x in ET.fromstring(z.read('xl/sharedStrings.xml'))]
        workbook=ET.fromstring(z.read('xl/workbook.xml'))
        targets={x.get('Id'):x.get('Target') for x in ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))}
        result={}
        for sheet in workbook.findall('m:sheets/m:sheet',NS):
            target=targets[sheet.get('{'+REL+'}id')]
            filename=posixpath.normpath(posixpath.join('xl',target))
            if not filename.startswith('xl/'):raise ValueError('External workbook relation')
            rows=[]
            for row in ET.fromstring(z.read(filename)).findall('.//m:sheetData/m:row',NS):
                values={}
                for cell in row.findall('m:c',NS):
                    coordinate=cell.get('r')
                    col=re.match('[A-Z]+',coordinate).group()
                    value=cell.find('m:v',NS)
                    if cell.find('m:f',NS) is not None and value is None:
                        raise ValueError(f'Formula without cached source value: {coordinate}')
                    text=value.text if value is not None else ''
                    if cell.get('t')=='s':text=strings[int(text)]
                    elif cell.get('t')=='inlineStr':text=''.join(cell.find('m:is',NS).itertext())
                    values[col]=text
                rows.append((int(row.get('r')),values))
            result[sheet.get('name')]=rows
        return result


def secretome_tables(paths):
    records=[];sheet_audit=[]
    for name,rows in workbook_values(paths['atlas_table']).items():
        header=None;count=0
        for n,row in rows:
            if 'Genes' in row.values() and 'AVG Log2 Ratio' in row.values():
                header={value:col for col,value in row.items() if value};continue
            if header is None:continue
            try:
                lfc=float(row.get(header['AVG Log2 Ratio'],''))
                q=float(row.get(header['Qvalue'],''))
            except ValueError:continue
            genes=row.get(header['Genes'],'')
            # Grouped proteins stay grouped; do not create duplicate gene observations.
            members=[{'IL8':'CXCL8'}.get(x.strip(),x.strip()) for x in re.split(r'[;,]',genes) if x.strip()]
            sd=row.get(header.get('Standard Deviation',''),'')
            record={'sheet':name,'source_row':n,'genes_as_deposited':genes,
                    'gene':members[0] if len(set(members))==1 else None,
                    'uniprot':row.get(header.get('UniProtIds',''),''),
                    'log2_senescent_over_control':lfc,'author_qvalue':q,
                    'author_SD':float(sd) if sd else None,
                    'compartment':'extracellular_vesicle' if 'Exosome' in name else 'soluble',
                    'cell_type':'renal_epithelial' if 'Epithelial' in name else 'IMR90_fibroblast',
                    'log2_ratio_cell':f"{header['AVG Log2 Ratio']}{n}",
                    'functional_activity_measured':False}
            records.append(record);count+=1
        sheet_audit.append({'sheet':name,'rows_imported':count,
                            'title':rows[0][1].get('A',''),
                            'selection_header':rows[1][1].get('A','')})
    table=pd.DataFrame(records)
    if table.empty:raise ValueError('No protein statistics imported')
    # Preserve all reported rows and statistics, even if not significant. Source
    # wording is not trusted as a guarantee that this is an unselected universe.
    panel=table[table.gene.isin(GENES)].copy()
    culture=[]
    for name,rows in workbook_values(paths['atlas_culture']).items():
        for n,row in rows[1:]:
            if not row.get('A','').isdigit() or not row.get('B',''):continue
            culture.append({'sheet':name,'source_row':n,'sample_number':row['A'],
                            'group':row['B'],'seeded_millions':float(row['C']),
                            'medium_mL':float(row['D']),'end_cells_millions':float(row['F']),
                            'author_correction_factor':float(row['G'])})
    return table,panel,{'sheets':sheet_audit,'reported_protein_rows':len(table),
                       'panel_rows':len(panel),'culture_normalization':culture,
                       'limitations':['Author summary comparisons; no reconstructed biological replicates.',
                                      'Reported SD and ratio counts are not assumed to be biological SEM or sample n.',
                                      'Missing proteins are not zero abundance; source selection and detectability limit coverage.',
                                      'Abundance ratios cannot identify secretory functional fraction or paracrine potency.',
                                      'Fibroblast and epithelial secretomes are separate contexts, not Huh-7 calibration.']}
