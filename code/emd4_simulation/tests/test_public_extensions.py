"""Contracts for public counts, source digitization and paired multimodal import."""
import gzip
import json
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from emd4_simulation.public_data.counts import frozen_median_ratio, washout_samples
from emd4_simulation.public_data.arsenite import axis_value, fit_luminescence
from emd4_simulation.public_data.single_cell import stream_selected_counts
from emd4_simulation.public_data.secretome import workbook_values
from emd4_simulation.public_data.inference import fit_candidate, predict
from emd4_simulation.public_data.models import simulate, Parameters, Protocol, observable_drivers
from emd4_simulation.tests.test_public_data import measured_fixture


def test_reference_normalization_never_uses_holdout_values():
    baseline=np.arange(10.,30.)
    counts=pd.DataFrame({'train1':baseline,'train2':2*baseline,'holdout':7*baseline})
    a,record_a=frozen_median_ratio(counts,['train1','train2'])
    counts['holdout']*=np.arange(1,21)
    b,record_b=frozen_median_ratio(counts,['train1','train2'])
    np.testing.assert_allclose(a[['train1','train2']],b[['train1','train2']])
    assert record_a['reference_geometric_means']==record_b['reference_geometric_means']
    assert record_a['reference_genes']==record_b['reference_genes']
    assert record_a['size_factors']['train1']==record_b['size_factors']['train1']


def test_size_factor_normalization_removes_known_library_multiplier():
    x=np.arange(10.,30.)
    f=pd.DataFrame({'a':x,'b':3*x,'c':9*x})
    normalized,audit=frozen_median_ratio(f,['a','b'])
    np.testing.assert_allclose(normalized.a,normalized.b)
    np.testing.assert_allclose(normalized.a,normalized.c)
    assert audit['size_factors']['c']/audit['size_factors']['a']==pytest.approx(9)


def test_bad_count_tables_fail_closed():
    f=pd.DataFrame({'a':np.ones(20),'b':np.ones(20)})
    with pytest.raises(ValueError,match='Too few'):frozen_median_ratio(f,['a'])
    f.loc[0,'a']=-1
    with pytest.raises(ValueError,match='nonnegative'):frozen_median_ratio(f,['a'])


def test_GEO_day_zero_is_after_24_hour_induction(tmp_path):
    path=tmp_path/'meta.soft.gz'
    text='\n'.join(['^SAMPLE = GSM1','!Sample_title = SDS_D0_1',
                    '^SAMPLE = GSM2','!Sample_title = SDS_D16_2',
                    '^SAMPLE = GSM3','!Sample_title = Untreated-control_1',
                    '^SAMPLE = GSM4','!Sample_title = Replicative Sen_1'])
    with gzip.open(path,'wt') as f:f.write(text)
    m=washout_samples({'timecourse_metadata':path}).set_index('sample')
    assert m.loc['GSM1','time_days']==1
    assert m.loc['GSM2','time_days']==17
    assert m.loc['GSM3','time_days']==0
    assert 'GSM4' not in m.index


def test_ADT_does_not_enter_RNA_QC_and_selected_counts_match(tmp_path):
    f=pd.DataFrame({'id':['a','b','ab'],'symbol':['CDKN1A','MT-CO1','CD44_TotalSeqB'],
                    'modality':['Gene Expression','Gene Expression','Antibody Capture']})
    p=tmp_path/'matrix.mtx.gz'
    data='%%MatrixMarket matrix coordinate integer general\n% test\n3 2 5\n1 1 10\n2 1 5\n3 1 1000\n1 2 20\n3 2 2000\n'
    with gzip.open(p,'wt') as out:out.write(data)
    selected,qc=stream_selected_counts(p,f,[0,2],chunk_lines=2)
    np.testing.assert_array_equal(selected,[[10,20],[1000,2000]])
    np.testing.assert_array_equal(qc['rna_UMI'],[15,20])
    np.testing.assert_array_equal(qc['detected_RNA_features'],[2,1])
    np.testing.assert_allclose(qc['mitochondrial_fraction'],[1/3,0])


def test_malformed_sparse_matrix_is_rejected(tmp_path):
    f=pd.DataFrame({'id':['a'],'symbol':['CDKN1A'],'modality':['Gene Expression']})
    p=tmp_path/'bad.gz'
    with gzip.open(p,'wt') as out:
        out.write('%%MatrixMarket matrix coordinate integer general\n1 1 1\n2 1 3\n')
    with pytest.raises(ValueError,match='Invalid count matrix'):stream_selected_counts(p,f,[0])


def test_duplicate_sparse_entries_cannot_inflate_detected_gene_QC(tmp_path):
    f=pd.DataFrame({'id':['a'],'symbol':['CDKN1A'],'modality':['Gene Expression']})
    p=tmp_path/'duplicate.gz'
    with gzip.open(p,'wt') as out:
        out.write('%%MatrixMarket matrix coordinate integer general\n1 1 2\n1 1 3\n1 1 4\n')
    # A chunk boundary must not hide a repeated coordinate.
    with pytest.raises(ValueError,match='unique, column-major'):
        stream_selected_counts(p,f,[0],chunk_lines=1)


def test_digitized_axis_respects_non_unit_scales():
    assert axis_value(727,802,636,60)==pytest.approx(27.1084337349)
    assert axis_value(789,828,648,40)==pytest.approx(8.6666666667)
    assert axis_value(164,233,25,3)==pytest.approx(0.9951923077)
    with pytest.raises(ValueError):axis_value(0,1,1,3)


def test_luminescence_recovery_fit_uses_observable_ratio():
    t=np.array([.5,1,2,3]);a=.7;r=.4
    y=-a*(-np.expm1(-r*t))/r
    fit=fit_luminescence(t,y,np.ones(4)*.1,recoverable=True)
    assert fit['apparent_recovery_per_day']==pytest.approx(r,rel=1e-4)
    assert fit['inhibition_per_day']==pytest.approx(a,rel=1e-4)


def make_workbook(path,cell):
    ns='http://schemas.openxmlformats.org/spreadsheetml/2006/main'
    with zipfile.ZipFile(path,'w') as z:
        z.writestr('xl/workbook.xml',f'<workbook xmlns="{ns}" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="Source" sheetId="1" r:id="rId1"/></sheets></workbook>')
        z.writestr('xl/_rels/workbook.xml.rels','<Relationships><Relationship Id="rId1" Target="worksheets/sheet1.xml"/></Relationships>')
        z.writestr('xl/worksheets/sheet1.xml',f'<worksheet xmlns="{ns}"><sheetData><row r="7">{cell}</row></sheetData></worksheet>')


def test_source_workbook_preserves_coordinates_and_cached_formula(tmp_path):
    path=tmp_path/'source.xlsx'
    make_workbook(path,'<c r="C7"><f>2+3</f><v>5</v></c>')
    assert workbook_values(path)=={'Source':[(7,{'C':'5'})]}


def test_source_workbook_rejects_uncomputed_formula(tmp_path):
    path=tmp_path/'source.xlsx'
    make_workbook(path,'<c r="C7"><f>2+3</f></c>')
    with pytest.raises(ValueError,match='without cached'):workbook_values(path)


def test_fitted_prediction_retains_medium_replacement_protocol():
    frame=measured_fixture()
    fit=fit_candidate(frame,'feedback',exposure_days=1,medium_changes=(1.,2.),starts=1)
    assert fit['medium_changes']==[1.,2.]
    # A subsequent prediction has to replay the fitted protocol rather than
    # silently keep a protein pool which was removed during training.
    params=Parameters(**fit['parameters'])
    from emd4_simulation.public_data.inference import channel_design
    state=simulate([4],params,Protocol(1,medium_changes=(1.,2.)))
    expected=np.einsum('nck,k->nc',channel_design(observable_drivers(state),fit['genes']),fit['coefficients'])
    np.testing.assert_allclose(predict(fit,[4]),expected)
