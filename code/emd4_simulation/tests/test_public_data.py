"""Scientific contracts for measured-data inference and intervention semantics."""
from dataclasses import replace
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile
import numpy as np
import pandas as pd
import pytest
from emd4_simulation.public_data.models import (
    Parameters, Protocol, simulate, donor_recipient, candidate_parameters,
    observable_drivers)
from emd4_simulation.public_data.adapters import (
    audit_washout_archive, platform_panel, validate_observations)
from emd4_simulation.public_data.inference import (
    noise_covariance, fit_candidate, score_holdout, channel_design, predict,
    compare_timecourse, endpoint_contrasts)
from emd4_simulation.public_data.sources import Source, fetch_source


def test_short_pulse_follows_analytic_damage_solution():
    p = replace(Parameters(), repair=1.7)
    duration = .0007
    t = np.array([2., 0., duration, 1.])  # unsorted outputs are supported
    y = simulate(t, p, Protocol(duration))
    peak = -np.expm1(-p.repair*duration)/p.repair
    expected = np.array([peak*np.exp(-p.repair*(2-duration)), 0, peak,
                         peak*np.exp(-p.repair*(1-duration))])
    np.testing.assert_allclose(y[:,0], expected, rtol=2e-5, atol=2e-9)


def test_medium_jump_applies_at_endpoint_and_rescue_after_wash():
    p = replace(Parameters(), secretion=0, feedback_gain=0)
    y = simulate([0, 1, 2], p, Protocol(0, medium_changes=(1,),
                                      rescue_at=1, rescue_protein=.2), initial_protein=1)
    np.testing.assert_allclose(y[:,6], [1., .2, .2*np.exp(-2+1)], rtol=1e-5)
    assert simulate([0], p, Protocol(0, medium_changes=(0,)), initial_protein=1)[0,6] == 0


def test_matched_clearance_does_not_leave_secreting_cells_after_removal():
    p = replace(Parameters(), arrest_loss=3, maturation_days=10, feedback_gain=0)
    matched = simulate([0, 8], p, Protocol(0), initial_arrest=.6)
    lag = simulate([0, 8], p, Protocol(0), initial_arrest=.6, structure='lag_memory')
    assert matched[-1,1:5].sum() < 1e-8
    assert observable_drivers(matched)[-1,2] < 1e-8
    assert observable_drivers(lag,'lag_memory')[-1,2] > .2
    assert lag[-1,6] > 100*matched[-1,6]


def test_durable_arrest_persists_without_feedback_or_secretion():
    p = candidate_parameters('durable_arrest', persistence=.002)
    y = simulate([0, 28], p, Protocol(0), initial_arrest=.5)
    assert y[-1,1:5].sum() > .47
    assert y[-1,3:5].sum() == 0
    assert y[-1,6] == 0


def test_one_way_paracrine_induces_arrest_without_a_secretory_loop():
    p = candidate_parameters('one_way_paracrine')
    t = np.array([0., 1., 3., 14.])
    for structure in ('matched_clearance','lag_memory'):
        y = simulate(t,p,Protocol(0),initial_protein=.5,structure=structure)
        np.testing.assert_allclose(y[:,6],.5*np.exp(-t),rtol=2e-4,atol=1e-8)
        assert y[-1,3:5].sum() > .1
        assert np.max(y[:,1:3]) == 0


def test_transfer_excludes_donor_cells_and_exposure():
    r = donor_recipient()
    y = r['states']
    assert r['transferred_protein'] > 0
    assert np.max(abs(y[:,0:3])) == 0
    assert y[-1,3:5].sum() > 0
    assert y[0,6] == r['transferred_protein']
    control = donor_recipient(scenario='control_medium')
    assert np.max(abs(control['states'])) == 0


def test_receptor_block_cannot_be_rescued_by_more_ligand():
    p=Parameters()
    y=simulate([0,1,3,10],p,Protocol(0,block_at=0,block_fraction=1,
                                   rescue_at=1,rescue_protein=100),initial_protein=1)
    assert np.max(y[:,3:5]) == 0
    assert y[1,6] > 100


def test_functional_fraction_and_response_scale_are_confounded():
    p=Parameters()
    a=simulate([0,1,3,7],p,Protocol(1))
    b=simulate([0,1,3,7],replace(p,functional_fraction=p.functional_fraction*1.5,
                              response_half=p.response_half*1.5),Protocol(1))
    np.testing.assert_allclose(a,b,atol=2e-8,rtol=1e-6)


def test_sampling_grid_does_not_change_event_solution():
    p=Parameters();protocol=Protocol(.7,medium_changes=(.7,2.3),block_at=1.2,block_fraction=.8)
    a=simulate([0,3],p,protocol)
    b=simulate(np.linspace(0,3,301),p,protocol)
    np.testing.assert_allclose(a[-1],b[-1],atol=1e-10)


def test_observation_covariance_retains_correlation_and_is_invertible():
    x=np.array([[0,0,1],[1,2,1],[2,4,1],[3,6,1]],float)
    cov=noise_covariance(x,[0,0,1,1])
    assert cov[0,1] > 0
    assert np.linalg.eigvalsh(cov).min() > 0
    diag=noise_covariance(x,[0,0,1,1],shrinkage=1)
    np.testing.assert_allclose(diag,np.diag(np.diag(diag)))


def measured_fixture():
    rng=np.random.default_rng(12)
    times=np.array([0.,.5,1.,2.,4.])
    genes=['CDKN1A','IL6','LMNB1']
    states=simulate(times,candidate_parameters('persistent_sasp'),Protocol(1))
    mean=np.einsum('nck,k->nc',channel_design(observable_drivers(states),genes),
                   [3.,1.2,.4,2.,1.1,7.,-.9])
    rows=[]
    for rep in range(3):
        y=mean+rng.normal(0,.04,mean.shape)
        for i,t in enumerate(times):
            for j,g in enumerate(genes):
                rows.append(dict(sample=f'{rep}_{i}',replicate=rep,condition='assay',
                                 time_days=t,gene=g,value=y[i,j]))
    return pd.DataFrame(rows)


def test_holdout_values_do_not_enter_training_or_noise_estimation():
    frame=measured_fixture();train=frame[frame.time_days<4];test=frame[frame.time_days==4]
    fit=fit_candidate(train,'persistent_sasp',exposure_days=1,starts=1)
    original=json.dumps(fit)
    changed=test.copy();changed.value+=20
    a=score_holdout(fit,test);b=score_holdout(fit,changed)
    assert b['holdout_whitened_MSE'] > 100*a['holdout_whitened_MSE']
    assert not set(fit['train_samples']) & set(a['holdout_samples'])
    assert json.dumps(fit)==original
    assert predict(fit,[4]).shape==(1,3)


def test_duplicate_observations_and_missing_channels_fail_closed():
    f=measured_fixture();f['study']='test';f['unit']='log2'
    validate_observations(f)
    with pytest.raises(ValueError,match='Duplicate'):
        validate_observations(pd.concat([f,f.iloc[:1]]))
    with pytest.raises(ValueError,match='Incomplete measured panel'):
        fit_candidate(f.iloc[1:],'feedback',exposure_days=1,starts=1)


def test_de_tables_are_quarantined_and_duplicates_detected(tmp_path):
    payload=b'baseMean\tlog2FoldChange\tlfcSE\tstat\tpvalue\tpadj\nGENE\t100\t2\t1\t2\t.05\t.2\n'
    path=tmp_path/'deposit.tar'
    with tarfile.open(path,'w') as tf:
        for i in range(2):
            data=gzip.compress(payload);m=tarfile.TarInfo(f'GSM{i}.txt.gz');m.size=len(data)
            tf.addfile(m,io.BytesIO(data))
    audit=audit_washout_archive(path)
    assert audit['DE_table_count']==2
    assert audit['unique_payloads']==1
    assert audit['status']=='excluded_pending_corrected_expression'


def test_ambiguous_probes_are_excluded_before_fitting(tmp_path):
    path=tmp_path/'family.soft.gz'
    text=('!platform_table_begin\nID\tgene_assignment\tmrna_assignment\n'
          'ok\tNR // IL8 // desc\tgene:ENSG00000169429\n'
          'bad\tNR // IL6 // desc /// NR // MMP3 // desc\tgene:ENSG00000136244\n'
          '!platform_table_end\n')
    with gzip.open(path,'wt') as f:f.write(text)
    probes,genes=platform_panel(path)
    assert probes=={'ok':'CXCL8'}
    assert genes=={'ENSG00000169429':'CXCL8'}


def test_cached_download_is_verified_and_tampering_is_rejected(tmp_path):
    s=Source('test','data.txt','https://example.org/data.txt','test','test','test')
    raw=tmp_path/'raw';raw.mkdir();path=raw/s.filename;path.write_text('original')
    sidecar=Path(str(path)+'.provenance.json')
    with pytest.raises(FileNotFoundError):fetch_source(s,tmp_path,offline=True)
    sidecar.write_text(json.dumps({'url':s.url,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}))
    assert fetch_source(s,tmp_path,offline=True)==path
    path.write_text('changed')
    with pytest.raises(ValueError,match='checksum mismatch'):fetch_source(s,tmp_path,offline=True)


def test_invalid_physical_protocol_rejected():
    with pytest.raises(ValueError):simulate([1],protocol=Protocol(1,block_fraction=1.1))
    with pytest.raises(ValueError):simulate([1],initial_arrest=1.1)
    with pytest.raises(ValueError):simulate([1],replace(Parameters(),maturation_days=0))
    with pytest.raises(ValueError):simulate([1],replace(Parameters(),repair=float('nan')))
    with pytest.raises(ValueError):simulate([1],replace(Parameters(),functional_fraction=1.1))


def test_forecast_baseline_is_computed_only_from_earlier_measurements():
    frame=measured_fixture()
    r=compare_timecourse(frame,exposure_days=1,structures=('matched_clearance',),starts=1)
    train=frame[frame.time_days<4]
    test=frame[frame.time_days==4].pivot(index='sample',columns='gene',values='value')
    last=train[train.time_days==2].groupby('gene').value.mean().loc[test.columns].to_numpy()
    expected=np.sqrt(np.mean((test.to_numpy()-last)**2))
    assert r['forecast_baselines']['last_observation']['holdout_RMSE_log2']==pytest.approx(expected)


def test_zero_tpm_does_not_create_a_zero_width_biological_interval():
    frame=pd.DataFrame([{'condition':c, 'gene':'IL6', 'value':0.}
                        for c in ('direct_DOX','control','DOX_CM','control_CM') for _ in range(5)])
    contrasts=endpoint_contrasts(frame)
    assert all(c['lo95'] is None and c['hi95'] is None for c in contrasts)
    assert all(c['all_values_zero'] for c in contrasts)
