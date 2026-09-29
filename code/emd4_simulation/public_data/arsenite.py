"""Auditable digitization of published aggregate figures, never invented replicates.

Coordinates refer to the pinned original downloaded JPEGs. Each entry records
axis anchors and a manually read mean marker/bar location. ±2 pixels is a
sensitivity range for reading the figure, not a confidence interval or raw-data
precision. SA-beta-gal fields are not assumed to be biological replicates.
"""
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from PIL import Image
from .sources import sha256


def axis_value(pixel, zero_pixel, reference_pixel, reference_value):
    if reference_pixel==zero_pixel or reference_value<=0:
        raise ValueError('Invalid digitization axis')
    return (zero_pixel-pixel)/(zero_pixel-reference_pixel)*reference_value


def digitized_observations(paths):
    rows=[]
    def add(fig,panel,analyte,condition,x,y,zero,ref,value,unit,post_h,
            sem_top=None,replication='aggregate_n3',dose=5.,assay_h=None):
        rows.append({'figure':fig,'panel':panel,'analyte':analyte,'condition':condition,
                     'x_pixel':x,'mean_y_pixel':y,'zero_y_pixel':zero,'reference_y_pixel':ref,
                     'reference_value':value,'unit':unit,'post_withdrawal_hours':post_h,
                     'assay_hours_after_reseeding':assay_h,'exposure_hours':72,'dose_uM':dose,
                     'mean':axis_value(y,zero,ref,value),
                     'digitization_half_width':abs(axis_value(y-2,zero,ref,value)-axis_value(y,zero,ref,value)),
                     'errorbar_top_pixel':sem_top,'reported_errorbar_size':abs(axis_value(sem_top,zero,ref,value)-axis_value(y,zero,ref,value)) if sem_top is not None else None,
                     'replication':replication,'source_sha256':sha256(paths[f'arsenite_fig{fig}'])})
    # Figure 5A: normalized luminescence, NOT direct cell counts. The first point
    # is fixed to 1 by the article's normalization, not an independently observed
    # biological baseline. The assay begins after harvesting/reseeding.
    for condition,ys,caps in [('control',[164,94,71,49],[None,90,66,41]),
                               ('arsenite',[164,160,172,177],[None,157,169,175])]:
        for x,h,y,cap in zip([81,146,224,302],[4,24,48,72],ys,caps):
            add(5,'A','relative_luminescence',condition,x,y,233,25,3.,
                'relative_to_4_hour_assay_signal',None,cap,assay_h=h,
                dose=0 if condition=='control' else 5)
            if h==4:rows[-1]['mean']=1.
    # SA-beta-gal: the paper reports five representative microscopic fields;
    # field count is not independent culture n. Controls differ strongly by phase.
    add(2,'D','SA_beta_gal','control',558,727,802,636,60.,'percent_positive_fields',0,711,'five_microscopy_fields',dose=0)
    add(2,'D','SA_beta_gal','arsenite',609,678,802,636,60.,'percent_positive_fields',0,672,'five_microscopy_fields')
    add(5,'D','SA_beta_gal','control',568,789,828,648,40.,'percent_positive_fields',168,781,'five_microscopy_fields',dose=0)
    add(5,'D','SA_beta_gal','arsenite',628,707,828,648,40.,'percent_positive_fields',168,695,'five_microscopy_fields')
    for analyte,points in [('CDKN1A',[(0,91,228),(2,127,214),(5,162,154),(10,199,122)]),
                           ('LMNB1',[(0,291,113),(2,327,144),(5,363,213),(10,398,229)])]:
        for dose,x,y in points:
            add(2,'A',analyte,'control' if dose==0 else 'arsenite',x,y,
                230 if analyte=='CDKN1A' else 231,72,10. if analyte=='CDKN1A' else 1.,
                'relative_mRNA_to_18S',0,dose=dose)
    for analyte,points,maximum in [('CDKN1A',[(516,230),(565,103)],3.),('LMNB1',[(662,111),(714,213)],2.)]:
        for condition,(x,y) in zip(['control','arsenite'],points):
            add(5,'B',analyte,condition,x,y,232,70,maximum,'relative_mRNA_to_RPLP1',100,
                dose=0 if condition=='control' else 5)
    # Figure 6: values close to the zero axis will receive no fold-change estimate.
    panels=[('MMP1',[(84,240),(146,95)],241,45,2.5),
            ('MMP3',[(268,239),(331,118)],240,41,5.),
            ('MMP10',[(468,239),(528,87)],241,45,2.),
            ('GDF15',[(654,208),(715,122)],240,44,2.),
            ('SERPINE1',[(188,514),(250,408)],553,352,2.),
            ('VEGFA',[(363,409),(425,381)],553,351,1.2),
            ('IL6',[(542,552),(605,392)],553,352,2.5)]
    for analyte,points,zero,ref,value in panels:
        for condition,(x,y) in zip(['control','arsenite'],points):
            add(6,'RNA',analyte,condition,x,y,zero,ref,value,'relative_mRNA_to_RPLP1',100,
                dose=0 if condition=='control' else 5)
    frame=pd.DataFrame(rows)
    for fig in (2,5,6):
        width,height=Image.open(paths[f'arsenite_fig{fig}']).size
        f=frame[frame.figure==fig]
        if not ((f.x_pixel>=0)&(f.x_pixel<width)&(f.mean_y_pixel>=0)&(f.mean_y_pixel<height)).all():
            raise ValueError('Digitization coordinates do not match source image dimensions')
    return frame


def fit_luminescence(time_days, log_ratio, sigma, *, recoverable):
    t=np.asarray(time_days,float);y=np.asarray(log_ratio,float)
    def prediction(theta):
        inhibition=theta[0]
        recovery=theta[1] if recoverable else 0.
        integral=t if recovery<1e-8 else -np.expm1(-recovery*t)/recovery
        return -inhibition*integral
    result=least_squares(lambda theta:(prediction(theta)-y)/sigma,
                         [.6,.3] if recoverable else [.6],
                         bounds=([0.,0.],[10.,4.]) if recoverable else ([0.],[10.]))
    return {'inhibition_per_day':float(result.x[0]),
            'apparent_recovery_per_day':float(result.x[1]) if recoverable else 0.,
            'weighted_residual_SSE':float(np.sum(result.fun**2)),
            'converged':bool(result.success),'predictions_log_ratio':prediction(result.x).tolist()}


def analyze_arsenite(paths, *, seed=3501, draws=400):
    frame=digitized_observations(paths)
    excess=[]
    for fig in (2,5):
        values=frame[(frame.figure==fig)&(frame.analyte=='SA_beta_gal')].set_index('condition')
        difference=float(values.loc['arsenite','mean']-values.loc['control','mean'])
        width=float(values.digitization_half_width.sum())
        excess.append({'post_withdrawal_hours':0 if fig==2 else 168,
                       'control_percent':float(values.loc['control','mean']),
                       'treated_percent':float(values.loc['arsenite','mean']),
                       'excess_percentage_points':difference,
                       'pixel_sensitivity_range':[difference-width,difference+width],
                       'biological_CI':None})
    pcr=[]
    for (fig,analyte),part in frame[(frame.figure.isin([5,6]))&frame.unit.eq('relative_mRNA_to_RPLP1')].groupby(['figure','analyte']):
        v=part.set_index('condition');c=float(v.loc['control','mean']);a=float(v.loc['arsenite','mean'])
        cwidth=float(v.loc['control','digitization_half_width']);awidth=float(v.loc['arsenite','digitization_half_width'])
        pcr.append({'figure':int(fig),'analyte':analyte,'control':c,'treated':a,
                    'log2_ratio':float(np.log2(a/c)) if c>cwidth and a>awidth else None,
                    'control_resolved_above_axis':c>cwidth,
                    'interpretation':'100-hour post-withdrawal RNA; not secreted protein activity'})
    lum=frame[frame.analyte=='relative_luminescence']
    ctrl=lum[lum.condition=='control'].sort_values('assay_hours_after_reseeding')
    treated=lum[lum.condition=='arsenite'].sort_values('assay_hours_after_reseeding')
    t=(ctrl.assay_hours_after_reseeding.to_numpy()[1:]-4)/24
    c,a=ctrl['mean'].to_numpy()[1:],treated['mean'].to_numpy()[1:]
    cs,ass=ctrl.reported_errorbar_size.to_numpy()[1:],treated.reported_errorbar_size.to_numpy()[1:]
    # Unknown covariance from shared normalization is not fabricated. Report
    # descriptive weighted fits and digitization sensitivity, no biological CI.
    sigma=np.sqrt((cs/c)**2+(ass/a)**2)
    y=np.log(a/c)
    fits={name:fit_luminescence(t,y,sigma,recoverable=recoverable)
          for name,recoverable in [('constant_inhibition',False),('relaxing_inhibition',True)]}
    rng=np.random.default_rng(seed);rates=[]
    cw=ctrl.digitization_half_width.to_numpy()[1:];aw=treated.digitization_half_width.to_numpy()[1:]
    for _ in range(draws):
        cd=c+rng.uniform(-cw,cw);ad=a+rng.uniform(-aw,aw)
        rates.append(fit_luminescence(t,np.log(ad/cd),sigma,recoverable=True)['apparent_recovery_per_day'])
    return frame,{'SA_beta_gal_control_adjusted':excess,'withdrawal_RNA':pcr,
                  'luminescence':{'time_since_4_hour_reference_days':t.tolist(),
                                  'observed_log_treated_over_control':y.tolist(),'fits':fits,
                                  'recovery_rate_pixel_sensitivity_quantiles':np.quantile(rates,[.05,.5,.95]).tolist(),
                                  'seed':seed,'draws':draws,'biological_CI':None},
                  'limitations':['Manual aggregate-figure digitization, ±2 pixels; inspect coordinate audit plots.',
                                 'Luminescence reflects viability/metabolic signal, not a direct cell-count assay.',
                                 'Its apparent recovery parameter is not the latent senescence clearance rate.',
                                 'Three post-reference times cannot identify paracrine feedback.',
                                 '18S-normalized exposure PCR and RPLP1-normalized withdrawal PCR are never pooled into a decay curve.',
                                 'Microscopy fields are not assumed to be independent biological replicates.',
                                 'Pixel sensitivity excludes experimental variability and is not a statistical confidence interval.']}
