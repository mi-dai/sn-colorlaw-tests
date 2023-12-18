from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table,vstack,unique,Row
# import sfdmap
from sfdmap2 import sfdmap
import sncosmo
import astropy.units as units
import errno
import os
import random 
from scipy.optimize import minimize
import pandas as pd
from scipy.interpolate import interp1d
import pickle
from astropy.coordinates import SkyCoord
from astropy import units as u

def plot_sed(sedmodels, day=[-10.,0.,20.], wave=[3000.,7000.], 
             npoints=50,color=None,labels=None,scale=1.,z=0.,restframe=True):
    if color == None:
        color = plt.cm.jet(np.linspace(0,1,len(sedmodels)))
    for i,sed in enumerate(sedmodels):
        ts = day
        if len(wave) == 2:
            w = np.linspace(wave[0],wave[1],npoints)
        else:
            w = wave
        for t in ts:
            label = 't={0:.2f}'.format(t)
            if labels:
                label = label + '(' + labels[i] +')'
            else:
                label = label + '(' + str(i) + ')'
            if isinstance(restframe, list) and restframe[i] is False:
                z = 0.
            plt.plot(w*(1.+z),sed.flux(t,w)/scale/(1.+z),label=label,color=color[i],alpha=1)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.grid(True)

def register_bands_for_sncosmo():
    #register bands in sncosmo
    #SDSS bands
    ffolder = '/Users/mi/Work/tape_explore/data/filters/'
    # ffolder = os.path.join(os.environ['SEDFIT_DIR'],'data/filters/')
    for b in ['u','g','r','i','z']:
        band = sncosmo.get_bandpass('sdss'+b)
        band.name = 'SDSS-'+b
        sncosmo.register(band,force=True)
    #CSP bands
    for b in ['u','g','r','i','B','V','V0','V1']:   
        if b == 'V0':
            b1 = 'V3014'
        elif b == 'V1':
            b1 = 'V3009'
        elif b == 'V':
            b1 = 'V9844'
        else:
            b1 = b
        band = sncosmo.get_bandpass('csp'+b1.lower())
        band.name = 'CSP-'+b
        sncosmo.register(band,force=True)
    for b in ['Y','J','H','Jrc2','Ydw','Jdw','Hdw']:
        band = sncosmo.read_bandpass(ffolder+'CSP/CSP_'+b+'.dat')
        band.name = 'CSP-'+b
        sncosmo.register(band,force=True)
    #Standard bands(CfA1,CfA2,Hamuy96)
    for b in ['U','B','V','R','I']:
        band = sncosmo.get_bandpass('standard::'+b.lower())
        band.name = 'CfA1-'+b
        sncosmo.register(band,force=True)
        band = sncosmo.get_bandpass('standard::'+b.lower())
        band.name = 'CfA2-'+b
        sncosmo.register(band,force=True)
        band = sncosmo.get_bandpass('standard::'+b.lower())
        band.name = 'Hamuy96-'+b
        sncosmo.register(band,force=True)
        band = sncosmo.get_bandpass('standard::'+b.lower())
        band.name = 'Standard-'+b
        sncosmo.register(band,force=True)
    #CfA3, CfA4 - Keplercam, 4Shooter
    for b in ['U','B','V','R','I']:
        inst = '4shooter2'
        if b in ["U","u'"]:
            b1 = 'Us'
            band = sncosmo.get_bandpass(inst+"::"+b1.lower())
        else:
            band = sncosmo.get_bandpass(inst+"::"+b.lower())
        band.name = 'CfA3-'+b.upper()+inst[:-1]
        sncosmo.register(band,force=True)
    for b in ['U','B','V',"r'","i'","u'"]:
        inst = 'Keplercam'
        if b in ["U","u'"]:
            b1 = 'Us'
            band = sncosmo.get_bandpass(inst+"::"+b1.lower())
            band2 = sncosmo.get_bandpass(inst+"::"+b1.lower())
        else:
            band = sncosmo.get_bandpass(inst+"::"+b[0:1].lower())
            band2 = sncosmo.get_bandpass(inst+"::"+b[0:1].lower())           
        band.name = 'CfA3-'+b+inst
        sncosmo.register(band,force=True)
#         band2 = band
        band2.name = 'CfA4-'+b+inst
        sncosmo.register(band2,force=True)
    #SNLS - Megacam, !!test only, focal radius = 0., focal radii need to be calculated sn by sn
    for b in ['g','r','i','z']:
        inst = 'megacampsf'
        band = sncosmo.get_bandpass(inst+"::"+b, 8.)
        band.name = 'SNLS-'+b                
        sncosmo.register(band,force=True)
    #CfAIR2
    for b in ['H','J','K_s']:
        band = sncosmo.read_bandpass(ffolder+'CfAIR2/'+b+'_RSR.txt',wave_unit=units.um)
        band.name = 'CfAIR2-'+b
        sncosmo.register(band,force=True)
    #Swift
    for b in ['U','B','V','UVW1','UVW2','UVM2']:
        band = sncosmo.read_bandpass(ffolder+'Swift/Swift_UVOT.'+b+'.dat')
        band.name = 'Swift-'+b
        sncosmo.register(band,force=True)
    #Pan-STARRS, Foundation
    bands = Table.read(ffolder+'/Pan-STARRS/Pan-STARRS_apj425122t3_mrt.txt',format='ascii.no_header')
    cols = ['wave','open','g','r','i','z','y','w','Aero','Ray','Mol']
    for i,col in enumerate(cols):
        bands.rename_column('col'+str(i+1),col)
    for b in ['g','r','i','z']:
        band = sncosmo.Bandpass(bands['wave'],bands[b],wave_unit=units.nm)
        band.name = 'Foundation-'+b
        sncosmo.register(band,force=True)
        band2 = sncosmo.Bandpass(bands['wave'],bands[b],wave_unit=units.nm)
        band2.name = 'PanSTARRS-'+b
        sncosmo.register(band2,force=True)
    #DES
    for b in ['g','r','i','z']:
        band = sncosmo.get_bandpass('des'+b)
        band.name = 'DES-'+b
        sncosmo.register(band,force=True)
    #LOSS
#     bandpass files are truncated at 9999A; new filter files available (but still truncated *** were removed)
#     for b in ['B','V','R','I']:
#         for inst in ['KAIT1','KAIT2','KAIT3','KAIT4','NICKEL']:
#             bandtable = Table.read(ffolder+'LOSS/old/'+b+'_'+inst.lower()+'_shifted.txt',format='ascii.no_header')
#             bandtable = bandtable[bandtable['col1'].astype(str) != '******']
#             band = sncosmo.Bandpass(bandtable['col1'],bandtable['col2'])
#             band.name = 'LOSS-'+b+inst
#             sncosmo.register(band,force=True)
    for b in ['B','V','R','I']:
        band = sncosmo.get_bandpass('standard::'+b.lower())
        band.name = 'LOSS-'+b
        sncosmo.register(band,force=True)

def get_refmag(survey,band): ## !!need to double check all the values here are correct
    # print survey,band
    if survey == 'SNLS': #Guy2010
        magsys = 'bd17'
        tmref = Table([['u','g','r','i','z'],
                      [9.7688,9.6906,9.2183,8.9142,8.7736]],
                      names=('band', 'mref'))
        mref = tmref['mref'][tmref['band']==band][0]
        return mref,magsys

    if survey == 'CSP': #http://csp.obs.carnegiescience.edu/data/filters
        if band[0:1] in ['u','g','r','i']:
            magsys = 'bd17'
            tmref = Table([['u','g','r','i'],
                          [10.518,9.644,9.352,9.25]],
                          names=('band', 'mref'))
            mref = tmref['mref'][tmref['band']==band[0:1]][0]
        elif band[0:1] in ['B','V']:
            magsys = 'vega'
            tmref = Table([['B','V','V0','V1'],
                          [0.03,0.0096,0.0096,0.0145]],
                          names=('band', 'mref'))
            mref = tmref['mref'][tmref['band']==band[0:1]][0]
        else:
            magsys = 'vega'
            mref = 0.
        return mref,magsys

    if survey in ['Pan-STARRS','Foundation','PanSTARRS']:
        return 0.,'ab'

    if survey == 'SDSS': #Sako 2014
        magsys = 'ab'
        tmref = Table([['u','g','r','i','z'],
                      [0.0679,-0.0203,-0.0049,-0.0178,-0.0102]],
                      names=('band', 'mref'))
        mref = tmref['mref'][tmref['band']==band.lower()][0]
        return mref,magsys

    if survey in ['CfA3','CfA4']:
        magsys = 'bd17'
        if band.startswith(('U4','B4','V4','R4','I4')):
            tmref = Table([['U4','B4','V4','R4','I4'],
                          [9.693,9.8744,9.4789,9.1554,8.8506]],
                          names=('band', 'mref'))
            mref = tmref['mref'][[band.startswith(x) for x in tmref['band']]][0]   
        elif band.startswith(('UK','BK','VK',"r'K","i'K","u'K")):
            # u' = U + 0.854 (CfA4,Hicken2012)
            tmref = Table([['UK','BK','VK',"r'K","i'K","u'K"],
                          [9.6922,9.8803,9.4722,9.3524,9.2542,10.5462]],
                          names=('band', 'mref'))
            mref = tmref['mref'][[band.startswith(x) for x in tmref['band']]][0]   
        return mref,magsys

    # if survey == 'CfA4':
    #     magsys = 'bd17'
    #     tmref = Table([['U','B','V',"r'","i'"],
    #                   [9.6922,9.8803,9.4722,9.3524,9.2542]],
    #                   names=('band', 'mref'))
    #     mref = tmref['mref'][tmref['band']==band][0]        
    #     return mref,magsys

    if survey in ['CfA1','CfA2','Hamuy96','Standard','LOSS']:
        magsys = 'bd17'
        tmref = Table([['U','B','V',"R","I"],
                      [9.724,9.907,9.464,9.166,8.846]],
                      names=('band', 'mref'))
        mref = tmref['mref'][tmref['band']==band][0]        
        return mref,magsys

#     if survey == 'LOSS':
#         return 0.,'vega'

    if survey == 'Swift':
        return 0.,'vega'

    if survey == 'CfAIR2':
        return 0.,'vega'

    else:
        return 0.,'ab'


def fit_lcparams(lcdata,metadata,modelsource='salt2',modelpars=['t0','x0','x1','c'],
                 mpbounds=None,modelcov=True,snlist=None,register_bands=False,
                 write_to_file=True,outfile='data/lcparams.dat',usebands='all',
                 combine_filt_inst=True,rescols=None,**kwargs):

    if register_bands:
        register_bands_for_sncosmo()
    if mpbounds is None:
        mpbounds = {}

    fit_result = Table()
    dustmap = sfdmap.SFDMap("/Users/mi/sfddata-master")
    model = sncosmo.Model(source=modelsource,
                          effects=[sncosmo.F99Dust()],
                          effect_names=['mw'],
                          effect_frames=['obs'])
    print("Base model info:")
    print(model)
    if snlist is None:
        snlist = list(set(lcdata['Name_upper']))
        snlist = np.sort(np.array(snlist))

    if write_to_file:
        f = open(outfile,'w')

    header=True
    for i,sn in enumerate(snlist):
        zp = 27.5
        lc = Table.from_pandas(lcdata[lcdata['Name_upper'] == sn])
        meta = Table.from_pandas(metadata[metadata['Name_upper'] == sn])

        lc['flux'] = np.power(10.,-0.4*(lc['Mag']-zp))
        lc['flux_err'] = np.absolute(0.921*lc['flux']*lc['MagErr'])
        lc['zp'] = zp
        lc['zpsys'] = 'ab'.ljust(10)

        if combine_filt_inst:
            a1 = lc['Filter']
            c = ("-" * len(lcdata))
            a2 = lc['Survey']
            lc['Instrument'].fill_value = ""
            inst = lc['Instrument'].filled()
            lc['Filter'] = list(map(''.join, zip(a2, c, a1,inst)))
        lc = lc[lc.filled(-99.)['flux']>0.]
        if len(meta) == 1:
            z = meta[0]['z_helio']
            zsource = meta[0]['Sample']
        else:
            z = meta[meta['Sample'] != 'Swift']['z_helio'][0]
            zsource = meta[meta['Sample'] != 'Swift']['Sample'][0]
        if not np.isfinite(z):
            z = meta[0]['z-osc']
            zsource = 'osc'
        if np.any(np.isfinite(meta['RA'])) and np.any(np.isfinite(meta['DEC'])):
            ra = meta[np.isfinite(meta['RA'])][0]['RA']
            dec = meta[np.isfinite(meta['DEC'])][0]['DEC']
        else:
            ra = meta[0]['ra-osc']
            dec = meta[0]['dec-osc']
            c = SkyCoord(ra=ra, dec=dec, unit=(u.hourangle,u.degree))
            ra = c.ra.degree
            dec = c.dec.degree
        mwebv = dustmap.ebv(ra,dec)
        print(i, "fitting ", sn, "z=", z, "zsource=", zsource, "mwebv=", mwebv)
        model.set(z=z,mwebv=mwebv)
        # surveys = meta[0]['Surveys'].split("::")
        surveys = [meta[0]['Sample']]
        for s in surveys:
            try:
                print('Survey = ', s)
                lc0 = lc[lc['Survey'] == s]

                if usebands != 'all':
                    lc0 = lc0[np.array([x.split('-')[-1][0:1] in list(usebands) for x in lc0['Filter']])]
                    if len(lc0) == 0:
                        print("No data in selected bands:",usebands)
                        continue

                for b in set(lc0['Filter']):
                    # mref,magsys = get_refmag(s,b.split('-')[1])
                    mref,magsys = get_refmag(s,b[b.find(s)+len(s):].split('-')[-1])
                    lc0['zp'][lc0['Filter']==b] = lc0['zp'][lc0['Filter']==b]-mref
                    lc0['zpsys'][lc0['Filter']==b] = magsys

                if s == 'SNLS':
                    xf = meta[0]['xf']
                    yf = meta[0]['yf']
                    rf = np.sqrt(xf**2 + yf**2)
                    print("calculate rf and re-register SNLS filters: rf= ",rf)
                    for b in ['g','r','i','z']:
                        inst = 'megacampsf'
                        band = sncosmo.get_bandpass(inst+"::"+b, rf)
                        band.name = 'SNLS-'+b                
                        sncosmo.register(band,force=True)

                result, fitted_model = sncosmo.fit_lc(lc0,model,modelpars,
                                                      modelcov=modelcov,
                                                      bounds=mpbounds.copy(),**kwargs)
                # print(result)
                res = []
                # res.append(sncosmo.flatten_result(result))
                res.append(_flatten_result(result))
                res = Table(res)
                res['Name'] = sn.ljust(30)
                res['Survey'] = s.ljust(20)
                res['ErrorType'] = 'None'.ljust(50)
                res['ErrorMesg'] = 'None'.ljust(50)
                res['bands'] = ''.join(sorted([x.split('-')[-1][0:1] for x in set(lc0[result.data_mask]['Filter'])])).ljust(20)
                res['bands2'] = ''.join(sorted([x.split('-')[-1][0:1] for x in set(lc0['Filter'])])).ljust(20)
                if len(lc0[result.data_mask]['MJD'])>0:
                    res['last_fitmjd'] = np.max(lc0[result.data_mask]['MJD'])
                    res['first_fitmjd'] = np.min(lc0[result.data_mask]['MJD'])
                else:
                    res['last_fitmjd'] = -99.
                    res['first_fitmjd'] = -99.
                # print res
                if write_to_file:
                    if header:
                        res.write(f,format='ascii')
                        header = False
                    else:
                        res.write(f,format='ascii.no_header')
                    f.flush()
                    os.fsync(f.fileno())
                fit_result = vstack([fit_result,res],join_type='outer')
                # print fit_result
            except Exception as e: 
                print('An error occured:')
                print(type(e))
                print(e)
                if len(fit_result) > 0:
                    extra_cols = list([sn,s,str(type(e)),str(e),'-99','-99',-99.,-99.])
                    res = list([-99.9]*(len(fit_result.colnames)-len(extra_cols))) + extra_cols
                    fit_result.add_row(res)   
                    if write_to_file:
                        fit_result[-1:].write(f,format='ascii.no_header')  
                        f.flush()
                        os.fsync(f.fileno())
                else:
                    fit_result = Table.from_pandas(pd.DataFrame(rescols))
                    fit_result['Name'] = sn.ljust(30)
    if write_to_file:
        f.close()

    fit_result = Table.to_pandas(fit_result)
    # fit_result['Name'] = [x.strip() for x in fit_result['Name']]
    # fit_result['Survey'] = [x.strip() for x in fit_result['Survey']]

    return fit_result


def salt2_params(lcdata,metadata,snlist=None,modelcov=True,register_bands=False,
                 write_to_file=True,outfile='data/salt2params.dat',usebands='all',
                 salt2name ='salt2',combine_filt_inst=True,**kwargs):
    modelpars = ['t0','x0','x1','c']
    mpbounds = {'x0':(0,1.),
              'x1':(-5.,5.),
              'c':(-3.,3.)}
    result = fit_lcparams(lcdata,metadata,modelsource=salt2name,modelpars=modelpars,
                 mpbounds=mpbounds,modelcov=modelcov,snlist=snlist,register_bands=register_bands,
                 write_to_file=write_to_file,outfile=outfile,usebands=usebands,
                 combine_filt_inst=combine_filt_inst,**kwargs)
    return result


class GPResults(object):
    def __init__(self, meanlc, meanlc_cov, peakmjds, filters,gppars=None,chisq=None,realizations=None):
        self.meanlc = meanlc
        self.meanlc_cov = meanlc_cov
        self.peakmjds = peakmjds
        self.filters = filters
        self.chisq = chisq
        self.realizations = realizations
        self.gppars = gppars



def fit_gp(sn,writepkl=False,outfolder='outputs/testgp',method='george2d',printlog=False,**kwargs):
    # register_bands_for_sncosmo()
    if printlog:
        print("Fitting gp using",method,sn.reset_index()['Name_upper'].iloc[0])
    if method == 'george2d':
        res = _fit_gp_george_2d(sn,printlog=printlog,**kwargs)
    elif method == 'pymc':
        res = _fit_gp_pymc(sn,printlog=printlog,**kwargs)
    elif method == 'george1d':
        res = _fit_gp_george_1d(sn,printlog=printlog,**kwargs)
    else:
        raise ValueError("unknown method")
    if writepkl:
        name = sn['Name_upper'].iloc[0]
        pickle_dict = {'snname':name,'gpres':res}
        pickle_out = open(os.path.join(outfolder,"gp_"+name+".pickle"),"wb")
        pickle.dump(pickle_dict, pickle_out)
        pickle_out.close()

    return res

def _fit_gp_george_1d(sn,plot=True,nrandomsamples=2, printlog=False,**kwargs):
    # Using george
    import george
    from george import kernels

    sn = pd.DataFrame(sn)

    meanlc = []
    lcvar = []
    samples = []
    peakmjds = []
    filters = []
    sample_funcs = []
    gppars = []
    chisq = []

    if plot:
        fig = plt.figure(figsize=(24,6*(len(sn['Filter'].unique())//6+1)))

    for i,f in enumerate(sn['Filter'].unique()):

        sn_f = sn.set_index('Filter').loc[f]
        y = np.array(sn_f['Mag'])
        yerr = np.array(sn_f['MagErr'])
        x = np.array(sn_f['MJD'])

        idx = np.array(y>0)    

        if np.sum(idx) == 0:
            continue

        y = y[idx]
        yerr = yerr[idx]
        x = x[idx]

        amp = np.var(y)
        lengthscale = 10.
        kernel = amp * kernels.Matern32Kernel(lengthscale)
        gp = george.GP(kernel,mean=np.median(y))
        gp.compute(x, yerr)

        if printlog:
            print("Initial ln-likelihood: {0:.2f}".format(gp.log_likelihood(y)))
        from scipy.optimize import minimize

        def neg_ln_like(p):
            gp.set_parameter_vector(p)
            return -gp.log_likelihood(y)

        def grad_neg_ln_like(p):
            gp.set_parameter_vector(p)
            return -gp.grad_log_likelihood(y)

        result = minimize(neg_ln_like, gp.get_parameter_vector(), jac=grad_neg_ln_like)
        if printlog:
            print(result)

        gp.set_parameter_vector(result.x)
        if printlog:
            print("\nFinal ln-likelihood: {0:.2f}".format(gp.log_likelihood(y)))
        
        xa = np.linspace(x.min(),x.max(),100)
        pred, pred_var = gp.predict(y, xa, return_var=True)

        ##estimate peaks
        meanf = interp1d(xa,pred,fill_value="extrapolate")
        varf = interp1d(xa,pred_var,fill_value="extrapolate")
        peakres = minimize(meanf,x[np.argmin(y)])
        pmjd = peakres.x[0]
        if pmjd < x.min() or pmjd > x.max():
            if printlog:
                print("No data before or after the peak for band", f)
            pmjd = -99.
        peakmjds.append(pmjd)
        
        filters.append(f)
        meanlc.append(meanf)
        lcvar.append(varf)
        gppars.append(gp.get_parameter_dict(include_frozen=True)) 
        chisq.append(gp.log_likelihood(y))

        if nrandomsamples>0:
            samples_f = gp.sample_conditional(y, xa, size=nrandomsamples)
            samples.append(samples_f)
            sample_func_f = []
            for j in range(nrandomsamples):
                finterp = interp1d(xa,samples_f[j],fill_value="extrapolate")
                sample_func_f.append(finterp)
            sample_funcs.append(sample_func_f)
        else:
            sample_funcs = None

        if plot:
            fig.add_subplot(len(sn['Filter'].unique())//6+1,6,i+1)
            for j in range(nrandomsamples):
                plt.plot(xa,samples_f[j],'-',lw=0.5,color='g',alpha=0.5)
            plt.fill_between(xa, pred - np.sqrt(pred_var), pred + np.sqrt(pred_var),
                            color="k", alpha=0.2)
            plt.plot(xa, pred, "k", lw=1.5, alpha=0.5)
            plt.errorbar(x, y, yerr=yerr, fmt=".k", capsize=0)
            plt.ylim(plt.ylim()[::-1])
            plt.title(f)

    return GPResults(meanlc,lcvar,peakmjds,filters,
                     gppars=gppars,realizations=sample_funcs,
                     chisq=chisq)


def _fit_gp_george_2d(sn,plot=True,nrandomsamples=2,printlog=False,**kwargs):
    import george
    from george import kernels

    # 2d GP using george
    sn = pd.DataFrame(sn)

    sn['effwave'] = 0.
    for f in sn['Filter'].unique():
        band = sncosmo.get_bandpass(f)
        sn.loc[sn['Filter'] == f,'effwave'] = band.wave_eff

    # for s in sn['Survey'].unique():
    # sn_s = sn.loc[sn['Survey']==s,:]
    sn_s = sn.copy()
    y = np.array(sn_s['Mag'])
    yerr = np.array(sn_s['MagErr'])
    x = np.vstack([sn_s['MJD'],sn_s['effwave']]).T

    amp = np.var(y)
    tlengthscale = 10.
    wlengthscale = 5.e7
    kernel = amp * kernels.Matern32Kernel(ndim=2,metric=[tlengthscale,wlengthscale])
    # kernel = amp * kernels.ExpSquaredKernel(ndim=2,metric=[tlengthscale,wlengthscale])
    kernel.freeze_parameter("k2:metric:log_M_1_1")
    gp = george.GP(kernel,mean=np.median(y))
    gp.compute(x, yerr)

    if printlog:
        print("Initial ln-likelihood: {0:.2f}".format(gp.log_likelihood(y)))

    def neg_ln_like(p):
        gp.set_parameter_vector(p)
        return -gp.log_likelihood(y)

    def grad_neg_ln_like(p):
        gp.set_parameter_vector(p)
        return -gp.grad_log_likelihood(y)

    result = minimize(neg_ln_like, gp.get_parameter_vector(), jac=grad_neg_ln_like)
    # print(result)

    gp.set_parameter_vector(result.x)
    if printlog:
        print("Final ln-likelihood: {0:.2f}".format(gp.log_likelihood(y)))

    meanlc = []
    lcvar = []
    samples = []
    peakmjds = []
    filters = []
    sample_funcs = []

    if plot:
        fig = plt.figure(figsize=(24,6*(len(sn['Filter'].unique())//6+1)))
        i = 0

    for f in sn_s['Filter'].unique():
        effw = sncosmo.get_bandpass(f).wave_eff
        idx = np.array(sn_s['Filter'] == f)
        xa0 = np.linspace(x[:,0].min(),x[:,0].max(),100)
        xa = np.vstack([xa0,[effw]*100]).T
        pred_f, pred_var_f = gp.predict(y, xa, return_var=True)

        if nrandomsamples>0:
            samples_f = gp.sample_conditional(y, xa, size=nrandomsamples)
            samples.append(samples_f)
            sample_func_f = []
            for j in range(nrandomsamples):
                finterp = interp1d(xa0,samples_f[j],fill_value="extrapolate")
                sample_func_f.append(finterp)
            sample_funcs.append(sample_func_f)
        else:
            sample_funcs = None
        
        ##estimate peak in each band
        meanf = interp1d(xa0,pred_f,fill_value="extrapolate")
        varf = interp1d(xa0,pred_var_f,fill_value="extrapolate")
        peakres = minimize(meanf,x[np.argmin(y[idx]),0])
        pmjd = peakres.x[0]
        if pmjd < x[idx,0].min() or pmjd > x[idx,0].max():
            if printlog:
                print("No data before or after the peak for band", f)
            pmjd = -99.
        peakmjds.append(pmjd)
        
        filters.append(f)
        meanlc.append(meanf)
        lcvar.append(varf)

        if plot:
            fig.add_subplot(len(sn['Filter'].unique())//6+1,6,i+1)
            for j in range(nrandomsamples):
                plt.plot(xa[:,0],samples_f[j,:],'-',lw=0.5,color='g',alpha=0.5)
            plt.fill_between(xa[:,0], pred_f - np.sqrt(pred_var_f), pred_f + np.sqrt(pred_var_f),
                            color="k", alpha=0.2)
            plt.plot(xa[:,0], pred_f, "k", lw=1.5, alpha=0.5)
            plt.errorbar(x[idx,0], y[idx], yerr=yerr[idx], fmt=".k", capsize=0)
            plt.ylim(plt.ylim()[::-1])
            plt.title(f)
            i += 1

    gppars = gp.get_parameter_dict(include_frozen=True)
    return GPResults(meanlc,lcvar,peakmjds,filters,
                     gppars=gppars,realizations=sample_funcs,
                     chisq=gp.log_likelihood(y))


def Constant(x,c):
        return 0.*x + c

def _fit_gp_pymc(sn,diff_degree=3.,scale=30.,plot=True,figsize=(15,3),nrandomsamples=2,
                 randomseed=None,printlog=False,**kwargs):

    import pymc.gp as gp
   
    np.random.seed(randomseed)

    meanlc = []
    meanlc_cov = []
    peakmjds = []
    filters = []
    chisq = []
    realizations = []

    sn = pd.DataFrame(sn)

    if plot:
        # figure = plt.figure(figsize=figsize)
        fig = plt.figure(figsize=(24,6*(len(sn['Filter'].unique())//6+1)))

    for i,f in enumerate(sn['Filter'].unique()):
#         print "Filter:", f
        
        if plot:
            plt.subplot(len(sn['Filter'].unique())//6+1,6,i+1)
        
        sn_f = sn[sn['Filter']==f]
        
        y = np.array(sn_f['Mag'])
        yerr = np.array(sn_f['MagErr'])
        x = np.array(sn_f['MJD'])

#         idx = yerr < 5.*np.median(yerr)
        idx = y>0    
        
        if np.sum(idx) == 0:
            continue

        y = y[idx]
        yerr = yerr[idx]
        x = x[idx]

        mean = np.median(y) 
        amp = np.std(y-mean)

        M = gp.Mean(Constant,c=mean)
        C = gp.Covariance(gp.matern.euclidean, diff_degree=diff_degree,
                          amp=amp, scale=scale)

        gp.observe(M, C, obs_mesh=x, obs_vals=y, obs_V=np.power(yerr,2))

        # calculate chisq
        f_chisq = np.sum((y-M(x))**2/yerr**2)
        chisq.append(f_chisq)

        pred_list = []
        if nrandomsamples > 0:
            for i in range(nrandomsamples):
                pred = gp.Realization(M,C)
                pred_list.append(pred)

        xa = np.atleast_1d(np.linspace(x.min(),x.max(),100))
        ya_err = np.sqrt(C(xa))

#         gp.plot_envelope(M, C, mesh=xa)
        
        if plot:
            plt.errorbar(x, y, yerr, fmt='b.')
            plt.plot(xa,M(xa),'r-')
            plt.fill_between(xa, M(xa)-ya_err, M(xa)+ya_err,color='red',alpha=0.5)

            for p in pred_list:
                ya = p(xa)
                plt.plot(xa, ya, lw =0.2, color='g')
            plt.title(f)
            plt.ylim(plt.ylim()[::-1])
            # plt.tight_layout()
               
        meanlc.append(M)
        meanlc_cov.append(C)
        realizations.append(pred_list)
                
        ##estimate peak in each band
        peakres = minimize(M,x[np.argmin(y)])
        pmjd = peakres.x[0]
        if pmjd < x.min() or pmjd > x.max():
            if printlog:
                print("No data before or after the peak for band", f)
            pmjd = -99.
        peakmjds.append(pmjd)
        
        filters.append(f)

    if plot:
        plt.show()

    return GPResults(meanlc,meanlc_cov,peakmjds,filters,
                         realizations=realizations,
                         chisq=chisq)

def load_gp_results(sn,outfolder='outputs/testgp',**kwargs):
    sn = pd.DataFrame(sn)
    snname = sn['Name_upper'].iloc[0]
    f = open(os.path.join(outfolder,"gp_"+snname+".pickle"),"rb")
    res = pickle.load(f)
    return res

## estimate peakmjd in restframe B band
def get_b_band_peakmjd(filters,peakmjds,redshift):

    restfilter_names = ['Bessell' + x for x in ['b','v']]    
    offset = [0., 1.]

    peakmjd = -99.
    for i,fname in enumerate(restfilter_names):

        restf = sncosmo.get_bandpass(fname)
        # print("Effective wavelength of {}: {}".format(restf.name,restf.wave_eff))
        
        efflam = []
        
        for f in filters:
            b = sncosmo.get_bandpass(f)
            efflam.append(b.wave_eff)

        efflam = np.array(efflam)

        # print(redshift)
        # print(np.shape(efflam/(1.+redshift)))
        # print(np.shape(restf.wave_eff))
        fid = np.argmin(np.abs(efflam/(1.+redshift)-restf.wave_eff))

        # print("wavelength difference:",efflam[fid]/(1.+redshift)-restf.wave_eff)
        # print efflam/(1.+redshift)

        if np.abs(efflam[fid]/(1.+redshift)-restf.wave_eff) > 1000.:
            continue
        else:
            if peakmjds[fid] == -99.:
                print("peak not available in ",filters[fid])
                continue
            else:
                peakmjd = peakmjds[fid] + offset[i]
                break

    return peakmjd


def vcmb_minus_vhelio(ra,dec):
    # https://ned.ipac.caltech.edu/help/velc_help.html#posint
    # v_con = v + v_apex[sin(b)*sin(b_apex) + cos(b)*cos(b_apex)*cos(l-l_apex)]
    # where l and b are the object's longitude and latitude, V is its unconverted velocity, and the apices (with Galactic coordinates) of the various motions are given as
    # Conversion  lapex   bapex   Vapex   Source
    # Heliocentric to 3K Background   264.14 deg  +48.26 deg  371.0 km/sec    ApJ 473, 576, 1996
    
    ra = np.array(ra)
    dec = np.array(dec)
    c = SkyCoord(ra=ra, dec=dec, unit=(u.degree, u.degree))
    l,b = c.galactic.l,c.galactic.b
    v_apex = 371.
    b_apex = np.deg2rad(48.26)
    l_apex = np.deg2rad(264.14)
    b_rad = np.deg2rad(b.degree)
    l_rad = np.deg2rad(l.degree)
    dv = v_apex*(np.sin(b_rad)*np.sin(b_apex)+np.cos(b_rad)*np.cos(b_apex)*np.cos(l_rad-l_apex))
    return dv

def vel_to_z(vel):
    vel = np.array(vel)
    c = 299792.46
    return vel/c

def z_to_vel(z):
    z = np.array(z)
    c = 299792.46
    return z*c

def zhelio_to_zcmb(zhelio,ra,dec):

    zhelio = np.array(zhelio)
    vhelio = z_to_vel(zhelio)
    vcmb = vhelio + vcmb_minus_vhelio(ra,dec)
    zcmb = vel_to_z(vcmb)

    return zcmb


def zcmb_to_zhelio(zcmb,ra,dec):

    zcmb = np.array(zcmb)
    vcmb = z_to_vel(zcmb)
    vhelio = vcmb - vcmb_minus_vhelio(ra,dec)
    zhelio = vel_to_z(vhelio)

    return zhelio


def _flatten_result(res):
    """Turn a result from fit_lc into a simple dictionary of key, value pairs.
    Useful when saving results to a text file table, where structures
    like a covariance matrix cannot be easily written to a single
    table row.
    Parameters
    ----------
    res : Result
        Result object from `~sncosmo.fit_lc`.
    Returns
    -------
    flatres : Result
        Flattened result. Keys are all strings, values are one of: float, int,
        string), suitable for saving to a text file.
    """
    from sncosmo.utils import Result
    res.cov_names = res.vparam_names
    flat = Result(success=(1 if res.success else 0),
                  ncall=res.ncall,
                  chisq=res.chisq,
                  ndof=res.ndof)

    # Parameters and uncertainties
    for i, n in enumerate(res.param_names):
        flat[n] = res.parameters[i]
        if res.errors is None:
            flat[n + '_err'] = float('nan')
        else:
            flat[n + '_err'] = res.errors.get(n, 0.)

    # Covariances.
    for n1 in res.param_names:
        for n2 in res.param_names:
            key = n1 + '_' + n2 + '_cov'
            if n1 not in res.cov_names or n2 not in res.cov_names:
                flat[key] = 0.
            elif res.covariance is None:
                flat[key] = float('nan')
            else:
                i = res.cov_names.index(n1)
                j = res.cov_names.index(n2)
                flat[key] = res.covariance[i, j]

    return flat