import glob
import gzip
# from __future__ import print_function
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


class SNUtils:
    def __init__(self):
        pass
    
    def fit_lcparams(self):
        pass
    
class MySNUtils(SNUtils):
    
    def __init__(self):
        pass
    
    def register_bands_for_sncosmo(self):
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

    def get_refmag(self,survey,band): ## !!need to double check all the values here are correct
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

        
    def fit_lcparams(self,lcdata,metadata,modelsource='salt2',modelpars=['t0','x0','x1','c'],
                     mpbounds=None,modelcov=True,snlist=None,register_bands=False,
                     write_to_file=True,outfile='data/lcparams.dat',usebands='all',
                     combine_filt_inst=True,rescols=None,**kwargs):

        if register_bands:
            self.register_bands_for_sncosmo()
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
                        mref,magsys = self.get_refmag(s,b[b.find(s)+len(s):].split('-')[-1])
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


class PanPlusUtils(SNUtils):
    
    def __init__(self):
        pass
    
    def _read_snana_kcor_input(self,kcorinputfile,rename_bands=False,rename_bands_thres=100):
        kcorinfo = {}
        filters = []
        kcorinfo["survey"] = kcorinputfile.split('kcor_')[1].split('.')[0]
        for n,line in enumerate(open(kcorinputfile)):
            if "MAGSYSTEM:" in line:
                kcorinfo["magsys"] = line.split(":")[1].strip().split()[0].strip()
            if "FILTPATH:" in line:
                kcorinfo["filtpath"] = line.split(":")[1].strip()
            # if "SURVEY:" in line:
            #     kcorinfo["survey"] = line.split(":")[1].strip()
            if "FILTER:" in line:
                filtname = line.split(":")[1].strip().split()[0].strip()
                if '/' in filtname:
                    filtname1 = filtname.split('/')[0]
                    filtname2 = filtname.split('/')[1]
                else:
                    filtname1 = filtname
                    filtname2 = ''
                filtfile = line.split(":")[1].strip().split()[1].strip()
                refmag = line.split(":")[1].strip().split()[2]
                if not refmag[-1].isdigit():
                    refmag = refmag[:-1] 
                refmag = eval(refmag)
                filtdict = {"filtname":filtname1,"filtstr":filtname2,"filtfile":filtfile,"refmag":refmag}
                filtdict.update(kcorinfo)
                filters.append(filtdict)
       
        df_filt = pd.DataFrame(filters)
        
        def assign_new_band_value(row):
            if len(set(row['survey']).intersection(row['filtname']))<rename_bands_thres:
                row['filtname'] = '%s-%s'%(row['survey'],row['filtname'])
            return row
        if rename_bands:
            df_filt = df_filt.apply(assign_new_band_value,axis=1)
        return df_filt
        
    def _register_single_band_for_sncosmo(self,filtfile,filtpath,filtname,survey,verbose=False,force=False):
        band = sncosmo.read_bandpass(os.path.expandvars(os.path.join(filtpath,filtfile)))
        # bandname = '%s-%s'%(survey,filtname) if survey == 'SDSS' else filtname
        # band.name = bandname
        band.name = filtname
        sncosmo.register(band,force=True)
        if verbose:
            print('band %s registered in sncosmo, effwave=%f'%(band.name,band.wave_eff))
    
    def register_bands_for_sncosmo(self,filtmap=None,**kwargs):
        if filtmap is None:
            raise ValueError("filtmap needs to be provided")
        if not isinstance(filtmap,pd.DataFrame):
            raise ValueError("filtmap must be a pandas.DataFrame")
        filtmap.apply(lambda row: self._register_single_band_for_sncosmo(row['filtfile'],row['filtpath'],row['filtname'],row['survey']),axis=1)

    def fit_lcparams(self,lcdata,metadata,modelsource='salt2',modelpars=['t0','x0','x1','c'],
                     mpbounds=None,modelcov=True,snlist=None,register_bands=False,
                     write_to_file=True,outfile='data/lcparams.dat',usebands='all',
                     combine_filt_inst=False,rescols=None,mwebv_from_coord=False,
                     filtmap=None,
                     **kwargs):

        if register_bands:
            self.register_bands_for_sncosmo(filtmap=filtmap)
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
            snlist = list(set(lcdata['SNID']))
            snlist = np.sort(np.array(snlist))

        if write_to_file:
            f = open(outfile,'w')

        header=True
        for i,sn in enumerate(snlist):
            zp = 27.5
            lc = Table.from_pandas(lcdata[lcdata['SNID'] == sn])
            meta = Table.from_pandas(metadata[metadata['SNID'] == sn])

            lc['flux'] = lc['FLUXCAL']
            lc['flux_err'] = lc['FLUXCALERR']
            lc['zp'] = zp
            lc['zpsys'] = 'ab'.ljust(10)

            z = meta[0]['REDSHIFT_HELIO']
            zsource = 'MetaData'
            
            if mwebv_from_coord:
                ra = meta[np.isfinite(meta['RA'])][0]['RA']
                dec = meta[np.isfinite(meta['DEC'])][0]['DEC']
                mwebv = dustmap.ebv(ra,dec)
            else:
                mwebv = meta['MWEBV'][0]
            print(i, "fitting ", sn, "z=", z, "zsource=", zsource, "mwebv=", mwebv)
            model.set(z=z,mwebv=mwebv)
            # surveys = meta[0]['Surveys'].split("::")
            try:
                lc0 = lc

                if usebands != 'all':
                    lc0 = lc0[np.array([x.split('-')[-1][0:1] in list(usebands) for x in lc0['Filter']])]
                    if len(lc0) == 0:
                        print("No data in selected bands:",usebands)
                        continue

                for b in set(lc0['FLT']):
                    # mref,magsys = get_refmag(s,b.split('-')[1])
                    # mref,magsys = self.get_refmag(b)
                    mref = filtmap.loc[filtmap.filtname==b,"refmag"].values[0]
                    magsys = filtmap.loc[filtmap.filtname==b,"magsys"].values[0]
                    lc0['zp'][lc0['FLT']==b] = lc0['zp'][lc0['FLT']==b]-mref
                    lc0['zpsys'][lc0['FLT']==b] = magsys

                result, fitted_model = sncosmo.fit_lc(lc0,model,modelpars,
                                                      modelcov=modelcov,
                                                      bounds=mpbounds.copy(),**kwargs)
                # print(result)
                res = []
                # res.append(sncosmo.flatten_result(result))
                res.append(_flatten_result(result))
                res = Table(res)
                res['Name'] = sn.ljust(30)
                s = meta['SURVEY2'][0]
                res['Survey'] = s.ljust(20)
                res['ErrorType'] = 'None'.ljust(50)
                res['ErrorMesg'] = 'None'.ljust(50)
                res['bands'] = ''.join(sorted([x.split('-')[-1][0:1] for x in set(lc0[result.data_mask]['FLT'])])).ljust(20)
                res['bands2'] = ''.join(sorted([x.split('-')[-1][0:1] for x in set(lc0['FLT'])])).ljust(20)
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
        
    def get_sn_data(self,filefolder=None,rename_bands=False,rename_bands_thres=100):
        if filefolder is None:
            raise ValueError("filefolder is not provided")
        data = pd.DataFrame()
        meta = pd.DataFrame()
        files = glob.glob(os.path.expandvars(filefolder+"/*.dat.gz"))
        # print(files)
        metacols = ["SURVEY","SNTYPE","SNID","RA","DEC","MWEBV","FILTERS",
                    "SEARCH_PEAKMJD","HOSTGAL_LOGMASS","PIXSIZE",
                    "REDSHIFT_HELIO","REDSHIFT_CMB","VPEC"]
        for f in files:   
            surveyname = f.split('/')[-2].split('K21_')[-1]

            headercount = 0
            datacount = 0
            meta0 = {}
            # print(meta0)
            for n,line in enumerate(gzip.open(f)):
                line = line.decode()
                headercount +=1 
                for col in metacols:
                    if col in line:
                        # print(col)
                        val = line.split(":")[1].split("#")[0].strip()
                        if "+-" in val:
                            err = val.split("+-")[1].strip()
                            val = val.split("+-")[0].strip()   
                            meta0[col+'_err'] = np.float64(err)
                        meta0[col] = np.float64(val) if val.replace('.','').isdigit() else val
                if "VARLIST" in line: 
                    datastart = headercount-1
                if "OBS" in line and "NOBS" not in line:
                    datacount +=1
                    # print(datacount)
                if "END_PHOTOMETRY" in line:
                    break
            data0 = pd.read_csv(f,skiprows=datastart,nrows=datacount,sep='\s+')
                                # data_start=header_start+1,data_end=-1,header_start=header_start)
            # print(data0)
            data0['SNID'] = meta0['SNID']
            if 'BAND' in data0.columns:
                data0 = data0.rename(columns={'BAND':'FLT'})
            meta0['SURVEY2'] = surveyname
                       
            def assign_new_band_value(row):
                if 'BAND' in data0.columns:
                    bandcol = 'BAND'
                elif 'FLT' in data0.columns:
                    bandcol = 'FLT'
                else:
                    bandcol = 'FILT'
                if len(set(meta0['SURVEY2']).intersection(row[bandcol]))<rename_bands_thres:
                    row[bandcol] = '%s-%s'%(meta0['SURVEY2'],row[bandcol])
                return row
            if rename_bands:          
                data0 = data0.apply(assign_new_band_value,axis=1)
            
            def separate_band_str(row):
                if '/' in row['FLT']:
                    row['FLT_str'] = row['FLT'].split('/')[1]
                    row['FLT'] = row['FLT'].split('/')[0]
                else:
                    row['FLT_str'] = ''
                return row
            data0 = data0.apply(separate_band_str,axis=1)
            
            meta0 = pd.DataFrame(meta0,index=[0])
            meta = pd.concat([meta,meta0],join='outer')
            data = pd.concat([data, data0], join='outer')
        return data,meta

