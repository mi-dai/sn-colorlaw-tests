import pandas as pd
from snutils import MySNUtils,PanPlusUtils

mysn = MySNUtils()
ppsn = PanPlusUtils()

def fit_salt2(name,time,flux,err,band,instrument,survey,
              ra,dec,z_helio,sample,**kwargs):
    data = pd.DataFrame({"Name_upper":name,"MJD":time,"Mag":flux,"MagErr":err,"Filter":band,"Instrument":instrument,"Survey":survey})
    meta = pd.DataFrame({"Name_upper":name,"RA":ra,"DEC":dec,"z_helio":z_helio,"Sample":sample})
    # res = {"mjd1":data["MJD"][0],"ra":meta["RA"][0]}
    # res = {"Name_upper":data["Name_upper"]}
    res = mysn.fit_lcparams(data,meta,**kwargs).iloc[0]
    # res = fit_lcparams(data,meta,**kwargs)
    return res

def fit_salt2_pp(name,time,flux,err,band,
              ra,dec,z_helio,sample,mwebv,**kwargs):
    data = pd.DataFrame({"SNID":name,"MJD":time,"FLUXCAL":flux,"FLUXCALERR":err,"FLT":band})
    meta = pd.DataFrame({"SNID":name,"RA":ra,"DEC":dec,"REDSHIFT_HELIO":z_helio,"SURVEY2":sample,"MWEBV":mwebv})
    # res = {"mjd1":data["MJD"][0],"ra":meta["RA"][0]}
    # res = {"Name_upper":data["Name_upper"]}
    res = ppsn.fit_lcparams(data,meta,**kwargs).iloc[0]
    # res = ppsn.fit_lcparams(data,meta,**kwargs)
    return res