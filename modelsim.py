TSCALE = 4
import numpy
from matplotlib.pyplot import clim
from model import sim, getInit
from datetime import datetime, timedelta, date
from matplotlib import pylab as plt
import pandas
import random

def ifin(vec,pick):
    return numpy.sum([vec==p for p in pick],axis=0)>0

paramQ = numpy.genfromtxt("mat/posterior_Cxquin.csv",delimiter=",")
paramP = numpy.genfromtxt("mat/posterior_Cxpip.csv",delimiter=",")

def next_weekday(d, weekday):
    days_ahead = weekday - d.weekday()
    return datetime(d.year,d.month,d.day) + timedelta(days=days_ahead)

def getParam(spc):
    parmat = {
        'P': paramP, 
        'Q': paramQ
    }[spc]
    return parmat[random.randint(0,parmat.shape[0]-1)]

def simClims(clims,param=[],funpar=None,init=[100,0,0,0],thr=0):
    ret = []
    init = numpy.array(init,dtype=numpy.float64)
    for ii in numpy.arange(0,52+1)*7*TSCALE:
        if funpar != None:
            param = funpar()
        for i in range(len(clims)):
            clim = clims[i]
            dt = clim['days'][ii:]
            tm = numpy.array(clim['mean_air_temp'][ii:].data)
            ph = clim['photoperiod'][ii:]
            sm = sim(tm,ph,param,init,thr)
            A = numpy.cumsum(sm[:,7])
            po = A[-1]
            if po < 1 or not numpy.any(A>(0.5*po)):
                ret.append([i,clim['days'][ii],po,numpy.nan])
            else:
                ret.append([i,clim['days'][ii],po,(numpy.where(A>(0.5*po))[0][0]-1)/TSCALE])
    return ret

print_date = True
def printData(clims,rets):
    global print_date
    if len(rets)==0:
        return
    rets = numpy.array(rets)
    for i in numpy.sort(numpy.unique(rets[:,0])):
        clim = clims[i]
        ret = rets[rets[:,0]==i,1:]
        tm = numpy.array(ret[:, 0])
        if print_date:
            print(">DATE_%d: %s" %(i, ",".join([t.isoformat() for t in tm])),flush=True)
            print_date = False
        print(">PRODA_%d: %d,%d,%.3f,%.3f,%s" % (i, clim['loni'],clim['lati'],clim['lon'],clim['lat'],",".join(["%g" %(t) for t in ret[:,1]])),flush=True)
        print(">DEVTM_%d: %d,%d,%.3f,%.3f,%s" % (i, clim['loni'],clim['lati'],clim['lon'],clim['lat'],",".join(["%g" %(t) for t in ret[:,2]])),flush=True)

def getGlobalSim(dset,typ,spc="Cxpip"):
    import pandas
    from datetime import datetime, timedelta, date
    if dset=='era5':
        a = numpy.array(pandas.read_csv("mat/global_ERA5_%s_%s.csv" %(spc,typ),header=None))
        a[a == numpy.inf] = numpy.nan
        dt = numpy.array([(date(2017,1,1)+timedelta(days=int(i*7))).isocalendar()[1] for i in range(a.shape[1]-4)])
    elif dset=='nasa0':
        a = numpy.array(pandas.read_csv("mat/global_NASA_model0_era0_%s_%s.csv" %(spc,typ),header=None))
        a[a == numpy.inf] = numpy.nan
        dt = numpy.array([(date(2010,1,1)+timedelta(days=int(i*7))).isocalendar()[1] for i in range(a.shape[1]-4)])
    elif dset=='nasa1':
        a = numpy.array(pandas.read_csv("mat/global_NASA_model0_era1_%s_%s.csv" %(spc,typ),header=None))
        a[a == numpy.inf] = numpy.nan
        dt = numpy.array([(date(2045,1,1)+timedelta(days=int(i*7))).isocalendar()[1] for i in range(a.shape[1]-4)])
    return a, dt

def readGlobal(dset='ERA5',kind='DEVTM'):
    if dset == 'ERA5':
        a = numpy.array(pandas.read_csv("mat/global_ERA5_Cxpip_y2x3_%s.csv" %(kind),header=None))
        a[a == numpy.inf] = numpy.nan
        dy = numpy.array([date(2017,1,1)+timedelta(days=int(i*7)) for i in range(a.shape[1]-4)])
        dt = numpy.array([i.isocalendar()[1] for i in dy])
    elif dset == 'NASA':
        a = numpy.vstack([numpy.array(pandas.read_csv("mat/global_NASA_%d_Cxpip_R3_%s_%d.csv" %(y,kind['type'],int(kind['diff']>0)),header=None)) for y in kind['years']])
        a[a == numpy.inf] = numpy.nan
        dy = numpy.array([date(kind['years'][0]+kind['diff'],1,1)+timedelta(days=int(i*7)) for i in range(a.shape[1]-4)])
        dt = numpy.array([i.isocalendar()[1] for i in dy])
    return a, dy, dt

def calcGlobal(mat,operation):
    if operation=="median":
        return numpy.nanmedian(mat,axis=1)
    if operation=="mean":
        return numpy.nanmean(mat,axis=1)
    if operation=="std":
        return numpy.nanstd(mat,axis=1)
    if operation=="ste":
        return numpy.nanstd(mat,axis=1)/numpy.sqrt(numpy.sum(~numpy.isnan(mat),axis=1))
    if operation=="slen":
        return numpy.sum(~numpy.isnan(mat),axis=1)
    return False

def calcEmergence(dy, hh):
    c = [numpy.array([dy[i] + timedelta(days=(h[i] if not numpy.isnan(h[i]) else 0)) for i in numpy.arange(dy.shape[0])]) for h in hh]
    return c

def calcEmergencePP(dy, aaa):
    c = [numpy.array([dy[i] + timedelta(days=(h[i] if not numpy.isnan(h[i]) else 0)) for i in numpy.arange(dy.shape[0])]) for h in aaa]
    for cc in c:
        print(datetime.strftime(cc[~numpy.isnan(numpy.array(aaa[1],dtype=numpy.float64))][0],'%B'))
    return c

def calcRet(ret):
    # tm = numpy.array([a.to_pydatetime() for a in ret[0,:,0]])
    tm = numpy.array(ret[0,:,0])
    pp = numpy.nanpercentile(numpy.array(ret[:,:,1:],dtype=numpy.float64),[5,50,95],axis=0)
    c = calcEmergence(tm, numpy.array(ret[:,:,2]))
    xr = numpy.array([numpy.min(c)+timedelta(days=i) for i in range((numpy.max(c)-numpy.min(c)).days)])
    yr = {}
    for i in range(len(c)):
        for j in range(len(c[i])):
            key = next_weekday(c[i][j], 0) # Monday!
            if key not in yr:
                yr[key] = []
            yr[key].append(ret[i,j,2])
    yr = numpy.array([numpy.hstack([key]+[numpy.nanpercentile(numpy.array(yr[key],dtype=numpy.float64),[5,50,95],axis=0)]) for key in sorted(yr)])
    m = numpy.array([numpy.nanmin(c[i][~numpy.isnan(numpy.array(ret[i,:,2],dtype=numpy.float64))]) for i in range(ret[:,:,2].shape[0])])
    return tm, pp, m

def filterGlobal(prd, dev):
    if not numpy.all(prd[:,:4]==dev[:,:4]):
        print("Results are not comparable!")
        return
    xr = (prd[:,4:]<5) & (dev[:,4:]>180)
    prd[:,4:][xr] = numpy.nan
    dev[:,4:][xr] = numpy.nan

def filterSim(ret):
    ret[(ret[:,:,2]>180) & (ret[:,:,1]<5),1:3] = numpy.nan
    
def sortGlobal(a, dy, rep=3):
    l = numpy.argsort(numpy.array(["%g,%g" %(m[0],m[1]) for m in a])).reshape((int(a.shape[0]/rep),rep))
    print(numpy.all([numpy.all(a[l[:,0],:3]==a[l[:,i],:3]) for i in range(l.shape[1])]))
    #
    m = numpy.array([calcGlobal(a[l[:,i],4:],'mean') for i in range(l.shape[1])])
    ml = numpy.array([calcGlobal(a[l[:,i],4:],'slen') for i in range(l.shape[1])])
    c = numpy.nanmean(m,axis=0)
    mlc = numpy.nanmean(ml,axis=0)
    #
    # med = numpy.array([calcGlobal(a[l[:,i],4:],'median') for i in range(l.shape[1])])
    medi = numpy.nanmedian(m,axis=0)
    #
    # sz = numpy.sum(~numpy.isnan(m),axis=0)
    # s = numpy.nanstd(m,axis=0)/numpy.sqrt(sz)
    tms = numpy.stack([a[l[:,i],4:] for i in range(l.shape[1])])
    sd = numpy.nanstd(tms,axis=0)
    s = numpy.nanmean(sd,axis=1)
    #
    nm = numpy.nanmean(a[l,4:],axis=1)
    mnm = numpy.array([[(dy[j] + timedelta(days=nm[i,j])).toordinal() if not numpy.isnan(nm[i,j]) else numpy.inf for i in range(nm.shape[0])] for j in range(53)])
    mnm2 = numpy.array(numpy.nanmin(numpy.array(mnm,dtype=numpy.float64),axis=0),dtype=numpy.int64)
    mnm3 = numpy.array([date.fromordinal(tmp).month if tmp>0 else numpy.nan for tmp in mnm2])
    mnm3d = numpy.array([date.fromordinal(tmp).timetuple().tm_yday if tmp>0 else numpy.nan for tmp in mnm2])
    #
    xr = numpy.arange(c.shape[0])
    lats = a[l[:,0],3].copy()
    lons = a[l[:,0],2].copy()
    lons[lons>180] -= 360
    #
    return {'lon':lons,'lat':lats,'mn':c,'mnstd':s,'first':mnm3,'firstday':mnm3d,'med':medi,'slen':mlc}, tms

def plotProfile(dy, matD, matP, lon,lat, show=True):
    lon, lat = matD[numpy.argmin(numpy.abs(matD[:,2]-lon)+numpy.abs(matD[:,3]-lat)),2:4]
    print(lon,lat)
    #
    aa = matD[(matD[:,2]==lon) & (matD[:,3]==lat),:]
    print(aa.shape)
    aaa = numpy.nanpercentile(aa[:,4:],[5,50,95],axis=0)
    #
    bb = matP[(matP[:,2]==lon) & (matP[:,3]==lat),:]
    print(bb.shape)
    bbb = numpy.nanpercentile(bb[:,4:],[5,50,95],axis=0)
    #
    c = calcEmergencePP(dy, aaa)
    #
    plt.rcParams['figure.figsize'] = [12, 4]
    plt.fill_between(dy,bbb[0],bbb[2],alpha=0.5)
    plt.plot(dy,bbb[1])
    plt.fill_between(dy,aaa[0],aaa[2],alpha=0.5)
    plt.plot(dy,aaa[1])
    plt.plot(c[0],aaa[0],color="black")
    plt.plot(c[1],aaa[1],color="black")
    plt.plot(c[2],aaa[2],color="black")
    if show:
        plt.show()
