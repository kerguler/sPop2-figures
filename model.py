TSCALE = 4
import numpy
from numpy.random import uniform
from numpy.random import poisson as rpois
from scipy.stats import gamma, poisson, nbinom
from datetime import datetime, timedelta, date
from ctypes import *
import numpy.ctypeslib as npct
array_1d_double = npct.ndpointer(dtype=numpy.float64, ndim=1, flags='CONTIGUOUS')

filename = "./models/model.dylib"
model = cdll.LoadLibrary(filename)

csim = model.sim
csim.restype = None
csim.argtypes = [c_int,array_1d_double,array_1d_double,array_1d_double,array_1d_double,c_double,array_1d_double]

cpar = model.getPD
cpar.restype = None
cpar.argtypes = [c_double,c_double,array_1d_double,array_1d_double]

prange = [5,50,95]

parnames = [
    'Egg mortality ($p_m$)',
    'Larva mortality ($p_m$)',
    'Pupa mortality ($p_m$)',
    'Adult mortality ($p_m$)',
    'Egg dev. time ($\mu$, days)',
    'Egg dev. time ($\sigma$, days)',
    'Larva dev. time ($\mu$, days)',
    'Larva dev. time ($\sigma$, days)',
    'Pupa dev. time ($\mu$, days)',
    'Pupa dev. time ($\sigma$, days)',
    'Impact of photoperiod ($\phi$)'
]
parnamesC = [
    'Daily mortality ($p_m$)',
    'Daily mortality ($p_m$)',
    'Daily mortality ($p_m$)',
    'Daily mortality ($p_m$)',
    'Dev. time ($\mu$, days)',
    'Dev. time ($\sigma$, days)',
    'Dev. time ($\mu$, days)',
    'Dev. time ($\sigma$, days)',
    'Dev. time ($\mu$, days)',
    'Dev. time ($\sigma$, days)',
    'Impact of photoperiod ($\phi$)'
]
parnamesCr = [
    'Daily mortality ($p_m$)',
    'Daily mortality ($p_m$)',
    'Daily mortality ($p_m$)',
    'Daily mortality ($p_m$)',
    'Dev. rate ($1/\mu$, 1/day)',
    'Dev. time ($\sigma$, days)',
    'Dev. time ($1/\mu$, 1/day)',
    'Dev. time ($\sigma$, days)',
    'Dev. time ($1/\mu$, 1/day)',
    'Dev. time ($\sigma$, days)',
    'Impact of photoperiod ($\phi$)'
]
namesC = [
    'Egg',
    'Larva',
    'Pupa',
    'Adult',
    'Egg',
    'Egg',
    'Larva',
    'Larva',
    'Pupa',
    'Pupa',
    'Impact of photoperiod ($\phi$)'
]
clscl = ['black',
         '#004488',
         '#ee3377',
         'yellow',
         'black',
         'black',
         '#004488',
         '#004488',
         '#ee3377',
         '#ee3377',
         'black'
]

species = numpy.array(['E','L','P','A'])
spc_names = {'E':'Egg',
             'L':'Larva',
             'P':'Pupa',
             'A':'Adult'}
colours = {
    'E':"blue",
    'L':"red",
    'P':"green",
    'A':"black"
}

param = numpy.array([
        [0,     40,        15.0],              #0  p1.1
        [-20,    0,          -3],              #1  p1.2
        [0,      1,       0.999],              #2  p1.3
        #
        [0,     40,        15.0],              #3  p2.1
        [-20,    0,          -3],              #4  p2.2
        [0,      1,       0.999],              #5  p2.3
        #
        [0,     40,        15.0],              #6  p3.1
        [-20,    0,          -3],              #7  p3.2
        [0,      1,       0.999],              #8  p3.3
        #
        [0,     40,        15.0],              #9  p4.1
        [-20,    0,          -3],              #10 p4.2
        [0,      1,       0.999],              #11 p4.3
        #
        [-10,   20,        15.0],              #12 d1m.1
        [25,    50,        20.0],              #13 d1m.2
        [-20,    0,          -5],              #14 d1m.3
        #
        [0.1,    1,      0.2456],              #15 d1s.1
        #
        [-10,   20,        15.0],              #16 d2m.1
        [25,    50,        20.0],              #17 d2m.2
        [-20,    0,          -5],              #18 d2m.3
        #
        [0.1,    1,      0.2456],              #19 d2s.1
        #
        [-10,   20,        15.0],              #20 d3m.1
        [25,    50,        20.0],              #21 d3m.2
        [-20,    0,          -5],              #22 d3m.3
        #
        [0.1,    1,      0.2456],              #23 d3s.1
        #
        [0,     24,        14.0],              #24 ph.thr
        [0,     10,         1.0],              #25 ph.scale
        [0,     10,         1.0]               #26 ph.steep
])
lower = param[:,0]
upper = param[:,1]
numPar = param.shape[0]

def rescalepar(pr):
    # p = pr.copy()
    return pr

def checkPar(pr):
    if (numpy.any(pr<lower) or 
        numpy.any(pr>upper)):
            return 1e13
    return 0

def checkParV(pr):
    if (numpy.any(pr<lower)):
        print("Low",numpy.where(pr<lower)[0])
    if (numpy.any(pr>upper)):
        print("High",numpy.where(pr>upper)[0])

def randomPar():
    pr = lower + uniform(size=lower.shape[0])*(upper-lower)
    pr[24] = 0
    pr[25] = 0
    pr[26] = 0
    return pr

def randomParQ():
    pr = lower + uniform(size=lower.shape[0])*(upper-lower)
    pr[0] = 0
    pr[1] = 0
    pr[2] = 0
    #
    pr[9] = 0
    pr[10] = 0
    pr[11] = 0
    #
    pr[12] = 0
    pr[13] = 25
    pr[14] = 0
    pr[15] = 1
    #
    pr[24] = 0
    pr[25] = 0
    pr[26] = 0
    return pr

def randomParPP():
    return lower + uniform(size=lower.shape[0])*(upper-lower)

def randomFile(filename):
    param = numpy.genfromtxt(filename,delimiter=",")[:,3:]
    def getPar():
        return param[numpy.random.randint(low=0,high=param.shape[0]),:]
    return getPar

def getPD(xr,ph,param,rate=False):
    ret = []
    vec = numpy.ndarray(11,dtype=numpy.float64)
    for i in range(len(xr)):
        x = xr[i]
        pp = ph[i]
        cpar(x,pp,param,vec)
        vv = vec.copy()
        # Mortality:
        vv[[0,1,2,3]] = 1.0 - ((1.0-vv[[0,1,2,3]])**TSCALE)
        # Development time:
        vv[[4,5,6,7,8,9]] /= TSCALE
        if rate:
            # Development rate:
            vv[[4,6,8]] = 1.0 / vv[[4,6,8]]
        ret.append(vv.tolist())
    return numpy.array(ret)

def plotPD(params,labels=[],ylog=False,filename="",filetype="png"):
    import matplotlib.pyplot as plt
    plt.rcParams.update({'text.usetex': True})
    #
    xr = numpy.arange(-5,50,0.1)
    ph = numpy.repeat(24.0,len(xr))
    pps = [numpy.percentile(numpy.array([getPD(xr,ph,rescalepar(pr)) for pr in param]),prange,axis=0) for param in params]
    for n in range(10):
        if ylog:
            if n in [0,1,2,3]:
                  plt.yscale("log")
            else:
                  plt.yscale("linear")
        for i in range(len(pps)):
            pp = pps[i]
            plt.fill_between(xr,pp[0][:,n],pp[2][:,n],alpha=0.5)
            plt.plot(xr,pp[1][:,n],label=None if not len(labels) else labels[i])
        plt.ylabel(parnames[n])
        plt.xlabel("Temperature (°C)")
        if len(labels):
            plt.legend()
        if filename:
            plt.savefig(filename+"_"+str(n)+"."+filetype,bbox_inches="tight",dpi=300)
        plt.show()
        plt.yscale("linear")
    #
    ph = numpy.arange(0,24,0.1)
    xr = numpy.repeat(25.0,len(ph))
    pps = [numpy.percentile(numpy.array([getPD(xr,ph,rescalepar(pr)) for pr in param]),prange,axis=0) for param in params]
    n = 10
    for i in range(len(pps)):
        pp = pps[i]
        plt.fill_between(ph,pp[0][:,n],pp[2][:,n],alpha=0.5)
        plt.plot(ph,pp[1][:,n],label=None if not len(labels) else labels[i])
    plt.ylabel(parnames[n])
    plt.xlabel("Daylength (hours)")
    if len(labels):
        plt.legend()
    if filename:
        plt.savefig(filename+"_"+str(n)+"."+filetype,bbox_inches="tight",dpi=300)
    plt.show()

def plotPDC(parmat,labels=[],ylog=False,subset=False,rate=False,xlim=[],xlimpp=[],ylim=[],ylimp=[],filename="",filetype="png"):
    import matplotlib
    from matplotlib import pyplot as plt
    plt.rcParams.update({'text.usetex': True})
    plt.rcParams.update({'font.size': 14})
    #
    xr = numpy.arange(-5,50,0.1)
    ph = numpy.repeat(24.0,len(xr))
    pp = numpy.percentile(numpy.array([getPD(xr,ph,rescalepar(pr),rate=rate) for pr in parmat]),prange,axis=0)
    sset = [0,1,2,3,-1, 4,6,8,-2, 5,7,9,-3] if not subset else [1,2,-1, 6,8,-2, 7,9,-3]
    for n in sset:
        if n in [-1,-2,-3]:
            if ((ylog and not rate and n in [-2,-3]) or 
                (ylog and rate and n in [-3])):
                plt.yscale("log")
                locs = [0.1, 0.5, 1, 5, 10, 50, 100]
                plt.yticks(locs, ["%g" %l for l in locs])
                # matplotlib.axis.Axis Axes.axes  (matplotlib.ticker.ScalarFormatter())
            if ((ylim and not rate and n in [-2,-3]) or 
                (ylim and rate and n in [-3])):
                plt.ylim(ylim)
            if ylimp and n in [-1]:
                plt.ylim(ylimp)
            if xlim:
                plt.xlim(xlim)
            legend = plt.legend()
            legend.get_frame().set_alpha(0.25)
            plt.rcParams.update({'font.size': 14})
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            if filename:
                plt.savefig(filename+"_"+str(-n)+"."+filetype,bbox_inches="tight",dpi=300)
            plt.show()
            plt.yscale("linear")
            continue
        plt.fill_between(xr,pp[0][:,n],pp[2][:,n],color=clscl[n],alpha=0.5)
        plt.plot(xr,pp[1][:,n],color=clscl[n],label=namesC[n])
        plt.ylabel(parnamesCr[n] if rate else parnamesC[n],fontsize=14)
        plt.xlabel("Temperature (°C)",fontsize=14)
    #
    ph = numpy.arange(0,24,0.1)
    xr = numpy.repeat(25.0,len(ph))
    pp = numpy.percentile(numpy.array([getPD(xr,ph,rescalepar(pr),rate=rate) for pr in parmat]),prange,axis=0)
    n = 10
    plt.fill_between(ph,pp[0][:,n],pp[2][:,n],color=clscl[n],alpha=0.5)
    plt.plot(ph,pp[1][:,n],color=clscl[n],label=None)
    if xlimpp:
        plt.xlim(xlimpp)
    plt.ylabel(parnames[n],fontsize=14)
    plt.xlabel("Daylength (hours)",fontsize=14)
    plt.rcParams.update({'font.size': 14})
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    if filename:
        plt.savefig(filename+"_"+str(n)+"."+filetype,bbox_inches="tight",dpi=300)
    plt.show()    
    
def sim(temp,photo,pr,init,thr):
    pr = rescalepar(pr)
    tf = temp.shape[0]
    ret = numpy.ndarray(tf*8,dtype=numpy.float64)
    csim(tf,temp,photo,pr,init,thr,ret)
    ret = ret.reshape((tf,8))
    return ret

def getInit(b):
    init = numpy.array([
        b['E'][0] if 'E' in b else 0,
        b['L'][0] if 'L' in b else 0,
        b['P'][0] if 'P' in b else 0,
        b['A'][0] if 'A' in b else 0
    ],dtype=numpy.float64)
    #
    if numpy.isnan(init[0]) and init[1] == 0:
        # If the initial number of eggs is not known,
        # but the experiment starts with eggs,
        # initiate with an appropriate number of eggs by assuming 
        # 30% mortality between 15 and 27.5oC (DOI:10.1093/jme/tjy224).
        # 
        mx = numpy.nanmax(b['L'])
        init[0] = mx + rpois(lam=(mx/0.7)-mx,size=1)
    #
    return init

matchSim_key = ["","",""]
matchSim_sm = None
matchSim_ss = None
def matchSim(b,pr):
    global matchSim_key
    global matchSim_sm
    global matchSim_ss
    #
    tt = b['temp']
    pp = b['photo']
    init = getInit(b)
    #
    strings = [str(tt),str(pr),str(init)]
    if numpy.all(strings==matchSim_key):
        return matchSim_sm, matchSim_ss
    #
    sm = sim(tt,pp,pr,init,0)
    # Cumulative production of Egg, Larva, Pupa, and Adult
    ss = numpy.cumsum(sm[:,4:],axis=0)
    #
    matchSim_key = strings
    matchSim_sm = sm
    matchSim_ss = ss
    return sm, ss

def plotMatches(obs,prs,dates=False,legend=True,mark=False,filename="",filetype="png",envir=False,plot=True,fig=False,ax1=False,ax2=False):
    import matplotlib.pyplot as plt
    plt.rcParams.update({'text.usetex': True})
    plt.rcParams.update({'font.size': 14})
    if not (fig and ax1 and ax2):
        fig, ax1 = plt.subplots()
        if envir:
            ax2 = ax1.twinx()
            ax1.set_zorder(ax2.get_zorder()+1)
            ax1.set_frame_on(False)
    envir_labels = True
    labels = True
    for i in range(len(obs)):
        o = obs[i]
        if not 'temp' in o:
            continue
        if envir:
            tm = numpy.arange(len(o['temp']))/TSCALE
            if dates and 'Date' in o:
                tm = numpy.array([o['Date'][0]+timedelta(days=t) for t in tm])
            labt = None
            labp = None
            if envir_labels:
                labt = "Temperature"
                labp = "Photoperiod"
            ax2.plot(tm,o['temp'],color="silver",lw=2,label=labt)
            ax2.plot(tm,o['photo'],color="silver",ls="dashed",lw=4,label=labp)
            envir_labels = False
        #
        if len(prs) > 0:
            sms = []; sss = []
            for pr in prs:
                sm, ss = matchSim(o,pr)
                sms.append(sm); sss.append(ss)
            sms = numpy.array(sms); sss = numpy.array(sss)
        #
        tm = numpy.arange(o['days'][-1]*TSCALE+1)/TSCALE
        if dates and 'Date' in o:
            tm = numpy.array([o['Date'][0]+timedelta(days=t) for t in tm])
        #
        for spc in o['compare']:
            labo = None
            lab = None
            if labels:
                labo = "Observed "+spc_names[spc]
                lab = "Simulated "+spc_names[spc]
            ax1.plot(o['Date'] if dates and 'Date' in o else o['days'],o[spc],marker='o',linestyle='dashed',c=colours[spc],label=labo)
            if len(prs) > 0:
                n = numpy.where(species==spc)[0][0]
                if o['type'][0] == 'C':
                    cvec = numpy.percentile(sss[:,:,n],prange,axis=0)
                elif o['type'][0] == 'A':
                    cvec = numpy.percentile(sms[:,:,n],prange,axis=0)
                else:
                    print("Wrong type",o['type'])
                ax1.fill_between(tm,cvec[0],cvec[2],color=colours[spc],alpha=0.1)
                ax1.plot(tm,cvec[1],c=colours[spc],label=lab)
        #
        labels = False
    fig.tight_layout()
    ax1.set_ylabel("Number of each stage")
    if envir:
        ax2.set_ylabel("Temperature (°C) and Photoperiod (hrs)")
    if legend:
        ax1.legend(loc='upper left')
        if envir:
            ax2.legend(loc='upper right')
    if dates:
        for tick in ax1.get_xticklabels():
            tick.set_rotation(30)
    if not (dates and 'Date' in o):
        ax1.set_xlabel("Time (days)")
    if plot:
        if mark:
            x = ax1.get_xlim()
            y = ax1.get_ylim()
            plt.text(x[1],y[1],'*',fontsize=28,ha='right',va='top')
        if filename:
            plt.savefig(filename+"."+filetype,bbox_inches="tight",dpi=300)
        plt.show()
    if envir:
        return fig, ax1, ax2
    else:
        return fig, ax1

def getScores(obs):
    def score(pr,verbose=False):
        scr = checkPar(pr)
        if scr >= 1e13:
            return scr
        for b in obs:
            #
            if 'funscr' in b:
                scr += b['funscr'](pr)
                continue
            #
            sm, ss = matchSim(b,pr)
            if numpy.all(sm==0) or numpy.all(ss==0):
                return 1e13
            for spc in species:
                if not (spc in b['compare']):
                    continue
                n = numpy.where(species==spc)[0][0]
                #
                if b['type'] == 'CN0wP':
                    scr += numpy.nansum( -poisson.logpmf(b[spc][1:], ss[:,n][ b['days'][1:]*TSCALE ]) )
                elif b['type'] == 'AN0sP':
                    scr += numpy.nansum( -poisson.logpmf(b[spc][1:], sm[:,n][ b['days'][1:]*TSCALE ]) )
                else:
                    print("Wrong type",b['type'])
                    return 1e13
        return scr
    return score
