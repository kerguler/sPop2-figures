TSCALE = 4
import numpy
import model
import matplotlib.pyplot as plt

temp = numpy.array([  15.0,   20.0,   23.0,   27.0,   30.0])
ph   = numpy.array([  24.0,   24.0,   24.0,   24.0,   24.0])
d1m  = numpy.array([  6.82,   3.08,   2.03,   1.03,    1.0]) # days
d1s  = numpy.array([   0.5,   0.28,   0.18,   0.12,   0.18]) # days

p1h  = numpy.array([24.0,   24.0,   24.0,   24.0,   24.0,   24.0, 24.0])
p1t  = numpy.array([ 5.0,   15.0,   20.0,   23.0,   27.0,   30.0, 35.0])
p1   = numpy.array([ 0.5, 0.0818, 0.1648, 0.1872, 0.4773, 0.7247,  1.0]) # daily mortality

p2h  = numpy.array([24.0, 24.0])
p2t  = numpy.array([32.0, 40.0])
p2   = numpy.array([ 0.5,  1.0]) # daily mortality
q2h  = numpy.array([   24.0,   24.0,   24.0,   24.0,   24.0, 24.0])
q2t  = numpy.array([   15.0,   20.0,   23.0,   27.0,   30.0, 32.0])
q2   = numpy.array([0.02134,0.01223,0.01922,0.01325,0.04190,  0.5]) # daily mortality

p3h  = numpy.array([24.0, 24.0, 24.0])
p3t  = numpy.array([ 5.0, 37.0, 40.0])
p3   = numpy.array([ 0.5,  0.5,  1.0]) # daily mortality
q3h  = numpy.array([24.0,   24.0,   24.0,   24.0,   24.0,   24.0, 24.0])
q3t  = numpy.array([ 5.0,   15.0,   20.0,   23.0,   27.0,   30.0, 37.0])
q3   = numpy.array([ 0.5,0.06360,0.02313,0.04085,0.01798,0.01731,  0.5]) # daily mortality

def fun_d1ms_p1(pr):
    p_sim = model.getPD(p1t,p1h,model.rescalepar(pr)) # Yields daily values
    d_sim = model.getPD(temp,ph,model.rescalepar(pr)) # Yields daily values
    return numpy.sum(((p_sim[:,0] - p1)/0.025)**2) + numpy.sum(((d_sim[:,4] - d1m)/0.125)**2) + numpy.sum(((d_sim[:,5] - d1s)/0.025)**2)

def fun_p2(pr):
    p_sim = model.getPD(p2t,p2h,model.rescalepar(pr)) # Yields daily values
    return numpy.sum(((p_sim[:,1] - p2)/0.025)**2)

def fun_p3(pr):
    p_sim = model.getPD(p3t,p3h,model.rescalepar(pr)) # Yields daily values
    return numpy.sum(((p_sim[:,2] - p3)/0.025)**2)

def plot(prs,par,ylog=False):
    xr = numpy.arange(0,50,0.1)
    ph = numpy.repeat(16.0,len(xr))
    #
    sims = numpy.array([model.getPD(xr,ph,model.rescalepar(pr)) for pr in prs]) # Yields daily values
    pp = numpy.percentile(sims,[5,50,95],axis=0)
    #
    plt.fill_between(xr,pp[0][:,par],pp[2][:,par],alpha=0.5)
    plt.plot(xr,pp[1][:,par])
    if ylog:
        plt.yscale("log")
        locs, labels = plt.yticks()
        plt.yticks(locs, ["%g" %l for l in locs])
    else:
        plt.yscale("linear")
    plt.ylabel(model.parnames[par])
    plt.xlabel("Temperature (°C)")

obs = {
    'd2s.1': {
        'funscr': lambda pr: ((pr[31]-0.2)/0.05)**2
    },
    'd1ms_p1': {
        'funscr': fun_d1ms_p1
    },
    'p2': {
        'funscr': fun_p2
    },
    'p3': {
        'funscr': fun_p3
    }
}
