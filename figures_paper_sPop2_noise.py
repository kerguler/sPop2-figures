## Copyright (C) 2021  Kamil Erguler
## 
##     This program is free software: you can redistribute it and/or modify
##     it under the terms of the GNU General Public License as published by
##     the Free Software Foundation, either version 3 of the License, or
##     any later version.
## 
##     This program is distributed in the hope that it will be useful,
##     but WITHOUT ANY WARRANTY; without even the implied warranty of
##     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
##     GNU General Public License for more details (<https://www.gnu.org/licenses/>).

import numpy
import matplotlib.pyplot as plt

from scipy.stats import gamma
from scipy.stats import nbinom
from scipy.stats import poisson
from scipy.stats import erlang

from numpy.random import uniform
from numpy.random import normal
from numpy.random import poisson as rpois
from numpy.random import binomial, choice
from numpy.random import geometric

from scipy.interpolate import interp1d
from scipy.special import gamma as fgamma
from scipy.special import gammaincc
from scipy.special import gammainccinv
from scipy.special import factorial

from spop2 import spop
from spop2 import approximate

def sim(xr,mu,sd,mode):
    a = spop(stochastic=False,gamma_fun=mode)
    a.add(0,0,0,0,100)
    ret = []
    ret.append([0, 0, 0] + a.read())
    for i in numpy.arange(xr.shape[0]):
        a.iterate(0,mu[i],sd[i],0,0,0)
        ret.append([i+1,mu[i],sd[i]]+a.read())
    return numpy.array(ret)

rep = 1000
xr = numpy.arange(0,50,1)

# -----------------------------------------------------------------

def getDev(xr,r):
    dev = normal(30.0, r, xr.shape[0])
    dev[dev < 3.0] = 3.0
    return dev

def getDevOpt(xr,r):
    dev = normal(30.0, r, xr.shape[0])
    dev = 30.0 + numpy.abs(dev - 30.0)
    return dev

# Switch to "True" to simulate the corresponding dataset
if False:
        r = 8
        retA = []
        for n in range(rep):
            print(i, r, n)
            dev = getDev(xr,r)
            retA.append(sim(xr,dev,0.1*dev,"MODE_ACCP_ERLANG").tolist())
        numpy.save("mat/figures_paper_noise3_retAs_rs0.npy",retA)

# Switch to "True" to simulate the corresponding dataset
if False:
        r = 8
        retA = []
        for n in range(rep):
            print(i, r, n)
            dev = getDevOpt(xr,r)
            retA.append(sim(xr,dev,0.1*dev,"MODE_ACCP_ERLANG").tolist())
        numpy.save("mat/figures_paper_noise3opt_retAs_rs0.npy",retA)

# Switch to "True" to simulate the corresponding dataset
if False:
    from spop2 import approximate
    eps = 1e-4
    rep = 100
    r = 8
    approximate(eps)
    retA = []
    for n in range(rep):
            print(i, r, n)
            dev = getDev(xr,r)
            sd = numpy.repeat(3.0,xr.shape[0])
            retA.append(sim(xr,dev,sd,"MODE_ACCP_ERLANG").tolist())
    numpy.save("mat/figures_paper_noise3_retAs_approx_sd2_rs0_%g_9.npy" %(eps),retA)
    approximate(0.0)
