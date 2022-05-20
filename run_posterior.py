TSCALE = 4
import numpy
from inferfun import *
import hoppMCMC

import model
import ftemp
import vtemp
import culex
import priorQ

from mpi4py import MPI
MPI_SIZE = MPI.COMM_WORLD.Get_size()
MPI_RANK = MPI.COMM_WORLD.Get_rank()
MPI_MASTER = 0

def getRandom(score,fun):
    while True:
        par = fun()
        if score(par) < 1e13:
            break
    return par

def getPosterior(task):
    filename = "mat/posterior_%s.csv" %(task['label'])
    score = model.getScores(task['obs'])
    if len(task['param'])==0:
        task['param'] = getRandom(score,task['pargen'])
    if task['optim'] == 1:
        kernel = numpy.array(task['kernel'],dtype=numpy.float64) * (model.upper - model.lower)
        scr, pr = mcmc(task['param'],score,model.lower,model.upper,kernel,niter=1000,thin=10,sig=1.0,verbose=True)
    elif task['optim'] == 2:
        if len(task['optopt']['inferpar']):
            inferpar = numpy.array(task['optopt']['inferpar'])
        else:
            inferpar = numpy.arange(len(task['param']))
        varmat = numpy.diag(numpy.repeat(1e-6,len(inferpar)))
        hop = hoppMCMC.hoppMCMC(score,
              task['param'],
              varmat,
              inferpar=inferpar,
              gibbs=True,
              rangeT=[task['optopt']['sigma'][1],task['optopt']['sigma'][0]],
              model_comp=task['optopt']['sigma'][0],
              num_hopp=task['optopt']['hopp'],
              num_adapt=task['optopt']['adapt'],
              num_chain=MPI_SIZE,
              chain_length=task['optopt']['chain'])
        pr = hop.parmat[0,1:]
        scr = hop.parmat[0,0]
    else:
        if len(task['pospos']['inferpar']):
            inferpar = numpy.array(task['pospos']['inferpar'])
        else:
            inferpar = numpy.arange(len(task['param']))
        kernel = numpy.array(task['kernel'],dtype=numpy.float64) * (model.upper - model.lower)[inferpar]
        def pargen():
            return task['param']
        mat = abc_smc(pargen if task['pargen']==None else task['pargen'],
              score,
              mat_init = [],
              epsseq = task['pospos']['eps'],
              lower = model.lower,
              upper = model.upper,
              kernel = kernel,
              inferpar = inferpar,
              size = task['pospos']['size'],
              niter = task['pospos']['niter'],
              resample = task['pospos']['resample'],
              adapt = task['pospos']['adapt'],
              multivariate = task['pospos']['multivariate'],
              verbose = True)
        numpy.savetxt(filename, mat, delimiter=",")

tasklist = [
    {
        'label': 'Cxquin',
        'obs': [[
            culex.obs['15-1'],
            culex.obs['15-2'],
            culex.obs['15-3'],
            culex.obs['20-1'],
            culex.obs['20-2'],
            culex.obs['20-3'],
            culex.obs['23-1'],
            culex.obs['23-2'],
            culex.obs['23-3'],
            culex.obs['27-1'],
            culex.obs['27-2'],
            culex.obs['27-3'],
            culex.obs['30-1'],
            culex.obs['30-2'],
            culex.obs['30-3']
        ][a] for a in [0,3,6,9,12]],
        'kernel': 1e-3,
        'pospos': {'eps': [10000.0], 'size': 1000, 'niter': 1000000, 'resample': False, 'particle': True, 'multivariate': True, 'adapt': numpy.arange(0,1000,5)},
        'optopt': {'sigma':[1000.0,10.0],'adapt':10,'chain':50,'hopp':8,'inferpar':[
            5,6,7,8,9,
            10,11,12,13,14,
            15,16,17,18,19,
            24,25,26,27,
            28,29,30,31
        ]}, # for optimisation
        'optim': 0,
        'pargen': None, # model.randomPar,
        'param': numpy.array([6.57398e+00,4.49960e+01,3.96071e-01,3.42160e-01,8.58386e-01,1.14825e+01,1.94776e+01,4.91615e-03,2.97761e-01,1.82851e-01,-2.67349e+00,4.13958e+01,1.54554e-02,6.65068e-01,4.25905e-02,-4.18135e-01,5.57574e+01,1.28583e-01,2.47986e-02,7.88965e-01,3.21150e+01,2.06405e+00,2.20412e+02,1.29821e+02,7.61155e-01,2.17173e-01,1.83414e+01,3.70145e+01,2.29322e+01,2.18493e+02,7.17811e-03,1.94256e-01,1.45248e+01,3.35249e+01,5.15903e+00,2.96453e+02,1.58669e-01,3.95734e-01,14,1.0])
    },
    {
        'label': 'Cxpip_photo',
        'obs': [[
            vtemp.obs[1], # 0
            vtemp.obs[2], # 1
            vtemp.obs[3], # 2
            vtemp.obs[4], # 3
            vtemp.obs[5], # 4
            vtemp.obs[6], # 5
            vtemp.obs[7], # 6
            vtemp.obs[8], # 7
            vtemp.obs[9], # 8
            vtemp.obs[10], # 9
            vtemp.obs[11], # 10
            vtemp.obs[12], # 11
            vtemp.obs[13], # 12
            vtemp.obs[14], # 13
            vtemp.obs[15], # 14
            vtemp.obs[16], # 15
        # ][a] for a in [11,9]], # in manuscript 
        ][a] for a in [3,11,12,8,9,10]], # to be tested
        'kernel': 1e-3,
        'pospos': {'eps': [400.0], 'size': 100, 'niter': 1000000, 'resample': False, 'particle': True, 'multivariate': True, 'adapt': numpy.arange(0,1000,5), 'inferpar':[]},
        'pargen': model.randomParPP, # None, # model.randomParPP,
        'optopt': {'sigma':[200.0,10.0],'adapt':10,'chain':50,'hopp':8,'inferpar':[]},
        'optim': 2,
        'param': []
    }
]

getPosterior(tasklist[1])
