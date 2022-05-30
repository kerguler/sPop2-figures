TSCALE = 4
import numpy
from inferfun import *
import hoppMCMC

import modelD as model
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
        'optopt': {'sigma':[1000.0,10.0],'adapt':10,'chain':50,'hopp':2,'inferpar':[
            5,6,7,8,9,
            10,11,12,13,14,
            15,16,17,18,19,
            24,25,26,27,
            28,29,30,31
        ]}, # for optimisation
        'optim': 2,
        'pargen': model.randomPar, # None, # model.randomPar,
        'param': []
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
        # ][a] for a in [11,9]], # in manuscript (supposed to be, but not)
        # ][a] for a in [11,10,8,0]], # This is very good, but not good enough given the extent of training data
        # ][a] for a in [11,8]], # in manuscript (in reality)
        ][a] for a in [11,7,0,8]], # suggested by the reviewer
        'kernel': 1e-3,
        'pospos': {'eps': [800.0], 'size': 100, 'niter': 1000000, 'resample': False, 'particle': True, 'multivariate': True, 'adapt': numpy.arange(0,1000,5), 'inferpar':[]},
        'pargen': None, # model.randomParPP, # None, # model.randomParPP,
        'optopt': {'sigma':[200.0,5.0],'adapt':10,'chain':50,'hopp':2,'inferpar':[]},
        'optim': 0,
        'param': [2.333596896479785,34.60242052873808,0.01403061266239496,0.6681384703153515,0.07541385593560161,9.012912627605107,53.0347795332536,0.0008435764663306788,0.3758406449748597,0.09643427326970375,-8.47462112012515,50.10372419531102,0.002308130213384392,0.4286163315598315,0.08449021408698712,-7.105814594413852,55.06615674320774,4.439324957229512e-05,0.2218637563072453,0.02579737043085537,4.994489152926351,17.49185878466518,-7.101476533328404,0.7393224742676755,16.34239908067132,21.38213757530077,-12.70390232557846,0.2322060176553762,17.83782327971186,23.56623168180224,-10.92018423369274,0.1037212979795569,13.88371994034933,0.4125379564465729,9.456426309184454]
    }
]

getPosterior(tasklist[0])
