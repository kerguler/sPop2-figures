TSCALE = 4
import numpy
from inferfun import *
import hoppMCMC

import modelB as model
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
        'pospos': {'eps': [1000.0], 'size': 100, 'niter': 100, 'resample': False, 'particle': True, 'multivariate': True, 'adapt': numpy.arange(0,1000,5),'inferpar':[
            3,4,5,
            6,7,8,
            16,17,18,19,
            20,21,22,23
        ]},
        'optopt': {'sigma':[500.0,5.0],'adapt':10,'chain':MPI_SIZE,'hopp':2,'inferpar':[
            3,4,5,
            6,7,8,
            16,17,18,19,
            20,21,22,23
        ]},
        'optim': 0,
        'pargen': None, # model.randomParQ,
        'param': [0,0,0,7.002156377512208,-17.53216981591055,0.004904754218633921,28.18614333276416,-16.67410419928161,0.006089471933767313,25,-19,0.005,0,50,0,1,9.906128563826497,49.62219359742059,-14.07664193467981,0.1557269337787638,11.62162733610185,47.55316836739943,-12.13487161521901,0.2024732575311672,0,0,0]
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
        # ][a] for a in [11,7,0,8]], # suggested by the reviewer
        ][a] for a in [6,11,7,0,8]], # suggested by the reviewer
        'kernel': 1e-3,
        'pospos': {'eps': [1000.0], 'size': 100, 'niter': 100, 'resample': False, 'particle': True, 'multivariate': True, 'adapt': numpy.arange(0,1000,5), 'inferpar':[]},
        'pargen': None, # model.randomParPP,
        'optopt': {'sigma':[500.0,5.0],'adapt':10,'chain':MPI_SIZE,'hopp':2,'inferpar':[]},
        'optim': 0,
        'param': [20.19099516750302,-12.2492626221371,0.02075824680270736,21.41482723418393,-13.84228576666895,0.001223424826238019,2.289873800010128,-18.46884439264853,0.0004273099204426801,18.92921244749423,-16.61033160183414,0.002119452744123731,19.09434727881188,29.58655279517001,-0.1759272647891741,0.4247621465110369,10.3414012325555,49.83285570355365,-13.93195602373159,0.1660498027376169,15.94968285588442,49.39011384932345,-11.22286677191957,0.2050205700903959,13.55522951941849,0.923830326312553,9.664287777609927]
    }
]

getPosterior(tasklist[1])
