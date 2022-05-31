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
        'pospos': {'eps': [800.0], 'size': 100, 'niter': 1000000, 'resample': False, 'particle': True, 'multivariate': True, 'adapt': numpy.arange(0,1000,5),'inferpar':[
            5,6,7,8,9,
            10,11,12,13,14,
            15,16,17,18,19,
            24,25,26,27,
            28,29,30,31
        ]},
        'optopt': {'sigma':[1000.0,10.0],'adapt':10,'chain':50,'hopp':2,'inferpar':[
            5,6,7,8,9,
            10,11,12,13,14,
            15,16,17,18,19,
            24,25,26,27,
            28,29,30,31
        ]},
        'optim': 0,
        'pargen': None, # model.randomPar, # None, # model.randomPar,
        'param': [59.63313825018248,19.84630435201534,0.2216030629236236,0.4112582043763288,0.09652621342827468,-1.99284559599864,35.27102433485844,0.005321295593566686,0.6532987905821004,0.04051886444525318,-6.356244060121677,40.52121423657937,0.01416163592046722,0.5901862133327224,0.04272473166735254,-7.392498334672977,11.7679368555553,0.1279267552618746,0.01302761550875341,0.03488470892063795,52.41498421210706,35.05916957592218,-16.89090375889076,0.783393412964936,9.366464215252105,58.95778915652801,-14.46258157580646,0.1511751571316022,11.05156853358932,58.36723392444317,-12.35046128343488,0.1723124169268649,0,0,0]
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
        'pospos': {'eps': [800.0], 'size': 100, 'niter': 1000000, 'resample': False, 'particle': True, 'multivariate': True, 'adapt': numpy.arange(0,1000,5), 'inferpar':[]},
        'pargen': None, # model.randomParPP, # None, # model.randomParPP,
        'optopt': {'sigma':[200.0,5.0],'adapt':10,'chain':50,'hopp':2,'inferpar':[]},
        'optim': 0,
        'param': [-1.146734844825423,37.42663865168799,0.01167377702751094,0.6923808046141008,0.07549965463597716,8.197388342078137,52.74758863384428,0.00106179096830298,0.3960806515071563,0.02311315443031847,0.6627733478779261,48.90660179582061,0.002985342086801476,0.5449186750484895,0.02817394424257101,6.134267898343062,35.42725447498584,0.001565157673188426,0.4040003706336209,0.07007470545101775,0.4781839059480877,23.50124712113245,-8.525423939735367,0.1777469883706828,11.81765929979703,41.05581132095108,-13.75614149547356,0.1911448515306042,18.44598691674348,12.39193281205724,-9.781787932632978,0.1311308584333077,13.58335263736845,0.8281493115276526,9.640622829036097]
    }
]

getPosterior(tasklist[0])
