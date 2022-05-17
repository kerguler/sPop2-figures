import numpy
from scipy.stats import multivariate_normal

from mpi4py import MPI
MPI_SIZE = MPI.COMM_WORLD.Get_size()
MPI_RANK = MPI.COMM_WORLD.Get_rank()
MPI_MASTER = 0
worker_indices = numpy.delete(numpy.arange(MPI_SIZE),MPI_MASTER)
print("Process %d of %d is running" %(MPI_RANK,MPI_SIZE),flush=True)

# https://darrenjw.wordpress.com/2010/08/15/metropolis-hastings-mcmc-algorithms/
def mcmc(pr,fun,lower,upper,kernel,niter=1000,thin=10,sig=1.0,verbose=False):
    acc = 0
    pr = numpy.array(pr)
    prn = pr
    scrn = scr = fun(pr)
    mat = [[scr] + pr.tolist()]
    for n in numpy.arange(niter):
        while True:
            prn = kernel * numpy.random.randn(pr.shape[0]) + pr
            if all(prn>=lower) and all(prn<=upper):
                break
        scrn = fun(prn)
        if scrn==scrn and (scrn < scr or numpy.log(numpy.random.random()) < (scr-scrn)/sig):
            pr = prn
            scr = scrn
            acc += 1
        if n % thin == 0:
            if verbose:
                vec = [scr] + pr.tolist()
                mat.append(vec)
                print("%d,%g,%s" %(n,acc,",".join([str(tmp) for tmp in vec])))
            acc = 0
    return(numpy.array(mat))

def abc_smc(pargen,score,epsseq,size,lower,upper,kernel,niter,inferpar=[],mat_init=[],resample=True,adapt=[],particle=False,multivariate=False,verbose=True):
    '''
    pargen: function to generate parameter samples from the prior
    score: distance function
    epsseq: a list of thresholds for each niter steps will be taken
    size: individual parameter sets to sample
    lower - upper: lower and upper bounds of parameter values (independent of infervar)
    kernel: standard deviation of transition kernel (scalar or vector of size inferpar)
    niter: number of iterations per eps
    inferpar: indices of parameters to infer
    mat_init: initial population (if not to be sampled from prior)
    resample: iterate each parameter set individually (if False, deviates from ABC-SMC) (default: True)
    adapt: list of steps at which to perform kernel adaptation (step numbering is overall and independent of niter and epsseq)
    particle: if True, larger kernel when particles are close to each other (deviates from ABC-SMC) (default: False)
    multivariate: sample using a multivariate normal distribution instead of a set of univariates (default: False)
    '''
    from scipy.stats import norm
    def abc_smc_sim(mat,index,init=False):
        fcount = 0
        while True:
            if init:
                pr_new = numpy.array(pargen())
            else:
                if resample:
                    v = mat[numpy.random.choice(numpy.arange(mat.shape[0]),size=1,p=mat[:,0],replace=True)[0],:]
                else:
                    v = mat[index,:]
                #
                if particle:
                    sigma = -numpy.log(mat[index,0])
                else:
                    sigma = 1.0
                #
                pr_new = v[3:].copy()
                if multivariate:
                    pr_new[inferpar] = numpy.random.multivariate_normal(mean = v[3+inferpar], cov = sigma * kernel)
                else:
                    pr_new[inferpar] = sigma * kernel * numpy.random.randn(infersize) + v[3+inferpar]
                #
            if all(pr_new >= lower) and all(pr_new <= upper):
                scr = score(pr_new,verbose=False)
                if not numpy.isnan(scr) and scr < eps:
                    break
            fcount += 1
        if init:
            w = 1.0 / size
        else:
            if multivariate:
                # w = 1.0 / numpy.sum([v[0]*numpy.prod(multivariate_normal.pdf(pr_new[inferpar], mean=v[3+inferpar], cov=kernel)) for v in mat])
                w = 1.0 / numpy.sum([v[0]*numpy.prod(norm.pdf(pr_new[inferpar],loc=v[3+inferpar],scale=numpy.diag(kernel)**0.5)) for v in mat])
            else:
                w = 1.0 / numpy.sum([v[0]*numpy.prod(norm.pdf(pr_new[inferpar],loc=v[3+inferpar],scale=kernel)) for v in mat])
        return [w,fcount,scr]+pr_new.tolist()
    #
    MPI.COMM_WORLD.Barrier()
    parsize = len(pargen())
    inferpar = numpy.array(inferpar)
    if len(inferpar)==0:
        inferpar = numpy.arange(parsize)
    infersize = len(inferpar)
    #
    kernel = numpy.array(kernel).copy()
    if len(kernel) == 1:
        kernel = numpy.repeat(kernel,infersize)
    if len(kernel.shape) == 1 and multivariate:
        kernel = numpy.diag(kernel).copy()
    if multivariate:
        kernel **= 2.0
    kernel_base = numpy.array(kernel).copy()
    if MPI_RANK==MPI_MASTER and verbose:
        if multivariate:
            print("kernel.sd: "+str(0)+","+",".join(["%g" %(m) for m in numpy.diagonal(kernel)]))
        else:
            print("kernel.sd: "+str(0)+","+",".join(["%g" %(m) for m in kernel]))
    #
    if len(mat_init) and mat_init.shape[0]==size and mat_init.shape[1]==3+parsize:
        print("Resuming from previous state...")
        mat = mat_init.copy()
        resume = True
    else:
        mat = numpy.zeros((size,3+parsize))
        resume = False
    mat_new = mat.copy()
    lepsseq = len(epsseq)
    #
    for i in numpy.arange(niter * lepsseq):
        MPI.COMM_WORLD.Barrier()
        if i % niter == 0:
            if len(epsseq):
                eps = epsseq.pop(0)
                if verbose:
                    print("EPS:", MPI_RANK, eps)
            else:
                break
        #
        for n in numpy.arange(MPI_RANK,mat_new.shape[0],MPI_SIZE):
            tmp = abc_smc_sim(mat,n,init = (not resume) and (i==0))
            mat_new[n,:] = tmp
            #
            if verbose:
                print("Matrix:", n, "(%g)" %(tmp[1]))
            #
            if MPI_RANK==MPI_MASTER:
                if MPI_SIZE>1:
                    m = 1
                    for worker in worker_indices:
                        if n+m < mat_new.shape[0]:
                            ntmp = MPI.COMM_WORLD.recv(source=worker, tag=1)
                            mat_new[ntmp[0],:] = ntmp[1:]
                            m += 1
            else:
                MPI.COMM_WORLD.send([n]+tmp, dest=MPI_MASTER, tag=1)
        mat_new[:,0] /= numpy.sum(mat_new[:,0])
        mat = mat_new.copy()
        MPI.COMM_WORLD.Barrier()
        mat = MPI.COMM_WORLD.bcast(mat, root=MPI_MASTER)
        #
        if len(adapt) and (i in adapt):
            tmp = numpy.cov(mat[:,3+inferpar].T)
            a = numpy.diag_indices_from(tmp)
            tmp[ tmp[a]<kernel_base[a], tmp[a]<kernel_base[a] ] = kernel_base[ tmp[a]<kernel_base[a], tmp[a]<kernel_base[a] ]
            if multivariate:
                kernel = tmp.copy()
                if MPI_RANK==MPI_MASTER:
                    if verbose:
                        print("kernel.sd: "+str(i)+","+",".join(["%g" %(m) for m in numpy.diagonal(kernel)]))
            else:
                kernel = (numpy.diag(tmp).copy())**0.5
                if MPI_RANK==MPI_MASTER:
                    if verbose:
                        print("kernel.sd: "+str(i)+","+",".join(["%g" %(m) for m in kernel]))
        #
        if MPI_RANK==MPI_MASTER:
            if verbose:
                print("param.mat: "+str(i)+","+("\nparam.mat: "+str(i)+",").join([",".join(["%g" %(mm) for mm in m]) for m in mat]))
    return mat
