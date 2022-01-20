import pandas
import numpy
from numpy.random import normal
# This is the Python wrapper for the sPop2 library available from https://github.com/kerguler/sPop2
from spop2 import spop
from spop2 import approximate

prange = [5,50,95]

# Variation in temperature (Gaussian with mean=30 and st.dev.=10)
def var_temp():
    return normal(30.0, 10.0)

# Reflection of temperature variation on average development time
def dev_time(temp, mode):
    if mode == 'fixed':
        return 30.0
    elif mode == 'linear':
        return numpy.max([1.0, temp]) # Development time of less than 1 unit is not allowed
    elif mode == 'non-linear':
        return temp + numpy.abs(temp - 30.0)

# Simulating a population of 100 individuals for 50 steps under deterministic assumption
# The variation is a result of environmental variation reflected to the daily mean development times
# Development times are assumed to have a 5 step standard deviation regardless of the mean (Erlang-distributed)
def sim(mode, eps=1e-4, verbose=False):
    approximate(eps)
    a = spop(stochastic=False,gamma_fun="MODE_ACCP_ERLANG")
    a.add(0,0,0,0,100)
    ret = []
    ret.append([0, numpy.nan] + a.read())
    for i in numpy.arange(50):
        temp = var_temp()
        mu = dev_time(temp, mode)
        a.iterate(0,mu,5,0,0,0)
        ret.append([i+1,mu]+a.read())
        if verbose:
            print("Step %d" %(i))
    approximate(0.0)
    return ret

def summarize(ret):
    tmp = numpy.percentile(ret, prange, axis=0)
    cols = ('step','devtime','size','developed','dead')
    return {
        'lower': pandas.DataFrame(tmp[0],columns=cols),
        'median': pandas.DataFrame(tmp[1],columns=cols),
        'higher': pandas.DataFrame(tmp[2],columns=cols)
    }

def repeat(mode, nrep, eps=1e-4):
    ret = []
    counter = 0
    while counter < nrep:
        ret.append(sim(mode, eps=eps, verbose=False))
        counter = len(ret)
        print(counter)
    return ret

def repeatMPI(mode, nrep, eps=1e-4):
    from mpi4py import MPI
    MPI_SIZE = MPI.COMM_WORLD.Get_size()
    MPI_RANK = MPI.COMM_WORLD.Get_rank()
    MPI_MASTER = 0
    worker_indices = numpy.delete(numpy.arange(MPI_SIZE),MPI_MASTER)
    #
    if MPI_RANK==MPI_MASTER:
        rets = []
    count = 0
    for index in numpy.arange(MPI_RANK,nrep,MPI_SIZE):
        ret = []
        while ret == []:
            try:
                ret = sim(mode, eps=eps)
            except:
                pass
        if MPI_RANK==MPI_MASTER:
            count += 1
            rets.append(ret)
            for worker in worker_indices:
                if count >= nrep:
                    break
                ret = MPI.COMM_WORLD.recv(source=worker, tag=1)
                count += 1
                rets.append(ret)
            print("Status: %d" %count)
        else:
            MPI.COMM_WORLD.send(ret, dest=MPI_MASTER, tag=1)
    if MPI_RANK==MPI_MASTER:
        rets = numpy.array(rets)
        filename = "mat/run_noise_m%s_n%d_e%g.npy" %(mode,nrep,eps)
        print("Saving %s" %(filename))
        numpy.save(filename, rets)

if __name__ == "__main__":
    # repeatMPI('linear',1000,5e-2)
    repeatMPI('non-linear',1000,5e-2)
