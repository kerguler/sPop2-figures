import pandas
import numpy
from numpy.random import normal
# This is the Python wrapper for the sPop2 library available from https://github.com/kerguler/sPop2
from spop2 import spop
from spop2 import approximate

prange = [5,50,95]

# Briere function (https://doi.org/10.1093/ee/28.1.22)
def briere1(T,T1,T2,a):
    return a*T*(T-T1)*numpy.sqrt(T2-T)

# Variation in temperature (Gaussian with mean={15,25,35}  and st.dev.=4)
def var_temp(mu, sd):
    if sd == 0:
        return mu
    y = normal(mu, sd)
    y = min(50, max(0, y))
    return y

# Reflection of temperature variation on average development time
def dev_time(temp):
    y = briere1(temp, 0, 50, 1.5e-5)
    if y == 0:
        return 1e3 # Some very high value
    return 1.0 / y

# Simulating a population of 100 individuals for 100 steps under deterministic assumption
# The variation is a result of environmental variation reflected to the daily mean development times
# Development times are assumed to have a 5 step standard deviation regardless of the mean (Erlang-distributed)
def sim(mu, sd, eps=1e-4, verbose=False):
    approximate(eps)
    a = spop(stochastic=False,gamma_fun="MODE_ACCP_ERLANG")
    a.add(0,0,0,0,100)
    ret = []
    ret.append([0, numpy.nan] + a.read())
    for i in numpy.arange(100):
        temp = var_temp(mu, sd)
        dev = dev_time(temp)
        a.iterate(0,dev,5,0,0,0)
        ret.append([i+1,dev]+a.read())
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

def repeat(mu, sd, nrep, eps=1e-4):
    ret = []
    counter = 0
    while counter < nrep:
        ret.append(sim(mu, sd, eps=eps, verbose=False))
        counter = len(ret)
        print(counter)
    return ret

def repeatMPI(mu, sd, nrep, eps=1e-4):
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
                ret = sim(mu, sd, eps=eps)
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
        filename = "mat/run_noise_mu%g_sd%g_n%d_e%g.npy" %(mu,sd,nrep,eps)
        print("Saving %s" %(filename))
        numpy.save(filename, rets)

if __name__ == "__main__":
    repeatMPI(15,4,1000,1e-4)
    repeatMPI(25,4,1000,1e-4)
    repeatMPI(35,4,1000,1e-4)
