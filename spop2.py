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
## ----------------------------------------
## This is a Python wrapper of the sPop2 library, which needs to be installed manually prior to running this code.
## The library and instructions for installation can be found at this link:
##
## https://github.com/kerguler/sPop2
##
## This wrapper requires the library "libspop2py.so", which is generated upon installing the sPop2 library.
## ----------------------------------------

from ctypes import *
import atexit
import numpy
import numpy.ctypeslib as npct
array_1d_double = npct.ndpointer(dtype=numpy.float64, ndim=1, flags='CONTIGUOUS')
array_1d_int = npct.ndpointer(dtype=numpy.int32, ndim=1, flags='CONTIGUOUS')
array_1d_uint = npct.ndpointer(dtype=numpy.uint32, ndim=1, flags='CONTIGUOUS')

gamma_funs = {
"MODE_GAMMA_RAW":0,
"MODE_GAMMA_HASH":1,
"MODE_NBINOM_RAW":2,
"MODE_GAMMA_MATRIX":3,
"MODE_BINOM_RAW":4,
"MODE_ACCP_ERLANG":5,
"MODE_ACCP_FIXED":6,
"MODE_ACCP_PASCAL":7,
"MODE_ACCP_GAMMA":8,
"MODE_ACCP_CASWELL":9
}

filename = "/usr/local/lib/libspop2py.so"
model = cdll.LoadLibrary(filename)

label = "GSL_RNG for Python".encode('utf-8')
model.rng_setup.restype = None
model.rng_setup.argtypes = [c_char_p]
model.rng_setup(label)
atexit.register(model.rng_destroy)

def approximate(val):
    model.set_APPROX.restype = None
    model.set_APPROX.argtypes = [c_double]
    model.set_APPROX(val)

class spop:
    def __init__(self,stochastic,gamma_fun):
        """
        Initiates an spop2 population
        :param stochastic:
        :param gamma_fun:
        :param accumulative:
        """
        if not (gamma_fun in gamma_funs):
            print("Gamma function needs to be one of these:",gamma_funs)
            return
        #
        self.id = None
        self.fun_add = False
        self.fun_iterate = False
        self.fun_read = False
        self.fun_retrieve = False
        #
        fun = model.spoplib_init
        fun.restype = c_uint
        fun.argtypes = [c_char,c_char]
        self.id = fun(stochastic,gamma_funs[gamma_fun])
        #
    def add(self,age,devcycle,development,stage,size):
        if not self.fun_add:
            self.fun_add = model.spoplib_add
            self.fun_add.restype = None
            self.fun_add.argtypes = [c_uint,c_uint,c_uint,c_uint,c_double,c_double]
        self.fun_add(self.id,age,devcycle,development,stage,numpy.float(size))
        #
    def iterate(self,dev_prob,dev_mean,dev_sd,death_prob,death_mean,death_sd):
        if not self.fun_iterate:
            self.fun_iterate = model.spoplib_iterate
            self.fun_iterate.restype = None
            self.fun_iterate.argtypes = [c_uint,c_double,c_double,c_double,c_double,c_double,c_double]
        self.fun_iterate(self.id,dev_prob,dev_mean,dev_sd,death_prob,death_mean,death_sd)
        #
    def retrieve(self,devtable=0):
        if not self.fun_retrieve:
            self.fun_retrieve = model.spoplib_retrieve
            self.fun_retrieve.restype = None
            self.fun_retrieve.argtypes = [c_uint,c_char,array_1d_double,array_1d_double,array_1d_uint]
        dev = numpy.ndarray(1000,dtype=numpy.float64)
        size = numpy.ndarray(1000,dtype=numpy.float64)
        limit = numpy.ndarray(1,dtype=numpy.uint32)
        self.fun_retrieve(self.id,devtable,dev,size,limit)
        return [dev[:limit[0]],size[:limit[0]]]
        #
    def read(self):
        if not self.fun_read:
            self.fun_read = model.spoplib_read
            self.fun_read.restype = None
            self.fun_read.argtypes = [c_uint,array_1d_double,array_1d_double,array_1d_double]
        size = numpy.array(0, dtype=numpy.float64, ndmin=1)
        developed = numpy.array(0, dtype=numpy.float64, ndmin=1)
        dead = numpy.array(0, dtype=numpy.float64, ndmin=1)
        self.fun_read(self.id,size,developed,dead)
        return [size[0],developed[0],dead[0]]
        #
    def destroy(self):
        fun = model.spoplib_destroy
        fun.restype = None
        fun.argtypes = [c_uint]
        fun(self.id)
        #
    def console(self):
        fun = model.spoplib_print
        fun.restype = None
        fun.argtypes = [c_uint]
        fun(self.id)
        #
    def destroy_all(self):
        fun = model.spoplib_destroy_all
        fun.restype = None
        fun.argtypes = []
        fun()
