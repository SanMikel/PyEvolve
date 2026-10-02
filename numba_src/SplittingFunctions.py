import numpy as np
from numpy import pi, sqrt
from numba import njit
import numba_src.SplittingFunctionsLO as LO
import numba_src.SplittingFunctionsNLO as NLO

CA = 3.0
CF = 4.0 / 3.0
TR = 0.5
pi = np.pi


# For the moment we only implement leading order terms
@njit(cache=True)
def Pqq(z, alphas, nf, order=0):
    a = alphas/(2*pi)
    lo = a * (LO.kq_lo()/(1-z) + LO.rqq_lo(z))
    if order==0:
        return lo
    nlo = a*a * (NLO.kq_nlo(nf)/(1-z) + NLO.rqq_nlo(z, nf)) 
    if order==1:
        return lo + nlo

@njit(cache=True)
def Pgq(z, alphas, nf, order=0):
    a = alphas/(2*pi)
    lo = a * LO.rgq_lo(z)
    if order==0:
        return lo
    nlo = a*a * NLO.rgq_nlo(z, nf)
    if order==1:
        return lo + nlo

@njit(cache=True)
def Pqg(z, alphas, order=0):
    a = alphas/(2*pi)
    lo = a * LO.rqg_lo(z)
    if order==0:
        return lo
    nlo = a*a * NLO.rqg_nlo(z)
    if order==1:
        return lo + nlo

@njit(cache=True)
def Pgg(z, alphas, nf, order=0):
    a = alphas/(2*pi)
    lo = a * (LO.kg_lo() / (1-z) + LO.rgg_lo(z))
    if order==0:
        return lo
    nlo = a*a * (NLO.kg_nlo(nf) / (1-z) + NLO.rgg_nlo(z, nf))
    if order==1:
        return lo + nlo

@njit(cache=True)
def evaluate(splitting, z, alphas, nf, order=0):
    if splitting==11:
        return Pgg(z, alphas, nf, order)
    elif splitting==12:
        return Pgq(z, alphas, nf, order)
    elif splitting==22:
        return Pqq(z, alphas, nf, order)
    elif splitting==21:
        return Pqg(z, alphas, order)
    else:
        raise ValueError("Unknown splitting function")
    


class SplittingFunctions:

    def __init__(self):
        print("Warming up splitting functions")
        # Warm up numba evaluation
        evaluate(11, 0.1, 3, 0.1, 1)
        evaluate(12, 0.1, 3, 0.1, 1)
        evaluate(22, 0.1, 3, 0.1, 1)
        evaluate(21, 0.1, 0.1, 1)

    def Pgg(self, z, alphas, order=0):
        return evaluate(11, z, alphas, order)
    def Pgq(self, z, alphas, order=0):
        return evaluate(12, z, alphas, order) 
    def Pqq(self, z, alphas, order=0):
        return evaluate(22, z, alphas, order) 
    def Pqg(self, z, alphas, order=0):
        return evaluate(21, z, alphas, order)  

    def __call__(self, splitting, z, alphas, order=0):
        return evaluate(splitting, z, alphas, order)



