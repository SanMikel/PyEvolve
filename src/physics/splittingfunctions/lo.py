import numpy as np
from numpy import pi, sqrt
from numba import njit

CA = 3.0
CF = 4.0 / 3.0
TR = 0.5
pi = np.pi

# Terms taken from (https://arxiv.org/pdf/1708.03279)
@njit(cache=True)
def kq_lo():
    return 2*CF  

@njit(cache=True)
def kg_lo():
    return 2*CA

@njit(cache=True)
def rgg_lo(z):
    return 2*CA*( (1-z)/z + z*(1-z) - 1 )

@njit(cache=True)
def rgq_lo(z):
    return CF* ( 1 + (1-z)**2 )/z

@njit(cache=True)
def rqg_lo(z):
    return TR*(z**2 + (1-z)**2)

@njit(cache=True)
def rqq_lo(z):
    return -CF*(1+z)




