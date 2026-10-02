from numba import njit
import numpy as np
from numba_src import SplittingFunctions

@njit(cache=True)
def sample_z_uniform(z_min, z_max):
    z = np.random.uniform(z_min, z_max)
    weight = (z_max - z_min)
    return z, weight    

@njit(cache=True)
def sample_z_harddiv(z_min, z_max):
    r = np.random.rand() 
    z = z_min * (z_max / z_min) ** r
    norm = np.log(z_max/z_min)
    weight = z*norm
    return z, weight

@njit(cache=True)
def sample_z_softdiv(z_min, z_max):
    r = np.random.rand()
    z = 1 - (1 - z_min) * ((1 - z_max)/(1 - z_min))**r
    norm = np.log((1-z_min)/(1-z_max))
    weight = (1-z)*norm

    return z, weight

@njit(cache=True)
def sample_z_pgg(z_min, z_max):
    Lh = np.log(z_max/z_min)
    Ls = np.log((1-z_min)/(1-z_max))
    z = 1.0
    if np.random.rand() < 0.5:
        z, _ = sample_z_harddiv(z_min, z_max)
    else:
        z, _ = sample_z_softdiv(z_min, z_max)

    qmix = 0.5/(z*Lh) + 0.5/((1-z)*Ls)
    weight = 1.0/qmix
    return z, weight

@njit(cache=True)
def sample_z(splitting, z_min, z_max, order=0):
    z = 1.0
    w_sampling = 1.0
    if splitting==11:
        z, w_sampling = sample_z_softdiv(z_min, z_max)
    elif splitting==22:
        z, w_sampling = sample_z_softdiv(z_min, z_max)
    else:
        z, w_sampling = sample_z_uniform(z_min, z_max)

    return z,  w_sampling

