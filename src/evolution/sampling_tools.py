from numba import njit
import numpy as np
from numba_src.interpolation.linear import linear_interp

@njit(cache=True)
def generate_scale(log_sudakov_start, log_mu2_grid, log_sudakov_grid, uniform, dx):
    '''
      Function to generate emission scale through inverse interpolation
      from the Sudakov form factor grid. As 
    '''
    # Generate new random value for the Sudakov betweem 0,1
    sudakov_random = np.random.rand()
    # Shift the new Sudakov value to generate scale above previous scale
    log_sudakov_new = log_sudakov_start - np.log(sudakov_random) 

    # While the log_mu2_grid is uniformly space, is not the case for the log_sudakov_grid
    uniform=False
    # Check if the new sudakov it's outside the grid boundaries
    if log_sudakov_new >= log_sudakov_grid[-1]:
        return log_mu2_grid[-1], log_sudakov_grid[-1]
    
    # Get new scale
    log_mu2 = linear_interp(log_sudakov_new, log_sudakov_grid, 
                            log_mu2_grid, uniform, dx)
    return log_mu2, log_sudakov_new

@njit(cache=True)
def select_gluon_splitting(log_mu2, log_mu2_grid, log_mu2_grid_uniform, log_mu2_grid_dx, 
                           dmu2_sudakov_qg, dmu2_sudakov_gg):
    '''
      Function to select splitting function for gluon channel according to
      interpolation grids of the splitting function integrals over z:
      11 : g -> g
      21 : g -> q
    '''

    integral_gg = linear_interp(log_mu2, log_mu2_grid, dmu2_sudakov_gg,
                                log_mu2_grid_uniform, log_mu2_grid_dx)
    integral_qg = linear_interp(log_mu2, log_mu2_grid, dmu2_sudakov_qg,
                                log_mu2_grid_uniform, log_mu2_grid_dx)


    integral_g = integral_gg + integral_qg
    ratio_gg = integral_gg / integral_g
    ratio_qg = integral_qg / integral_g

    r = np.random.rand()
    if r < ratio_gg:
        return 11, 1.0 / integral_gg
    else:
        return 21, 1.0 / integral_qg

@njit(cache=True)
def select_quark_splitting(log_mu2, log_mu2_grid, log_mu2_grid_uniform, log_mu2_grid_dx, 
                           dmu2_sudakov_qq, dmu2_sudakov_gq, dmu2_sudakov_qbarq, dmu2_sudakov_qjqi):
    '''
      Function to select splitting function for quark channel according to
      interpolation grids of the splitting function integrals over z:
      12 : q -> g
      22 : q -> q
      23 : q_i -> q_j 
      24 : q_i -> qbar_{i}
      
    '''

    integral_qq    = linear_interp(log_mu2, log_mu2_grid, dmu2_sudakov_qq,
                                   log_mu2_grid_uniform, log_mu2_grid_dx)
    integral_gq    = linear_interp(log_mu2, log_mu2_grid, dmu2_sudakov_gq,
                                   log_mu2_grid_uniform, log_mu2_grid_dx)
    integral_qbarq = linear_interp(log_mu2, log_mu2_grid, dmu2_sudakov_qbarq,
                                   log_mu2_grid_uniform, log_mu2_grid_dx)
    integral_qjqi  = linear_interp(log_mu2, log_mu2_grid, dmu2_sudakov_qjqi,
                                   log_mu2_grid_uniform, log_mu2_grid_dx)

    integral_q = integral_qq + integral_gq + integral_qbarq + integral_qjqi 
    ratio_qq    = integral_qq / integral_q
    ratio_gq    = integral_gq / integral_q
    ratio_qbarq = integral_qbarq / integral_q
    ratio_qjqi  = integral_qjqi / integral_q

    r = np.random.rand()
    if r < ratio_qq:
        return 22, 1.0 / integral_qq
    elif r < ratio_qq + ratio_gq:
        return 12, 1.0 / integral_gq
    elif r < ratio_qq + ratio_gq + ratio_qbarq:
        return 24, 1.0 / integral_qbarq
    else:
        return 23, 1.0 / integral_qjqi

@njit(cache=True)
def select_splitting(parton_in, log_mu2, log_mu2_grid, log_mu2_grid_uniform, log_mu2_grid_dx,
                     dmu2_sudakov_qq, dmu2_sudakov_gq, dmu2_sudakov_qbarq, dmu2_sudakov_qjqi,
                     dmu2_sudakov_qg, dmu2_sudakov_gg):
    
    if parton_in == 0:
        return select_gluon_splitting(log_mu2, log_mu2_grid, 
                                      log_mu2_grid_uniform, log_mu2_grid_dx, 
                                      dmu2_sudakov_qg, dmu2_sudakov_gg)
    else:
        return select_quark_splitting(log_mu2, log_mu2_grid, log_mu2_grid_uniform, log_mu2_grid_dx, 
                           dmu2_sudakov_qq, dmu2_sudakov_gq, dmu2_sudakov_qbarq, dmu2_sudakov_qjqi) 



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

