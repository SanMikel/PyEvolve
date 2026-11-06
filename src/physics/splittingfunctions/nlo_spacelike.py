from src.physics.constants import CA, CF, TR
from numpy import pi, sqrt, log
from numba import njit
import numpy as np


@njit(cache=True)
def li2_series(z, precision_lim = 1e-10):
    '''
      Function to compute the li2 series at given precision
    '''
    result  = z
    z_term = z
    k = 2
    precision = 1.0
    while  precision > precision_lim and k < 200:
        z_term *= z 
        li2_term = z_term / (k*k)
        result += li2_term
        k += 1
        precision = np.abs(li2_term / result)
    return result

@njit(cache=True)
def li2(z, precision_lim = 1e-10):
    '''
      Function to compute the li2 series using Li2
      relations 
    '''
    if z == 0.0:
        return 0.0
    elif z == 1:
        return pi**2/6.
    elif z <= 0.5:
        return li2_series(z, precision_lim)
    elif z > 0.5:
        z_mirror = 1.0-z
        li2_mirror = li2_series(z_mirror, precision_lim)
        return pi**2/6. - log(z)*log(z_mirror) - li2_mirror    
    else:
        return 0.0

@njit(cache=True)
def s2(z, precision_lim = 1e-10):
    return -2 * li2(-z) + 0.5 * log(z)**2 - 2*log(z)*log(1+z) - pi**2/6.0

@njit(cache=True)
def Gamma(nf):
    return CA*(67/18 - pi**2 / 6) - TR*nf*10/9

'''
 Terms taken from (https://arxiv.org/pdf/1708.03279)
 Real splitting kernel: 
    P_ab^{R} = K_ab/(1-z) + R_ab 
    K_ab is diagonal in flavour K_ab = k_a * delta_ab

    non-singlet terms  -> ns
    pure singlet terms -> ps
'''


@njit(cache=True)
def kq_nlo(nf):
    return 2*CF*Gamma(nf)

@njit(cache=True)
def kg_nlo(nf):
    return 2*CA*Gamma(nf)

@njit(cache=True)
def _pqq(z):
    return 2./(1-z) - 1 - z

@njit(cache=True)
def _pqg(z):
    return z**2 + (1-z)**2

@njit(cache=True)
def _pgq(z):
    return (1 + (1-z)**2) / z

@njit(cache=True)
def _pgg(z):
    return 1/ (1-z) + 1/z -2 + z*(1-z)

@njit(cache=True)
def rgq_nlo(z, nf):
    logz = log(z)
    logz2 = logz * logz
    log1mz = log(1 - z) 
    log1mz2 = log1mz*log1mz
    pgqz = _pgq(z)
    cf_term   = (
        -5./2. - 7./2. * z + (2+7./2.*z)*logz + (0.5*z -1)*logz2  
        -2*z*log1mz - ( 3*log1mz + log1mz2  )*pgqz 
    )
    
    cfca_term = (
        28./9. + 65./18. *z + 44./9 * z**2 + (-12 -5*z - 8./3. *z**2) * logz  
        +(4.+z)*logz2 + 2*z*log1mz 
        +(-2.*logz*log1mz + 0.5*logz2 + 11./3.*log1mz + log1mz2 - pi**2/6. + 0.5)*pgqz   
        +_pgq(-z)*s2(z) 
    )
    
    cftrnf_term = -4./3.*z - (20./9. + 4./3.*log1mz)* pgqz

    return CF**2 * cf_term + CF*CA * cfca_term + CF*TR*nf * cftrnf_term

@njit(cache=True)
def rqg_nlo(z):
    logz = log(z)
    logz2 = logz * logz
    log1mz = log(1 - z) 
    log1mz2 = log1mz*log1mz 
    pqgz = _pqg(z)
    pi2 = pi*pi

    cftr_term = (
        4. -9.*z + (-1.+4.*z)*logz + (-1.+2.*z)*logz2 + 4.*log1mz 
        +(-4.*logz*log1mz + 4.*logz +2.*logz2 - 4.*log1mz + 2*log1mz2
        -2./3. * pi2 + 10)*pqgz  
    )
    catr_term = (
        182./9. + 14./9.*z + 40./(9.*z) + (136./3.*z -38./3.)*logz
        -4.*log1mz - (2.+8.*z)*logz2 + (-logz2 + 44./3.*logz - 2.*log1mz2
        +4.*log1mz + pi2/3. - 218./9.)*pqgz + 2.*_pqg(-z)*s2(z)     
    )

    return 0.5*CF*TR*cftr_term + 0.5*CA*TR*catr_term

@njit(cache=True)
def rgg_nlo(z, nf):
    logz = log(z)
    logz2 = logz * logz
    log1mz = log(1 - z) 
    log1mz2 = log1mz*log1mz 
    pggz = _pgg(z)
    pi2 = pi*pi
    z2 = z*z

    cftrnf_term = (
        -16. + 8.*z + 20./3.*z2 + 4./(3.*z) + (-6.-10.*z)*logz + (-2.-2.*z)*logz2 
    )

    catrnf_term = (
        2.-2.*z + 26./9.*z2 - 26./(9.*z) - 4./3.*(1+z)*logz - 20./9.*(1./z -2. + z - z2)
    )

    ca2_term = (
        27./2.*(1.-z) + 67./9.*(z2-1./z) + (-25./3. + 11./3.*z -44./3.*z2)*logz
        +4.*(1+z)*logz2 + 2*_pgg(-z)*s2(z) + (-4.*logz*log1mz + logz2)*pggz
        +(67./9.-pi2/3.)*(1./z - 2 + z - z2)
    )

    return CF*TR*nf*cftrnf_term + CA*TR*nf*catrnf_term + CA*CA*ca2_term

@njit(cache=True)
def rqqbar_ns_nlo(z):
    return CF*(CF - CA/2.)* (2.*_pqq(-z)*s2(z) + 2.*(1.+z)*log(z) + 4.*(1.-z))

@njit(cache=True)
def rqq_ns_nlo(z, nf):
    logz = log(z)
    logz2 = logz * logz
    log1mz = log(1 - z) 
    log1mz2 = log1mz*log1mz 
    pqqz = _pqq(z)
    pi2 = pi*pi
    z2 = z*z 

    cf2_term = (
        - (2.*logz*log1mz + 3./2.*logz)*pqqz - (3./2. + 7./2.*z)*logz
        - 0.5*(1.+z)*logz2 - 5.*(1-z)
    )

    cfca_term = (
        (0.5*logz2 + 11./6.*logz)*pqqz - (1.+z)*(67./18.-pi2/6.)
        +(1.+z)*logz + 20./3.*(1.-z)
    )

    cftrnf_term = (
        -2./3.*logz*pqqz + 10./9.*(1.+z) - 4./3.*(1.-z)
    )

    return CF*CF*cf2_term + CF*CA*cfca_term + CF*TR*nf*cftrnf_term

@njit(cache=True)
def rqq_ps_nlo(z):
    logz = log(z)
    logz2 = logz * logz
    z2 = z*z 
    trnfcf_term = (
        20./(9.*z) - 2. +6.*z -56./9.*z2 
        + (1 + 5.*z + 8./3.*z2)*logz - (1+z)*logz2
    )
    return TR*CF*trnfcf_term

@njit(cache=True)
def rqq_nlo(z, nf):
    logz = log(z)
    logz2 = logz * logz
    log1mz = log(1 - z) 
    log1mz2 = log1mz*log1mz 
    pqqz = _pqq(z)
    pqqmz = _pqq(-z)
    pi2 = pi*pi
    z2 = z*z 

    cf2_term = (
        -1. + z + (0.5 - 3./2.*z)*logz - 0.5*(1.+z)*logz2 + 2*pqqmz*s2(z)
        -(3./2.*logz + 2.*logz*log1mz )*pqqz
    )
    cfca_term = (
        14./3.*(1.-z) - pqqmz*s2(z) + (11./6.*logz +0.5*logz2)*pqqz 
        - (z+1.)*(67./18. - pi2/6.)
    )
    cftrnf_term = (
        -16./3. + 40./3.*z + (10.*z + 16./3.*z2 + 2.)*logz - 112./9.*z2 
        +40./(9.*z) - 2.*(1+z)*logz2 - 2./3.*logz*pqqz + 10./9.*(z+1.)
    )

    return CF*CF*cf2_term + CF*CA*cfca_term + CF*TR*nf*cftrnf_term
