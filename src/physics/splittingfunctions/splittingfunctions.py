import numpy as np
from numpy import pi, sqrt
from numba import njit
from src.physics.splittingfunctions import lo, nlo_spacelike, nlo_timelike
from src.parameters import LO, NLO, G_TO_G, G_TO_Q, Q_TO_G, Q_TO_Q, Q_TO_QBAR, Q_TO_QPRIME, TIMELIKE, SPACELIKE

CA = 3.0
CF = 4.0 / 3.0
TR = 0.5
pi = np.pi


@njit(cache=True)
def P_qq(z, alphas, nf, order=LO, mode=SPACELIKE):
    a = alphas/(2*pi)
    lo_term = a * (lo.kq_lo()/(1-z) + lo.rqq_lo(z))
    if order == LO:
        return lo_term
    nlo_term = 0.0
    if mode == SPACELIKE:
        nlo_term = a*a * (nlo_spacelike.kq_nlo(nf)/(1-z) + nlo_spacelike.rqq_ps_nlo(z) + nlo_spacelike.rqq_ns_nlo(z, nf)) 
    else:
        nlo_term = a*a * (nlo_timelike.kq_nlo(nf)/(1-z) + nlo_timelike.rqq_ps_nlo(z) + nlo_timelike.rqq_ns_nlo(z, nf))  
    if order == NLO:
        return lo_term + nlo_term

@njit(cache=True)
def P_gq(z, alphas, nf, order=LO, mode=SPACELIKE):
    a = alphas/(2*pi)
    lo_term = a * lo.rgq_lo(z)
    if order == LO:
        return lo_term
    nlo_term = 0
    if  mode == SPACELIKE:
        nlo = a*a * nlo_spacelike.rgq_nlo(z, nf)
    else:
        nlo = a*a * nlo_timelike.rgq_nlo(z, nf) 
    if order == NLO:
        return lo_term + nlo_term

@njit(cache=True)
def P_qg(z, alphas, order=LO, mode=SPACELIKE):
    a = alphas/(2*pi)
    lo_term = a * lo.rqg_lo(z)
    if order == LO:
        return lo_term
    nlo_term = 0
    if  mode == SPACELIKE:
        nlo_term = a*a * nlo_spacelike.rqg_nlo(z)
    else:
        nlo_term = a*a * nlo_timelike.rqg_nlo(z) 
    if order == NLO:
        return lo_term + nlo_term

@njit(cache=True)
def P_gg(z, alphas, nf, order=LO, mode=SPACELIKE):
    a = alphas/(2*pi)
    lo_term = a * (lo.kg_lo() / (1-z) + lo.rgg_lo(z))
    if order == LO:
        return lo_term
    nlo_term = 0.0
    if  mode == SPACELIKE:
        nlo_term = a*a * (nlo_spacelike.kg_nlo(nf) / (1-z) + nlo_spacelike.rgg_nlo(z, nf))
    else:
        nlo_term = a*a * (nlo_timelike.kg_nlo(nf) / (1-z) + nlo_timelike.rgg_nlo(z, nf))
   
    return lo_term + nlo_term

@njit(cache=True)
def P_qjqi(z, alphas, nf, order=LO, mode=SPACELIKE):
    a = alphas/(2*pi)
    lo_term = 0.0
    if order == LO:
        return 0.0
    if  mode == SPACELIKE:
        nlo_term = a*a * (nlo_spacelike.rqq_ps_nlo(z))
    else:
        nlo_term = a*a * (nlo_timelike.rqq_ps_nlo(z)) 
    if order == NLO:
        return lo_term + nlo_term

@njit(cache=True)
def P_qbarq(z, alphas, nf, order=LO, mode=SPACELIKE):
    a = alphas/(2*pi)
    lo_term = 0.0
    if order == LO:
        return 0.0
    if  mode == SPACELIKE:
        nlo_term = a*a * (nlo_spacelike.rqqbar_ns_nlo(z) + nlo_spacelike.rqq_ps_nlo(z))
    else:
        nlo_term = a*a * (nlo_timelike.rqqbar_ns_nlo(z) + nlo_timelike.rqq_ps_nlo(z)) 
    if order == NLO:
        return lo_term + nlo_term


@njit(cache=True)
def evaluate(splitting, z, alphas, nf, order=LO, mode=SPACELIKE):
    if splitting==G_TO_G:
        return P_gg(z, alphas, nf, order, mode)
    elif splitting==Q_TO_G:
        return P_gq(z, alphas, nf, order, mode)
    elif splitting==Q_TO_Q:
        return P_qq(z, alphas, nf, order, mode)
    elif splitting==G_TO_Q:
        return P_qg(z, alphas, order, mode)
    elif splitting==Q_TO_QPRIME:
        return P_qjqi(z, alphas, nf, order, mode)
    elif splitting==Q_TO_QBAR:
        return P_qbarq(z, alphas, nf, order, mode)
    else:
        raise ValueError("Unknown splitting function")
    
@njit(cache=True)
def warmup():
    evaluate(G_TO_G, 0.1, 0.1, 3, NLO, SPACELIKE)
    evaluate(G_TO_Q, 0.1, 0.1, 3, NLO, SPACELIKE)
    evaluate(Q_TO_G, 0.1, 0.1, 3, NLO, SPACELIKE)
    evaluate(Q_TO_Q, 0.1, 0.1, 3, NLO, SPACELIKE)
    evaluate(Q_TO_QPRIME, 0.1, 0.1, 3, NLO, SPACELIKE)
    evaluate(Q_TO_QBAR, 0.1, 0.1, 3, NLO, SPACELIKE)
    #evaluate(G_TO_G, 0.1, 0.1, 3, NLO, SPACELIKE)