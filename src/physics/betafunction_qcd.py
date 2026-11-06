from numba_src.physics.constants import CA, TR, CF
from numpy import pi
from scipy.special import zeta

def betaLOqcd(nf):
    return  (11*CA - 4*TR * nf) / 3

def betaNLOqcd(nf):
    return 34/3*CA**2 - 20/3*CA*TR*nf - 4*CF*TR*nf

def beta_qcd(nf, order):
    if order == 0:
        return betaLOqcd(nf)
    elif order == 1:
        return betaNLOqcd(nf)

def softgluonNLO(nf):
    return CA*(67./18. - pi**2 / 6.) - 5/9*nf

def softgluonNNLO(nf):
    return CA**2 * ( 245./24. - 67./9.*zeta(2) + 11./6.*zeta(3) + 11./5.*(zeta(2)**2) ) \
           + CF * nf * (-55./24. + 2*zeta(3)) \
           + CA * nf * (-209./108. + 10./9.*zeta(2) - 7./3.*zeta(3)) \
           - 1./27. * nf**2 \
           + pi*betaqcd(nf, 0) / (2*4*pi) * (CA * (808./27. - 28*zeta(3)) - 224./54. * nf)

def softgluon(nf, order):
    if order == 0:
        return 0
    elif order == 1:
        return softgluonNLO(nf)
    elif order == 2:
        return softgluonNNLO(nf)