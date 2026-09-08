from numba_src.AlphaQCD import AlphaQCD
from numba_src.SudakovFormFactor import SudakovFormFactor
import numpy as np
import matplotlib.pyplot as plt
from numba_src.SplittingFunctions import Pqq, Pqg, Pgq, Pgg

alpha_qcd_lo = AlphaQCD()
alpha_qcd_lo.tabulate()
alpha_qcd_nlo = AlphaQCD(order=1)
alpha_qcd_nlo.tabulate()

x = alpha_qcd_lo.mu_grid
y_lo  = alpha_qcd_lo.alphas_grid
y_nlo = alpha_qcd_nlo.alphas_grid

fig, ax = plt.subplots()

ax.plot(x, y_lo)
ax.plot(x, y_nlo)

ax.set_xscale('log')


z = np.linspace(1e-5, 1-1e-5, 200)
nf=3
alphas_lo = y_lo[100]
alphas_nlo = y_nlo[100]
pqq_lo = np.zeros(200); pqq_nlo = np.zeros_like(pqq_lo)
pgg_lo = np.zeros(200); pgg_nlo = np.zeros_like(pqq_lo)
pgq_lo = np.zeros(200); pgq_nlo = np.zeros_like(pqq_lo)
pqg_lo = np.zeros(200); pqg_nlo = np.zeros_like(pqq_lo)
for i, c in enumerate(z):
    pqq_lo[i] = Pqq(c, alphas_lo, nf, order=0)
    pqq_nlo[i] = Pqq(c, alphas_nlo, nf, order=1)

    pqg_lo[i] = Pqg(c, alphas_lo, order=0)
    pqg_nlo[i] = Pqg(c, alphas_nlo, order=1)

    pgq_lo[i] = Pgq(c, alphas_lo, nf, order=0)
    pgq_nlo[i] = Pgq(c, alphas_nlo, nf, order=1)

    pgg_lo[i] = Pgg(c, alphas_lo, nf, order=0)
    pgg_nlo[i] = Pgg(c, alphas_nlo, nf, order=1)

fig, ax = plt.subplots(2, 2, figsize=(10,10), tight_layout=True)
ax = ax.flatten()
ax[0].plot(z, pqq_lo)
ax[0].plot(z, pqq_nlo)

ax[1].plot(z, pqg_lo)
ax[1].plot(z, pqg_nlo)

ax[2].plot(z, pgq_lo)
ax[2].plot(z, pgq_nlo)

ax[3].plot(z, pgg_lo)
ax[3].plot(z, pgg_nlo)

for i in range(4):
    #ax[i].set_yscale('log')
    ax[i].set_ylim(-1, 1)


sudakov = SudakovFormFactor(alpha_qcd = alpha_qcd_nlo, order=1)
sudakov.tabulate()

fig, ax = plt.subplots()
ax.plot(x, np.exp(-sudakov.log_sudakov_gg), label='gg')
ax.plot(x, np.exp(-sudakov.log_sudakov_qq), label='qq')
ax.plot(x, np.exp(-sudakov.log_sudakov_gq), label='gq')
ax.plot(x, np.exp(-sudakov.log_sudakov_qg), label='qg')
ax.plot(x, np.exp(-sudakov.log_sudakov_g), label='g')
ax.plot(x, np.exp(-sudakov.log_sudakov_q), label='q')
ax.set_yscale('log')
ax.set_xscale('log')

plt.show()


#kernel = EvolutionKernel()