from numba_src.PDFevolution import PDFEvolution
import matplotlib.pyplot as plt
import numpy as np
import apfelpy as ap 
import time
xmin = 1e-5
xmax = 1.0

def dummy_pdf(x):
    xuv = 5.107200 * x**0.8 * (1 - x)**3
    xdv = 3.064320 * x**0.8 * (1 - x)**4
    xg = 1.7 * x**-0.1 * (1 - x)**5
    xdbar = 0.1939875 * x**-0.1 * (1 - x)**6
    xubar = xdbar * (1 - x)
    xs = 0.2 * (xubar + xdbar)
    return {
        0: xg,
        1: xuv + xubar, -1: xubar,
        2: xdv + xdbar, -2: xdbar,
        3: xs, -3: xs,
    }


xgrid    = np.logspace(np.log(xmin), np.log(xmax), num = 200)
dummies = dummy_pdf(xgrid)

evolution = PDFEvolution(n_samples=5_000_000, n_cores=10)
evolution.set_starting_distribution(xgrid, dummies[0], 0)
evolution.set_starting_distribution(xgrid, dummies[1], 1)
evolution.set_starting_distribution(xgrid, dummies[-1],-1)
evolution.set_starting_distribution(xgrid, dummies[2], 2)
evolution.set_starting_distribution(xgrid, dummies[-2], -2)
evolution.set_starting_distribution(xgrid, dummies[3], 3)
evolution.set_starting_distribution(xgrid, dummies[-3], -3)

x, pdf, pdf_w2 = evolution.evolve(10.0)

fig, ax = plt.subplots(tight_layout=True)
ax.errorbar(xgrid, pdf[0], np.sqrt(pdf_w2[0]), label='$g$')
ax.errorbar(xgrid, pdf[1], np.sqrt(pdf_w2[1]), label='$u_{v}$')
ax.errorbar(xgrid, pdf[2], np.sqrt(pdf_w2[2]), label='$d_{v}$')
ax.errorbar(xgrid, pdf[-1], np.sqrt(pdf_w2[-1]), label='$\\bar{u}$')
ax.errorbar(xgrid, pdf[-2], np.sqrt(pdf_w2[-2]), label='$\\bar{d}$')
ax.errorbar(xgrid, pdf[-3], np.sqrt(pdf_w2[-3]), label='$\\bar{s}$')

ax.set_yscale("log")
ax.set_xscale("log")

ax.set_xlim(xmin, xmax)
ax.set_ylabel("$xf_{a}(x, Q^{2})$", fontsize=12)
ax.set_xlabel("$x$", fontsize=12)
ax.set_title("$Q^{2}=m_{Z}^{2}$", loc="left")
ax.legend(frameon=False)


plt.show()




'''
evolution.set_starting_distribution(xgrid, dummy_pdf(xgrid, 0.1, 0), 0)
evolution.set_starting_distribution(xgrid, pdf_mu0[2], 1)
evolution.set_starting_distribution(xgrid, pdf_mu0[-2], -1)
evolution.set_starting_distribution(xgrid, pdf_mu0[1], 2)
evolution.set_starting_distribution(xgrid, pdf_mu0[-1], -2)
evolution.set_starting_distribution(xgrid, pdf_mu0[-3], -3)
evolution.set_starting_distribution(xgrid, pdf_mu0[3], 3)

mu_f = 1.31

start = time.time()
pdff = evolution.evolve(mu_f)
end = time.time()
print(f'{end-start:.4f} secs')
pdf_f = get_lhapdf_pdfQ(pdf, mu_f, xgrid)
#pdf_f = get_pdf("HERAPDF20_LO_EIG", mu_f)

fig, ax = plt.subplots(1,2, figsize=(5*2, 5), tight_layout=True)

ax[0].set_xscale('log')
ax[0].set_yscale('log')
ax[0].errorbar(xgrid, pdff[1][0], yerr=np.sqrt(pdff[2][0]))
ax[0].plot(xgrid, pdf_f[21])
ax[0].set_ylabel('xg(x, Q)')
ax[0].set_title(f'{mu_f}')
ax[0].set_xlabel('x')


ax[1].set_xscale('log')
ax[1].set_yscale('log')
ax[1].errorbar(xgrid, pdff[1][1], yerr=np.sqrt(pdff[2][1]))
ax[1].plot(xgrid, pdf_f[2])
ax[1].set_ylabel('xu(x, Q)')
ax[1].set_title(f'{mu_f}')
ax[1].set_xlabel('x')

ax[0].set_xlim(xmin, xmax)
ax[1].set_xlim(xmin, xmax)

#print(np.divide(pdff[1][0], pdf_f[0]))
#print(np.divide(np.sqrt(pdff[2][0]), pdf_f[0]))

plt.savefig(f'test_{mu_f}.pdf')
'''