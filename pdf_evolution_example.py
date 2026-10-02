from numba_src.PDFevolution import PDFEvolution
import matplotlib.pyplot as plt
import numpy as np
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

