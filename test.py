from numba_src.PDFevolution import PDFEvolution
import matplotlib.pyplot as plt
import numpy as np
import lhapdf 
import time

def get_pdf(pdf_name):
    #x_array = np.logspace(np.log10(xmin), np.log10(xmax), nbins)
    pdf = lhapdf.mkPDF(pdf_name)
    qmin = np.sqrt(pdf.q2Min)
    qmax = np.sqrt(pdf.q2Max)
    xmin = pdf.xMin; xmax = pdf.xMax
    threholds = [0.0, 0.0, 0.0, pdf.quarkThreshold(4), pdf.quarkThreshold(5), pdf.quarkThreshold(6)] 
    #print(np.sqrt(pdf.q2Min), pdf.quarkThreshold(4), pdf.quarkThreshold(5))
    return pdf, xmin, xmax, qmin, qmax, threholds

def get_lhapdf_pdfQ(pdf, Q, xgrid):
    g_pdf = np.array([pdf.xfxQ(21, x, Q) for x in xgrid])
    u_pdf = np.array([pdf.xfxQ(2, x, Q) for x in xgrid])
    d_pdf = np.array([pdf.xfxQ(1, x, Q) for x in xgrid])
    s_pdf = np.array([pdf.xfxQ(3, x, Q) for x in xgrid])
    c_pdf = np.array([pdf.xfxQ(4, x, Q) for x in xgrid])
    b_pdf = np.array([pdf.xfxQ(5, x, Q) for x in xgrid]) 
    ubar_pdf = np.array([pdf.xfxQ(-2, x, Q) for x in xgrid])
    dbar_pdf = np.array([pdf.xfxQ(-1, x, Q) for x in xgrid])
    sbar_pdf = np.array([pdf.xfxQ(-3, x, Q) for x in xgrid])
    cbar_pdf = np.array([pdf.xfxQ(-4, x, Q) for x in xgrid]) 
    bbar_pdf = np.array([pdf.xfxQ(-5, x, Q) for x in xgrid])  

    pdfs_at_Q = {
        21: g_pdf,
        2 : u_pdf,
        -2: ubar_pdf,
        1 : d_pdf,
        -1: dbar_pdf,
         3 : s_pdf,
        -3: sbar_pdf,
         4 : c_pdf,
        -4: cbar_pdf,
         5 : b_pdf,
        -5: bbar_pdf
    }

    return pdfs_at_Q
    
 




pdf, xmin, xmax, qmin, qmax, thresholds = get_pdf("CT18LO")

print(type(xmin), type(xmax), type(qmin))
print(xmin)
# Set pyEvolve grids corresponding with evolution
kernel_z_grid = (xmin, xmax, 71)
kernel_q_grid = (qmin, 14000.0, 71)
sudakov_log_mu2_grid = (qmin, 14000.0, 10000)
print(thresholds)

evolution = PDFEvolution(mu_grid_param=sudakov_log_mu2_grid, 
                         k_log_mu2_grid_param=kernel_q_grid,
                         k_z_grid_param = kernel_z_grid, n_samples=10_000_000,
                         quark_thresholds=thresholds)

evolution.save_kernels('kernels_test.h5')
#evolution = PDFEvolution(read_kernels=True)
#evolution.read_kernels('kernels_test.h5')
# Get starting distributions
xgrid    = np.logspace(np.log(xmin), np.log(xmax), num = 200)
pdf_mu0 = get_lhapdf_pdfQ(pdf, qmin, xgrid)

evolution.set_starting_distribution(xgrid, pdf_mu0[21], 0)
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

mu_f = 3.0

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


mu_f = 10.0
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

#print(np.divide(pdff[1][0], pdf_f[0]))
#print(np.divide(np.sqrt(pdff[2][0]), pdf_f[0]))
ax[0].set_xlim(xmin, xmax)
ax[1].set_xlim(xmin, xmax)

plt.savefig(f'test_{mu_f}.pdf')

mu_f = 20.0
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


mu_f = 50.0
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

mu_f = 100.0
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


mu_f = 1000.0
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

plt.show()