from numba_src.EvolutionKernel import EvolutionKernel
from numba_src.interpolation.linear import linear_interp
from numba import njit
import numpy as np
import time
import h5py


@njit(cache=True)
def convolve_kernelxpdf(log_muf2, x_grid, pdf_0, kernel, kernel_w2, mu_grid, z_grid):
    start = time.time()

    interpolator = interp1d(mu_grid, kernel, axis=0)
    interpolator_w2 = interp1d(mu_grid, kernel_w2, axis=0)

    z_centers = 0.5 * (z_grid[:-1] + z_grid[1:])
    z_widths = np.diff(z_grid)

    kernel_z = interpolator(log_muf2)
    kernel_w2_interp = interpolator_w2(log_muf2)

    output = np.zeros_like(x_grid)
    output_w2 = np.zeros_like(x_grid)

    for i, xv in enumerate(x_grid):
        integral_xv = 0.0
        integral_xv_w2 = 0.0
        for j, z in enumerate(z_centers):
            if z < xv:
                continue
            ratio = xv / z
            pdf_val = np.interp(ratio, x_grid, pdf_0)
            integral_xv    += kernel_z[j] * z_widths[j] * pdf_val / z
            integral_xv_w2 += kernel_w2_interp[j] * (z_widths[j] * pdf_val / z) ** 2

        output[i] = integral_xv
        output_w2[i] = integral_xv_w2

    end = time.time()
    #print(f"PDF convolved, time elapsed {end-start:.2f} secs")
    return output, output_w2

def convolve_kernelxpdf_new(log_muf2, x_grid, pdf_0, kernel, kernel_w2, k_mu_grid, k_z_grid):
    start = time.time()
    
    # Grid interpolation, this should go into interpolation source module, 
    # for the moment only uniformly spaced grids
    dx = k_mu_grid[1] - k_mu_grid[0]
    idx = int((log_muf2 - k_mu_grid[0]) / dx)
    frac = (log_muf2 - k_mu_grid[idx]) / dx
    kernel_z = kernel[idx] * (1-frac) + kernel[idx + 1]*frac
    kernel_z_w2 = kernel_w2[idx] * (1-frac) + kernel_w2[idx+1]*frac 

    z_centers = 0.5 * (k_z_grid[:-1] + k_z_grid[1:])
    z_widths = np.diff(k_z_grid)

    ratio = x_grid[:, None] / z_centers[None, :]
    #pdf_vals = np.interp(ratio.ravel(), x_grid, pdf_0).reshape(ratio.shape)
    pdf_vals = np.interp(np.log(ratio), np.log(x_grid), pdf_0)

    valid    = z_centers[None, :] >= x_grid[:, None]
    term     = kernel_z[None, :] * z_widths[None, :] * pdf_vals / z_centers[None, :]
    term_w2  = kernel_z_w2[None, :] * (z_widths[None, :] * pdf_vals / z_centers[None, :]) ** 2

    convolution = np.where(valid, term, 0.0).sum(axis=1)
    convolution_w2 = np.where(valid, term_w2, 0.0).sum(axis=1)

    end = time.time()
    #print(f"PDF convolved, time elapsed {end-start:.2f} secs")

    return convolution, convolution_w2

    


class PDFEvolution:
    def __init__(self, alphas_ref=0.130, mu_ref=91.1876, mu_grid_param=(1.3, 14000, 10000),
                 k_log_mu2_grid_param=(1.3, 14000, 51), k_z_grid_param=(9.9e-7, 1.0, 51),
                 quark_thresholds = [0.0, 0.0, 0.0, 1.47, 4.5, 173.0], order=0, 
                 epsilon=1e-6, n_cores=20, read_kernels=False, n_samples=10_000_000, max_n_splittings=100):
    
        self.epsilon = epsilon    
        self.quark_thresholds = quark_thresholds
        self.order = order
        self.mu_min = k_log_mu2_grid_param[0]
        self.mu_max = k_log_mu2_grid_param[1]

        self.x_min = k_z_grid_param[0]
        self.x_max = k_z_grid_param[1] 

        if read_kernels==False:
            self.evolution_kernel = EvolutionKernel(alphas_ref, mu_ref,
                 mu_grid_param, k_log_mu2_grid_param, k_z_grid_param, 
                 quark_thresholds = quark_thresholds,
                 epsilon = epsilon, n_cores=n_cores, max_n_splittings=max_n_splittings)
            
            self.sudakov = self.evolution_kernel.sudakov
            
            kernel_ig = self.evolution_kernel.generate_kernel_with_resampling(0,   n_samples)
            kernel_iq = self.evolution_kernel.generate_kernel_with_resampling(1, 3*n_samples) 
            #kernel_ig = self.evolution_kernel.generate_evolution_kernel_batch(0, n_samples)
            #kernel_iq = self.evolution_kernel.generate_evolution_kernel_batch(1, 2*n_samples) 

            self.kernel_gg = (kernel_ig[0], kernel_ig[1])
            self.kernel_gq = (kernel_iq[8], kernel_iq[9]) 
            
            self.kernel_qq = (kernel_iq[0], kernel_iq[1])
            self.kernel_qiqj = (kernel_iq[2], kernel_iq[3])  
            self.kernel_qg = (kernel_ig[2], kernel_ig[3])
            
            self.kernel_cg = (kernel_ig[4], kernel_ig[5])
            self.kernel_cq = (kernel_iq[4], kernel_iq[5]) 
            
            self.kernel_bg = (kernel_ig[6], kernel_ig[7])
            self.kernel_bq = (kernel_iq[6], kernel_iq[7]) 

            self.sudakov_g   = (kernel_ig[10], kernel_ig[11])
            self.sudakov_q   = (kernel_iq[10], kernel_iq[11]) 
        
            self.k_z_grid = self.evolution_kernel.k_z_grid
            self.k_log_mu2_grid = self.evolution_kernel.k_log_mu2_grid

        else:
            print('Please provide path to evolution kernels via PDFEvolution.read_kernels(file_name)')
        
        self.g_pdf0    = None; self.u_pdf0    = None
        self.d_pdf0    = None; self.s_pdf0    = None
        self.c_pdf0    = None; self.b_pdf0    = None
        self.ubar_pdf0 = None; self.dbar_pdf0 = None
        self.sbar_pdf0 = None; self.cbar_pdf0 = None
        self.bbar_pdf0 = None;

    
    def set_starting_distribution(self, x_grid, pdf0, parton_flavour):

        self.x_grid = x_grid

        if parton_flavour == 0:
            self.g_pdf0 = pdf0
        elif parton_flavour == 1:
            self.u_pdf0 = pdf0
        elif parton_flavour == 2:
            self.d_pdf0 = pdf0
        elif parton_flavour == 3:
            self.s_pdf0 = pdf0
        elif parton_flavour == 4:
            self.c_pdf0 = pdf0
        elif parton_flavour == 5:
            self.b_pdf0 = pdf0
        
        elif parton_flavour == -1:
            self.ubar_pdf0 = pdf0
        elif parton_flavour == -2:
            self.dbar_pdf0 = pdf0
        elif parton_flavour == -3:
            self.sbar_pdf0 = pdf0
        elif parton_flavour == -4:
            self.cbar_pdf0 = pdf0
        elif parton_flavour == -5:
            self.bbar_pdf0 = pdf0
        
        self.parton_flavour_array = [0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5]
        self.starting_pdf_array   = [self.g_pdf0, 
                                     self.u_pdf0, self.ubar_pdf0,
                                     self.d_pdf0, self.dbar_pdf0,
                                     self.s_pdf0, self.sbar_pdf0,
                                     self.c_pdf0, self.cbar_pdf0,
                                     self.b_pdf0, self.bbar_pdf0]
    
    def convolve_all(self, muf):

        log_muf2 = np.log(muf**2)

        pdf_dic = {}
        for i in range(len(self.parton_flavour_array)):
            parton_flavour = self.parton_flavour_array[i]
            pdf0 = self.starting_pdf_array[i]
            output = np.zeros((10, len(self.x_grid)))
            if pdf0 is None:
                continue

            if parton_flavour == 0:
                kernel_ii = self.kernel_gg
                kernel_qi = self.kernel_qg
                kernel_ci = self.kernel_cg
                kernel_bi = self.kernel_bg
            
            else:
                kernel_ii = self.kernel_qq
                kernel_qi = self.kernel_qiqj
                kernel_ci = self.kernel_cq
                kernel_bi = self.kernel_bq
                kernel_gi = self.kernel_gq

            pdff_ii, pdff_ii_w2 = convolve_kernelxpdf_new(log_muf2, self.x_grid, pdf0, 
                                                    *kernel_ii, 
                                                    self.k_log_mu2_grid, self.k_z_grid)
            pdff_qi, pdff_qi_w2 = convolve_kernelxpdf_new(log_muf2, self.x_grid, pdf0, 
                                                    *kernel_qi, 
                                                    self.k_log_mu2_grid, self.k_z_grid)
            pdff_ci, pdff_ci_w2 = convolve_kernelxpdf_new(log_muf2, self.x_grid, pdf0, 
                                                    *kernel_ci, 
                                                    self.k_log_mu2_grid, self.k_z_grid)
            pdff_bi, pdff_bi_w2 = convolve_kernelxpdf_new(log_muf2, self.x_grid, pdf0, 
                                                    *kernel_bi, 
                                                    self.k_log_mu2_grid, self.k_z_grid)
 
            if parton_flavour == 0:
                pdff_gi = np.zeros_like(self.x_grid)
                pdff_gi_w2 = np.zeros_like(self.x_grid) 
            else:
                pdff_gi, pdff_gi_w2 = convolve_kernelxpdf_new(log_muf2, self.x_grid, pdf0, 
                                                    *kernel_gi, 
                                                    self.k_log_mu2_grid, self.k_z_grid)

            output[0, :] = pdff_ii; output[1, :] = pdff_ii_w2
            output[2, :] = pdff_qi; output[3, :] = pdff_qi_w2
            output[4, :] = pdff_ci; output[5, :] = pdff_ci_w2
            output[6, :] = pdff_bi; output[7, :] = pdff_bi_w2
            output[8, :] = pdff_gi; output[9, :] = pdff_gi_w2      

            pdf_dic[parton_flavour] = output 

        return pdf_dic

    def evolve(self, muf):

        pdf_dic = self.convolve_all(muf)
        n_x = len(self.x_grid)

        evolved = {i : np.zeros(n_x) for i in self.parton_flavour_array}
        evolved_w2 = {i : np.zeros(n_x) for i in self.parton_flavour_array}

        for k, i in enumerate(self.parton_flavour_array):
            pdf_0 = self.starting_pdf_array[k]
            for j, contrib in pdf_dic.items():
                if i == j:
                    slot = 0
                elif i == 0:
                    slot = 8
                elif abs(i) == 4:
                    slot = 4
                elif abs(i) == 5:
                    slot = 6
                else:
                    slot = 2
            
                evolved[i] += contrib[slot]
                evolved_w2[i] += contrib[slot+1]
            if pdf_0 is not None:
                if i!=0:
                    evolved[i] += linear_interp(np.log(muf**2), self.k_log_mu2_grid, self.sudakov_q[0], False, 0.1) * pdf_0
                    evolved_w2[i] += linear_interp(np.log(muf**2), self.k_log_mu2_grid, self.sudakov_q[1], False, 0.1) * pdf_0
                else:
                    evolved[i] += linear_interp(np.log(muf**2), self.k_log_mu2_grid, self.sudakov_g[0], False, 0.1) * pdf_0
                    evolved_w2[i] += linear_interp(np.log(muf**2), self.k_log_mu2_grid, self.sudakov_g[1], False, 0.1) * pdf_0
        
        return self.x_grid, evolved, evolved_w2
    
    def save_kernels(self, file_path):
        kernels = {
            "gg": self.kernel_gg,
            "gq": self.kernel_gq,
            "qq": self.kernel_qq,
            "qiqj": self.kernel_qiqj,
            "qg": self.kernel_qg,
            "cg": self.kernel_cg,
            "cq": self.kernel_cq,
            "bg": self.kernel_bg,
            "bq": self.kernel_bq,
            "suda_g": self.sudakov_g,
            "suda_q": self.sudakov_q
        }
        order = 'LO'
        if self.order==1:
            order='NLO'
    
        #kernel_info = f'# Space-like evolution kernel at {order}, qmin = {self.mu_min} to qmax = {self.mu_max}, xmin = {self.x_min} to xmax = {self.x_max}'

        with h5py.File(file_path, 'w') as f:
            kgrp = f.create_group('kernels')
            for name, (val, aux) in kernels.items():
                g = kgrp.create_group(name)
                g.create_dataset("w", data=val, compression="gzip")
                g.create_dataset("w2",   data=aux,  compression="gzip")
             
            f.create_dataset("x_grid", data=self.evolution_kernel.k_z_grid, compression="gzip")
            f.create_dataset("mu_grid", data=self.evolution_kernel.k_log_mu2_grid, compression="gzip")



            # Metadata
            f.attrs["q_min"] = self.mu_min
            f.attrs["q_max"] = self.mu_max
            f.attrs["epsilon"] = self.epsilon
            f.attrs["quark_thresholds"] = self.quark_thresholds
            f.attrs["order"] = self.order
    
    def read_kernels(self, file_path):
        with h5py.File(file_path, 'r') as f:
            self.kernel_gg = (f['kernels/gg/w'][:], f['kernels/gg/w2'][:])
            self.kernel_gq = (f['kernels/gq/w'][:], f['kernels/gq/w2'][:])
            self.kernel_qg = (f['kernels/qg/w'][:], f['kernels/qg/w2'][:])
            self.kernel_qq = (f['kernels/qq/w'][:], f['kernels/qq/w2'][:])
            self.kernel_qiqj = (f['kernels/qiqj/w'][:], f['kernels/qiqj/w2'][:])
            self.kernel_cg = (f['kernels/cg/w'][:], f['kernels/cg/w2'][:])
            self.kernel_cq = (f['kernels/cq/w'][:], f['kernels/cq/w2'][:])
            self.kernel_bg = (f['kernels/bg/w'][:], f['kernels/bg/w2'][:])
            self.kernel_bq = (f['kernels/bq/w'][:], f['kernels/bq/w2'][:])
            self.sudakov_g = (f['kernels/suda_g/w'][:], f['kernels/suda_g/w2'][:])
            self.sudakov_q = (f['kernels/suda_q/w'][:], f['kernels/suda_q/w2'][:])

            self.k_log_mu2_grid = f['mu_grid'][:]
            self.k_z_grid = f['x_grid'][:]
 
            print(f"Evolution kernels loaded")
            print(f"    q_min: {f.attrs['q_min']}")
            print(f"    q_max: {f.attrs['q_max']}")
            print(f"    Quark thresholds: ")
            print(f"        m_d: {f.attrs['quark_thresholds'][0]} GeV")
            print(f"        m_u: {f.attrs['quark_thresholds'][1]} GeV")
            print(f"        m_s: {f.attrs['quark_thresholds'][2]} GeV")
            print(f"        m_c: {f.attrs['quark_thresholds'][3]} GeV")
            print(f"        m_b: {f.attrs['quark_thresholds'][4]} GeV")

        









            

        
