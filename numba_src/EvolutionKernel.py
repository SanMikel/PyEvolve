from numba_src.AlphaQCD import AlphaQCD
from numba_src.SplittingFunctions import SplittingFunctions, evaluate
from numba_src.parton_evolution import evolve_parton
from numba_src.SudakovFormFactor import SudakovFormFactor, generate_scale, select_splitting
from numba_src.z_sampling import sample_z
from numba_src.banner import print_banner
from numba_src.interpolation.linear import linear_interp
from numba import njit, prange, set_num_threads
from numba_src.resampling import advance_particle, systematic_resample, advance_population
import numpy as np
import matplotlib.pyplot as plt
import time

@njit(cache=True)
def fill_row_from_population(g, parton, z_cum, w_cum, parton_in, k_z_grid, bin_widths, n_samples,
                             z_ii_w, z_ii_w2, z_qi_w, z_qi_w2,
                             z_ci_w, z_ci_w2, z_bi_w, z_bi_w2,
                             z_gi_w, z_gi_w2, delta_w, delta_w2):
    n_z = len(k_z_grid) - 1
    for i in range(len(parton)):
        if z_cum[i] == 1.0:
            delta_w[g]  += w_cum[i] / n_samples
            delta_w2[g] += w_cum[i]**2 / n_samples**2
            continue

        bin_id = np.searchsorted(k_z_grid, z_cum[i]) - 1
        if bin_id < 0 or bin_id >= n_z:
            continue

        bw = bin_widths[bin_id]
        weight  = w_cum[i] / bw / n_samples
        weight2 = w_cum[i]**2 / bw**2 / n_samples**2
        p = parton[i]

        if p == parton_in:
            z_ii_w[g, bin_id] += weight;  z_ii_w2[g, bin_id] += weight2
        elif p == -1 or abs(p) == 2 or abs(p) == 3:
            z_qi_w[g, bin_id] += weight/5.0;  z_qi_w2[g, bin_id] += weight2/25.0
        elif abs(p) == 4:
            z_ci_w[g, bin_id] += weight/2.0;  z_ci_w2[g, bin_id] += weight2/4.0
        elif abs(p) == 5:
            z_bi_w[g, bin_id] += weight/2.0;  z_bi_w2[g, bin_id] += weight2/4.0
        elif p == 0:
            z_gi_w[g, bin_id] += weight;  z_gi_w2[g, bin_id] += weight2

@njit(cache=True)
def generate_samples( parton_in, mu_0, mu_f,
                      log_mu2_grid, log_mu2_grid_uniform, log_mu2_grid_dx,
                      log_delta_q_grid, log_delta_g_grid, alphas_grid,
                      dmu2_sudakov_qq_grid, dmu2_sudakov_qg_grid, 
                      dmu2_sudakov_gq_grid, dmu2_sudakov_gg_grid,
                      log_thresholds2, n_samples=100_000, epsilon=1e-6, max_n_splittings=100):
    
    log_mu2_start = np.log(mu_0**2)
    log_mu2_finish = np.log(mu_f**2)

    mu_all      = np.zeros((n_samples, max_n_splittings))
    z_all       = np.zeros((n_samples, max_n_splittings))
    parton_all  = np.zeros((n_samples, max_n_splittings))
    w_all       = np.zeros((n_samples, max_n_splittings))
    n_all       = np.zeros(n_samples,  dtype=np.int32)
    for i in range(n_samples):
        mu, z, parton, w, n = evolve_parton(parton_in, log_mu2_start, log_mu2_finish,
                                            log_mu2_grid, log_mu2_grid_uniform, log_mu2_grid_dx,
                                            log_delta_q_grid, log_delta_g_grid, alphas_grid,
                                            dmu2_sudakov_qq_grid, dmu2_sudakov_qg_grid, 
                                            dmu2_sudakov_gq_grid, dmu2_sudakov_gg_grid, 
                                            log_thresholds2, epsilon, max_n_splittings)
        n_all[i]      = n
        mu_all[i, :n]     = mu[:n]
        z_all[i, :n]      = z[:n]
        parton_all[i, :n] = parton[:n]
        w_all[i, :n]      = w[:n]
        

    return mu_all, z_all, parton_all, w_all, n_all

# Work in progress do not use
@njit(parallel=True, cache=True)
def generate_samples_batch( parton_in, mu_0, mu_f,
                      log_mu2_grid, log_mu2_grid_uniform, log_mu2_grid_dx,
                      log_delta_q_grid, log_delta_g_grid, alphas_grid,
                      dmu2_sudakov_qq_grid, dmu2_sudakov_qg_grid, 
                      dmu2_sudakov_gq_grid, dmu2_sudakov_gg_grid,
                      log_thresholds2, n_samples=100_000, epsilon=1e-6, max_n_splittings=100):
    
    log_mu2_start = np.log(mu_0**2)
    log_mu2_finish = np.log(mu_f**2)

    mu_all      = np.zeros((n_samples, max_n_splittings))
    z_all       = np.zeros((n_samples, max_n_splittings))
    parton_all  = np.zeros((n_samples, max_n_splittings))
    w_all       = np.zeros((n_samples, max_n_splittings))
    n_all       = np.zeros(n_samples,  dtype=np.int32)
    for i in prange(n_samples):
        mu, z, parton, w, n = evolve_parton(parton_in, log_mu2_start, log_mu2_finish,
                                            log_mu2_grid, log_mu2_grid_uniform, log_mu2_grid_dx,
                                            log_delta_q_grid, log_delta_g_grid, alphas_grid,
                                            dmu2_sudakov_qq_grid, dmu2_sudakov_qg_grid, 
                                            dmu2_sudakov_gq_grid, dmu2_sudakov_gg_grid, 
                                            log_thresholds2, epsilon, max_n_splittings)
        n_all[i]      = n
        mu_all[i, :n]     = mu[:n]
        z_all[i, :n]      = z[:n]
        parton_all[i, :n] = parton[:n]
        w_all[i, :n]      = w[:n]

    return mu_all, z_all, parton_all, w_all, n_all

@njit(cache=True)
def construct_evolution_kernel_old(parton_in, k_mu, k_z, k_parton, k_w, k_n, mu_grid, z_grid, n_samples):
    n_mu = len(mu_grid)
    n_z = len(z_grid) - 1
    z_ii_w   = np.zeros((n_mu, n_z)); z_ii_w2  = np.zeros((n_mu, n_z))
    z_qi_w   = np.zeros((n_mu, n_z)); z_qi_w2 = np.zeros((n_mu, n_z))
    z_ci_w   = np.zeros((n_mu, n_z)); z_ci_w2  = np.zeros((n_mu, n_z))
    z_bi_w   = np.zeros((n_mu, n_z)); z_bi_w2  = np.zeros((n_mu, n_z))
    z_gi_w   = np.zeros((n_mu, n_z)); z_gi_w2  = np.zeros((n_mu, n_z))
    delta_w  = np.zeros(n_mu)          # <-- NEW: no-splitting weight sum
    delta_w2 = np.zeros(n_mu)          # <-- NEW: its variance
    widths = np.diff(z_grid)
    for g in range(n_mu):
        log_mu2_step = mu_grid[g]
        n_underflow = 0 
        for j in range(n_samples):
            n_emissions = k_n[j]
            z_prod = 1.0
            w_prod = 1.0
            k_emissions = 0
            for k in range(n_emissions):
                if k_mu[j, k] >= log_mu2_step:
                    break
                z_prod *= k_z[j, k]
                w_prod *= k_w[j, k]
                k_emissions += 1
                if z_prod < z_grid[0]:
                    n_underflow += 1

            parton_out = k_parton[j, k_emissions] if k_emissions < n_emissions else parton_in
            if k_emissions == 0:
                # no splitting has occurred yet: goes into the delta term, not the histogram
                delta_w[g]  += 1.0 / n_samples
                delta_w2[g] += 1.0 / n_samples**2
                continue
            bin_id = np.searchsorted(z_grid, z_prod) - 1
            if bin_id < 0 or bin_id >= n_z:
                continue
            bin_width = widths[bin_id]
            weight  = w_prod / bin_width / n_samples
            weight2 = w_prod**2 / bin_width**2 / n_samples**2
            if parton_out == parton_in:
                z_ii_w[g, bin_id]  += weight
                z_ii_w2[g, bin_id] += weight2
            elif parton_out == -1 or abs(parton_out) == 2 or abs(parton_out) == 3:
                z_qi_w[g, bin_id]  += weight / 5.0
                z_qi_w2[g, bin_id] += weight2 / 25.0
            elif abs(parton_out) == 4:
                z_ci_w[g, bin_id]  += weight / 2.0
                z_ci_w2[g, bin_id] += weight2 / 4.0
            elif abs(parton_out) == 5:
                z_bi_w[g, bin_id]  += weight / 2.0
                z_bi_w2[g, bin_id] += weight2 / 4.0
            elif parton_out == 0:
                z_gi_w[g, bin_id]  += weight
                z_gi_w2[g, bin_id] += weight2

    return (z_ii_w, z_ii_w2, z_qi_w, z_qi_w2, z_ci_w, z_ci_w2,
            z_bi_w, z_bi_w2, z_gi_w, z_gi_w2, delta_w, delta_w2)


@njit(cache=True)
def construct_evolution_kernel(parton_in, k_mu, k_z, k_parton, k_w, k_n, mu_grid, z_grid, 
                                                         z_ii_w, z_ii_w2, z_qi_w, z_qi_w2,
                                                         z_ci_w, z_ci_w2, z_bi_w, z_bi_w2,
                                                         z_gi_w, z_gi_w2, delta_w, delta_w2,
                                                         underflow_w, underflow_w2,
                                                         n_events, n_samples=-99):
    n_mu = len(mu_grid)
    n_z = len(z_grid) - 1

    widths = np.diff(z_grid)

    if n_samples == -99:
        n_samples = n_events
    # scratch: cumulative products for ONE event, reused every iteration
    max_steps = k_mu.shape[1]
    z_cum = np.empty(max_steps)
    w_cum = np.empty(max_steps)

    for j in range(n_events):
        n_emissions = k_n[j]

        # precompute cumulative products ONCE per event
        z_prod = 1.0
        w_prod = 1.0
        for k in range(n_emissions):
            z_prod *= k_z[j, k]
            w_prod *= k_w[j, k]
            z_cum[k] = z_prod
            w_cum[k] = w_prod

        # sweep mu_grid, advancing the pointer forward only (never resets)
        k_emissions = 0
        for g in range(n_mu):
            log_mu2_step = mu_grid[g]

            # advance pointer while the NEXT emission is still below this mu
            while k_emissions < n_emissions and k_mu[j, k_emissions] < log_mu2_step:
                k_emissions += 1

            if k_emissions == 0:
                delta_w[g]  += 1.0 / n_samples
                delta_w2[g] += 1.0 / n_samples**2
                continue

            z_val = z_cum[k_emissions - 1]
            w_val = w_cum[k_emissions - 1]
            parton_out = k_parton[j, k_emissions - 1] 

            bin_id = np.searchsorted(z_grid, z_val) - 1
            if bin_id < 0:
                underflow_w[g] += w_val / n_samples
                underflow_w2[g] += w_val**2 / n_samples**2
                continue
            if bin_id >= n_z:
                continue

            bin_width = widths[bin_id]
            weight  = w_val / bin_width / n_samples
            weight2 = w_val**2 / bin_width**2 / n_samples**2

            if parton_out == parton_in:
                z_ii_w[g, bin_id]  += weight;  z_ii_w2[g, bin_id]  += weight2
            elif parton_out == -1 or abs(parton_out) == 2 or abs(parton_out) == 3:
                z_qi_w[g, bin_id]  += weight/5.0;  z_qi_w2[g, bin_id]  += weight2/25.0
            elif abs(parton_out) == 4:
                z_ci_w[g, bin_id]  += weight/2.0;  z_ci_w2[g, bin_id]  += weight2/4.0
            elif abs(parton_out) == 5:
                z_bi_w[g, bin_id]  += weight/2.0;  z_bi_w2[g, bin_id]  += weight2/4.0
            elif parton_out == 0:
                z_gi_w[g, bin_id]  += weight;  z_gi_w2[g, bin_id]  += weight2

    return (z_ii_w, z_ii_w2, z_qi_w, z_qi_w2, z_ci_w, z_ci_w2,
            z_bi_w, z_bi_w2, z_gi_w, z_gi_w2, delta_w, delta_w2, underflow_w, underflow_w2)

def renormalise_kernel_rows(kernels, k_z_grid):

    has_underflow = (len(kernels) == 14)

    if has_underflow:
        (z_ii_w, z_ii_w2, z_qi_w, z_qi_w2,
         z_ci_w, z_ci_w2, z_bi_w, z_bi_w2,
         z_gi_w, z_gi_w2, delta_w, delta_w2,
         underflow_w, underflow_w2) = kernels

    else:
        (z_ii_w, z_ii_w2, z_qi_w, z_qi_w2,
         z_ci_w, z_ci_w2, z_bi_w, z_bi_w2,
         z_gi_w, z_gi_w2, delta_w, delta_w2) = kernels 

    bin_widths = np.diff(k_z_grid)

    total_per_row = (z_ii_w + 5.0*z_qi_w + 2.0*z_ci_w + 2.0*z_bi_w + z_gi_w) @ bin_widths

    total_per_row += delta_w

    #if has_underflow:
    #    total_per_row += underflow_w


    safe_total = np.where(total_per_row > 1e-300, total_per_row, 1.0)
    scale = 1.0 / safe_total
    scale2 = scale**2

    z_ii_w_r = z_ii_w * scale[:, None]; z_ii_w2_r = z_ii_w2 *scale2[:, None]
    z_qi_w_r = z_qi_w * scale[:, None]; z_qi_w2_r = z_qi_w2 *scale2[:, None]
    z_ci_w_r = z_ci_w * scale[:, None]; z_ci_w2_r = z_ci_w2 *scale2[:, None]
    z_bi_w_r = z_bi_w * scale[:, None]; z_bi_w2_r = z_bi_w2 *scale2[:, None]
    z_gi_w_r = z_gi_w * scale[:, None]; z_gi_w2_r = z_gi_w2 *scale2[:, None]
    delta_w_r = delta_w * scale; delta_w2_r = delta_w2 * scale2

    if has_underflow:
        underflow_w_r = underflow_w * scale; underflow_w2_r = underflow_w2 * scale2
        return (z_ii_w_r, z_ii_w2_r, z_qi_w_r, z_qi_w2_r,
         z_ci_w_r, z_ci_w2_r, z_bi_w_r, z_bi_w2_r,
         z_gi_w_r, z_gi_w2_r, delta_w_r, delta_w2_r,
         underflow_w_r, underflow_w2_r) 
    
    else:
        return (z_ii_w_r, z_ii_w2_r, z_qi_w_r, z_qi_w2_r,
         z_ci_w_r, z_ci_w2_r, z_bi_w_r, z_bi_w2_r,
         z_gi_w_r, z_gi_w2_r, delta_w_r, delta_w2_r) 


class EvolutionKernel:

    def __init__(self, alphas_ref=0.13, mu_ref=91.1876,
                 mu_grid_param=(1.3, 14000, 10000),
                 k_log_mu2_grid_param=(1.3, 14000, 51), k_z_grid_param=(1e-6, 1.0, 51),
                 quark_thresholds = [0.0, 0.0, 0.0, 1.47, 4.5, 173.0], 
                 epsilon=1e-6, n_cores=1, max_n_splittings=100, order=0):

        self.order=order

        self.mu_0 = mu_grid_param[0]
        self.mu_f = mu_grid_param[1]
        self.epsilon = epsilon
        self.max_n_splittings = max_n_splittings
        self.print_banner()
        print(quark_thresholds)
        self.alpha_qcd = AlphaQCD(mu_ref=mu_ref, alphas_ref=alphas_ref, thresholds=quark_thresholds, order=self.order)
        self.alpha_qcd.tabulate(mu_grid_param)
        self.log_thresholds2 = self.alpha_qcd.log_thresholds2
        self.alphas_interp_info = self.alpha_qcd.alphas_interp_info
        
        self.sudakov = SudakovFormFactor(self.alpha_qcd, epsilon=epsilon, mu_grid_param=mu_grid_param, order=self.order)
        self.sudakov.tabulate()
        
        self.log_sudakov_qq  = self.sudakov.log_sudakov_qq
        self.log_sudakov_qg  = self.sudakov.log_sudakov_qg
        self.log_sudakov_gq  = self.sudakov.log_sudakov_gq
        self.log_sudakov_gg  = self.sudakov.log_sudakov_gg
        self.log_sudakov_q = self.sudakov.log_sudakov_q
        self.log_sudakov_g = self.sudakov.log_sudakov_g 

        self.dmu2_sudakov_qq = self.sudakov.dmu2_sudakov_qq
        self.dmu2_sudakov_qg = self.sudakov.dmu2_sudakov_qg
        self.dmu2_sudakov_gq = self.sudakov.dmu2_sudakov_gq
        self.dmu2_sudakov_gg = self.sudakov.dmu2_sudakov_gg


        # Create grid for Evolution kernel:
        self.k_log_mu2_grid = np.linspace(np.log(k_log_mu2_grid_param[0]**2), np.log(k_log_mu2_grid_param[1]**2), k_log_mu2_grid_param[2])
        n_z_bins = k_z_grid_param[2] // 5 + 1
        k_z_grid_lowx = np.linspace(np.log(k_z_grid_param[0]), np.log(0.01), n_z_bins)
        k_z_grid_lowmidx = np.linspace(np.log(0.01), np.log(0.1), n_z_bins) 
        k_z_grid_highmidx = np.linspace(np.log(0.1), np.log(0.4), n_z_bins)
        k_z_grid_lowhighx = np.linspace(np.log(0.4), np.log(0.9), n_z_bins)
        k_z_grid_highx = np.linspace(np.log(0.9), np.log(k_z_grid_param[1]), n_z_bins)
        k_z_grid = np.unique(np.concatenate([k_z_grid_lowx, k_z_grid_lowmidx, k_z_grid_highmidx, k_z_grid_lowhighx, k_z_grid_highx]))
        self.k_z_grid = np.exp(k_z_grid)
        #self.k_z_grid = np.logspace(np.log10(k_z_grid_param[0]), np.log10(k_z_grid_param[1]), k_z_grid_param[2])
  
        
        print("Warming up functions")
        if n_cores == 1:
            generate_samples(0, self.mu_0, self.mu_f,
                             self.sudakov.log_mu2_grid, self.sudakov.uniform, self.sudakov.dx,
                             self.log_sudakov_q, self.log_sudakov_g, self.alphas_interp_info[1],
                             self.dmu2_sudakov_qq, self.dmu2_sudakov_qg, 
                             self.dmu2_sudakov_gq, self.dmu2_sudakov_gg,
                             self.log_thresholds2, epsilon=self.epsilon, n_samples=1)
        else:
            set_num_threads(n_cores)
            generate_samples_batch(0, self.mu_0, self.mu_f,
                                   self.sudakov.log_mu2_grid, self.sudakov.uniform, self.sudakov.dx,
                                   self.log_sudakov_q, self.log_sudakov_g, self.alphas_interp_info[1],
                                   self.dmu2_sudakov_qq, self.dmu2_sudakov_qg, 
                                   self.dmu2_sudakov_gq, self.dmu2_sudakov_gg,
                                   self.log_thresholds2, epsilon=self.epsilon, n_samples=1)
    

    def generate_evolution_kernel_batch(self, parton_in, n_samples=1_000_000):
        print(f'Generating {n_samples} samples for {parton_in}')
        start = time.time()

        n_mu = len(self.k_log_mu2_grid)
        n_z = len(self.k_z_grid) - 1
        
        z_ii_w   = np.zeros((n_mu, n_z)); z_ii_w2  = np.zeros((n_mu, n_z))
        z_qi_w   = np.zeros((n_mu, n_z)); z_qi_w2  = np.zeros((n_mu, n_z))
        z_ci_w   = np.zeros((n_mu, n_z)); z_ci_w2  = np.zeros((n_mu, n_z))
        z_bi_w   = np.zeros((n_mu, n_z)); z_bi_w2  = np.zeros((n_mu, n_z))
        z_gi_w   = np.zeros((n_mu, n_z)); z_gi_w2  = np.zeros((n_mu, n_z))
        delta_w  = np.zeros(n_mu);        delta_w2 = np.zeros(n_mu)
        underflow_w = np.zeros(n_mu); underflow_w2 = np.zeros(n_mu)
        
        if n_samples < 2_000_000:
            k_mu, k_z, k_parton, k_w, k_n =  generate_samples_batch(parton_in, self.mu_0, self.mu_f,
                                   self.sudakov.log_mu2_grid, self.sudakov.uniform, self.sudakov.dx,
                                   self.log_sudakov_q, self.log_sudakov_g, self.alphas_interp_info[1],
                                   self.dmu2_sudakov_qq, self.dmu2_sudakov_qg, 
                                   self.dmu2_sudakov_gq, self.dmu2_sudakov_gg,
                                   self.log_thresholds2, epsilon=self.epsilon, n_samples=n_samples, max_n_splittings=self.max_n_splittings)
            
            kernels = construct_evolution_kernel_old(parton_in, k_mu, k_z, k_parton, k_w, k_n, self.k_log_mu2_grid, self.k_z_grid,
                                                         n_samples=n_samples) 
            end = time.time()
            print(f'Time elapsed: {end-start:.2f}') 
            return kernels
        else:
            print(f'Separating in batches of 1 million')
            n_batches = n_samples // 1_000_000 
            n_batches += 1 if n_samples % 1_000_000 !=0 else 0

            
            for i in range(n_batches):
                time_batch_start = time.time()
                n_events = 1_000_000 if (i+1)*1_000_000 < n_samples else (n_samples - i*1_000_000)
                k_mu, k_z, k_parton, k_w, k_n =  generate_samples_batch(parton_in, self.mu_0, self.mu_f,
                                   self.sudakov.log_mu2_grid, self.sudakov.uniform, self.sudakov.dx,
                                   self.log_sudakov_q, self.log_sudakov_g, self.alphas_interp_info[1],
                                   self.dmu2_sudakov_qq, self.dmu2_sudakov_qg, 
                                   self.dmu2_sudakov_gq, self.dmu2_sudakov_gg,
                                   self.log_thresholds2, epsilon=self.epsilon, n_samples=n_events, max_n_splittings=self.max_n_splittings)
                time_batch_end = time.time()
                #print(f'Time elapsed for {n_events}: {time_batch_end-time_batch_start} secs')
                kernels = construct_evolution_kernel(parton_in, k_mu, k_z, k_parton, k_w, k_n, self.k_log_mu2_grid, self.k_z_grid,
                                                         z_ii_w, z_ii_w2, z_qi_w, z_qi_w2,
                                                         z_ci_w, z_ci_w2, z_bi_w, z_bi_w2,
                                                         z_gi_w, z_gi_w2, delta_w, delta_w2,
                                                         underflow_w, underflow_w2, 
                                                         n_events=n_events,n_samples=n_samples) 
                z_ii_w = kernels[0]; z_ii_w2 = kernels[1]; z_qi_w = kernels[2]; z_qi_w2 = kernels[3]
                z_ci_w = kernels[4]; z_ci_w2 = kernels[5]; z_bi_w = kernels[6]; z_bi_w2 = kernels[7]
                z_gi_w = kernels[8]; z_gi_w2 = kernels[9]; delta_w = kernels[10]; delta_w2 = kernels[11]
                underflow_w = kernels[12]; underflow_w2 = kernels[13]
                #print(f'Time elapsed for contructing Kernels: {time.time()-time_batch_start} secs')
            
                del k_mu, k_z, k_parton, k_w, k_n

        kernels = (z_ii_w, z_ii_w2, z_qi_w, z_qi_w2,
                   z_ci_w, z_ci_w2, z_bi_w, z_bi_w2,
                   z_gi_w, z_gi_w2, delta_w, delta_w2,
                   underflow_w, underflow_w2) 

        kernels_norm = kernels #renormalise_kernel_rows(kernels, self.k_z_grid)
    
        
        end = time.time()
        print(f'Time elapsed: {end-start:.2f}')

        return kernels_norm


    def generate_evolution_kernel(self, parton_in, n_samples=10_000):
        print(f'Generating evolution kernel K_i{parton_in} -> i')
        print(f'Generating {n_samples} samples')
        start = time.time()
        mu, z, parton, w, n =  generate_samples(parton_in, mu_0, mu_f,
                                                log_mu2_grid, log_mu2_grid_uniform, log_mu2_dx,
                                                log_delta_q_grid, log_delta_g_grid, alphas_grid,
                                                dmu2_sudakov_qq_grid, dmu2_sudakov_qg_grid, 
                                                dmu2_sudakov_gq_grid, dmu2_sudakov_gg_grid,
                                                log_thresholds2, n_samples=100_000, epsilon=1e-6, max_n_splitting=100)
        end = time.time()
        print(f'Samples generated, time elapsed: {end-start:.2f}')
        #print(f'Constructing evolution kernel')
        #start = time.time()
        #kernel_ig = construct_evolution_kernel(parton_in, k_mu, k_z, k_parton, k_w, k_n, 
        #                                       mu_grid, z_grid, n_samples=n_samples)
        #end = time.time()
        #print(f'K_i{parton_in} constructed, time elapsed: {end-start:.2f}')

        return (mu, z, parton, w, n)
    
    def generate_kernel_with_resampling(self, parton_in, n_samples=1_000_000, ess_frac=0.5):
        n_mu = len(self.k_log_mu2_grid)
        n_z = len(self.k_z_grid) - 1
        bin_widths = np.diff(self.k_z_grid)
    
        z_ii_w = np.zeros((n_mu, n_z)); z_ii_w2 = np.zeros((n_mu, n_z))
        z_qi_w = np.zeros((n_mu, n_z)); z_qi_w2 = np.zeros((n_mu, n_z))
        z_ci_w = np.zeros((n_mu, n_z)); z_ci_w2 = np.zeros((n_mu, n_z))
        z_bi_w = np.zeros((n_mu, n_z)); z_bi_w2 = np.zeros((n_mu, n_z))
        z_gi_w = np.zeros((n_mu, n_z)); z_gi_w2 = np.zeros((n_mu, n_z))
        delta_w = np.zeros(n_mu); delta_w2 = np.zeros(n_mu)
    
        parton = np.full(n_samples, parton_in, dtype=np.int64)
        log_mu2_cur = np.full(n_samples, np.log(self.mu_0**2))
        log_delta_i_aux0 = self.log_sudakov_g if parton_in == 0 else self.log_sudakov_q
        log_sudakov_cur = np.full(n_samples,
            linear_interp(np.log(self.mu_0**2), self.sudakov.log_mu2_grid, log_delta_i_aux0,
                          self.sudakov.uniform, self.sudakov.dx))
        z_cum = np.ones(n_samples)
        w_cum = np.ones(n_samples)
    
        for g in range(n_mu):
            log_mu2_target = self.k_log_mu2_grid[g]
            advance_population(parton, log_mu2_cur, log_sudakov_cur, z_cum, w_cum,
                               log_mu2_target, self.sudakov.log_mu2_grid, self.sudakov.uniform, self.sudakov.dx,
                               self.log_sudakov_q, self.log_sudakov_g, self.alphas_interp_info[1],
                               self.dmu2_sudakov_qq, self.dmu2_sudakov_qg,
                               self.dmu2_sudakov_gq, self.dmu2_sudakov_gg,
                               self.log_thresholds2, self.epsilon)
    
            fill_row_from_population(g, parton, z_cum, w_cum, parton_in, self.k_z_grid, bin_widths, n_samples,
                                     z_ii_w, z_ii_w2, z_qi_w, z_qi_w2,
                                     z_ci_w, z_ci_w2, z_bi_w, z_bi_w2,
                                     z_gi_w, z_gi_w2, delta_w, delta_w2)
    
            sum_w = w_cum.sum(); sum_w2 = (w_cum**2).sum()
            n_eff = sum_w**2 / sum_w2 if sum_w2 > 0 else float(n_samples)
            if n_eff < ess_frac * n_samples:
                idx = systematic_resample(w_cum)
                parton = parton[idx].copy()
                log_mu2_cur = log_mu2_cur[idx].copy()
                log_sudakov_cur = log_sudakov_cur[idx].copy()
                z_cum = z_cum[idx].copy()
                mean_w = w_cum.mean()
                w_cum = np.full(n_samples, mean_w)
    
        return (z_ii_w, z_ii_w2, z_qi_w, z_qi_w2, z_ci_w, z_ci_w2,
            z_bi_w, z_bi_w2, z_gi_w, z_gi_w2, delta_w, delta_w2)
    
    def print_banner(self):
        banner = r"""
        ____        _____            _
       |  _ \ _   _| ____|_   _____ | |_   __
       | |_) | | | |  _| \ \ / / _ \| \ \ / /
       |  __/| |_| | |___ \ V / (_) | |\ V /
       |_|    \__, |_____| \_/ \___/|_| \_/
              |___/
    
            Parton Evolution via Monte Carlo Sudakov Sampling
        """
        equation = r"""
          f_i(x, mu) = Delta_i(mu0,mu) f_i(x, mu0)
                     + sum_j  Integral[x,1] dz/z  K_ij(z, mu) f_j(x/z, mu0)
        """
    
        print(banner)
        print("  " + "-" * 58)
        print(f"  PyEvolv v{getattr(self, '__version__', '0.1.0')}")
        print("  " + "-" * 58)
        if self is not None:
            print(f"  mu_0            : {self.mu_0:.4g} GeV")
            print(f"  mu_f            : {self.mu_f:.4g} GeV")
            print(f"  epsilon (z cut) : {self.epsilon:.2e}")
            if hasattr(self, "log_mu2_grid"):
                n_bins = len(self.log_mu2_grid) - 1
                print(f"  log(mu^2) grid  : {n_bins} bins "
                      f"[{self.log_mu2_grid[0]:.3f}, {self.log_mu2_grid[-1]:.3f}]")
        print("  " + "-" * 58)
        print("  DGLAP-equivalent evolution equation:")
        print(equation)
        print("  " + "-" * 58)
        print("  Ready to evolve. Try: model.generate_split_kernels('g', ...)")
        print("  " + "-" * 58)

        


    

        
            

    
    






