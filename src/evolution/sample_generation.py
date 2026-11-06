from src.parameters import LO, NLO, TIMELIKE, SPACELIKE, ANGULAR_ORDERING, PT_ORDERING
from src.evolution.parton_evolution import evolve_parton
from numba import njit, prange
import numpy as np
import time


@njit(parallel=True, cache=True)
def generate_n_samples( parton_in, mu_0, mu_f,
                      log_mu2_grid, log_mu2_grid_uniform, log_mu2_grid_dx,
                      alphas_grid, log_thresholds2,
                      log_delta_q_grid, log_delta_g_grid, 
                      dmu2_sudakov_qq_grid, dmu2_sudakov_qg_grid, 
                      dmu2_sudakov_gq_grid, dmu2_sudakov_gg_grid,
                      dmu2_sudakov_qqb_grid, dmu2_sudakov_qqp_grid,
                      n_samples=100_000, epsilon=1e-6, qcd_order=LO,
                      max_n_splittings=100):
    
   
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
                                            dmu2_sudakov_qqb_grid, dmu2_sudakov_qqp_grid, 
                                            log_thresholds2, epsilon, max_n_splittings, qcd_order)
        n_all[i]      = n
        mu_all[i, :n]     = mu[:n]
        z_all[i, :n]      = z[:n]
        parton_all[i, :n] = parton[:n]
        w_all[i, :n]      = w[:n]

    return mu_all, z_all, parton_all, w_all, n_all

@njit(cache=True)    
def fill_pyevolve_grid(parton_in, mu, k_z, k_parton, k_w, k_n,
                       kernel_w, kernel_w2, kernel_d, kernel_u,
                       log_mu2_grid, z_grid, log_kt2_grid,
                       n_events, n_samples=-99,
                       ordering = ANGULAR_ORDERING):
    
    # Set grid information
    n_bins_mu = len(log_mu2_grid)
    n_bins_z  = len(z_grid) - 1
    z_widths = np.diff(z_grid)
    n_bins_kt = len(log_kt2_grid) - 1
    log_kt2_widths = np.diff(log_kt2_grid)

    if n_samples == -99:
        n_samples = n_events

    #precompute values used more than once
    n_samples2 = n_samples * n_samples
    pi2 = 2*np.pi
    
    # Get parton evolution kinematics ready!
    max_steps = k_mu.shape[1]
    z_cumulative = np.empty(max_steps)
    w_cumulative = np.empty(max_steps)
    log_kt2_cumulative = np.empty(max_steps)
    for i in range(n_events):
        n_emissions = k_n[i]
        z_prod = 1.0; w_prod = 1.0
        kt_x_vecsum = 0.0; kt_y_vecsum = 0.0
        for j in range(n_emissions):
            phi = pi2*np.random.random()
            z  = k_z [i, j]; kt = np.exp(0.5*k_mu[i, j])
            if ordering == ANGULAR_ORDERING:
                kt *= (1.-z)
            kt_x_vecsum -= kt*np.cos(phi)
            kt_y_vecsum -= kt*np.sin(phi)
            z_prod *= z
            w_prod *= k_w[i, j]

            z_cumulative [j] = z_prod
            w_cumulative [j] = w_prod
            log_kt2_cumulative[j] = np.log(kt_x_vecsum*kt_x_vecsum +
                                           kt_y_vecsum*kt_y_vecsum   )
        
        # Fill grid with parton evolution kinematics
        k_emissions = 0
        for k in range(n_bins_mu):
            log_mu2_step = log_mu2_grid[k]

            while k_emissions < n_emissions and k_mu[i, k_emissions] < log_mu2_step:
                k_emissions +=1
            
            if k_emissions == 0:
                kernel_d[k, 0] += 1.0 / n_samples
                kernel_d[k, 1] += 1.0 / n_samples2
                continue
            
            k_id = k_emissions - 1
            w_val  = w_cumulative[k_id]

            # Search z value bin
            z_val  = z_cumulative[k_id]
            if z_val < z_grid[0] or z_val > z_grid[-1]:
                continue
            z_bin_id = np.searchsorted(z_grid, z_val) - 1
            z_bin_width = z_widths[z_bin_id]
            
            # Search kT value bin
            log_kt2_val = log_kt2_cumulative[k_id]
            if log_kt2_val > log_kt2_grid[-1] or log_kt2_val < log_kt2_grid[0]:
                continue
            log_kt2_bin_id = np.searchsorted(log_kt2_grid, log_kt2_val) - 1 
            log_kt2_bin_width = log_kt2_widths[log_kt2_bin_id]

            parton_out = k_parton[i, k_emissions - 1]

            weight  = w_val / (log_kt2_bin_width * z_bin_width * n_samples)
            weight2 = weight * weight
            if parton_in !=0:
                if parton_out == parton_in:
                    kernel_w[k, 0, z_bin_id, log_kt2_bin_id]  += weight
                    kernel_w2[k, 0, z_bin_id, log_kt2_bin_id] += weight2
                elif parton_out == -parton_in:
                    kernel_w[k, 1, z_bin_id, log_kt2_bin_id]  += weight
                    kernel_w2[k, 1, z_bin_id, log_kt2_bin_id] += weight2
                elif parton_out == 0:
                    kernel_w[k, 3, z_bin_id, log_kt2_bin_id]  += weight
                    kernel_w2[k, 3, z_bin_id, log_kt2_bin_id] += weight2 
                elif np.abs(parton_out) != np.abs(parton_in) and np.abs(parton_out) < 4:
                    kernel_w[k, 2, z_bin_id, log_kt2_bin_id]  += weight / 4.0
                    kernel_w2[k, 2, z_bin_id, log_kt2_bin_id] += weight2 / 16.0
                elif np.abs(parton_out) == 4:
                    kernel_w[k, 4, z_bin_id, log_kt2_bin_id]  += weight / 2.0
                    kernel_w2[k, 4, z_bin_id, log_kt2_bin_id] += weight2 / 4.0
                elif np.abs(parton_out) == 5:
                    kernel_w[k, 5, z_bin_id, log_kt2_bin_id]  += weight / 2.0
                    kernel_w2[k, 5, z_bin_id, log_kt2_bin_id] += weight2 / 4.0
            
            else:
                if parton_out == parton_in:
                    kernel_w[k, 0, z_bin_id, log_kt2_bin_id]  += weight
                    kernel_w2[k, 0, z_bin_id, log_kt2_bin_id] += weight2
                elif parton_out != parton_in and np.abs(parton_out) < 4:
                    kernel_w[k, 2, z_bin_id, log_kt2_bin_id]  += weight / 6.0
                    kernel_w2[k, 2, z_bin_id, log_kt2_bin_id] += weight2/ 36.0
                elif np.abs(parton_out) == 4:
                    kernel_w[k, 4, z_bin_id, log_kt2_bin_id]  += weight / 2.0
                    kernel_w2[k, 4, z_bin_id, log_kt2_bin_id] += weight2 / 4.0
                elif np.abs(parton_out) == 5:
                    kernel_w[k, 5, z_bin_id, log_kt2_bin_id]  += weight / 2.0
                    kernel_w2[k, 5, z_bin_id, log_kt2_bin_id] += weight2 / 4.0

