from numba_src.AlphaQCD import AlphaQCD, alpha_qcd_fast, get_nf
from numba_src.SplittingFunctions import SplittingFunctions, evaluate
from numba_src.SudakovFormFactor import SudakovFormFactor, generate_scale, select_splitting
from numba_src.z_sampling import sample_z
from numba_src.banner import print_banner
from numba_src.interpolation.linear import linear_interp
from numba import njit, prange, set_num_threads
import numpy as np

'''

Function to evolve a single parton from starting sclae mu start to mu finish
    It is based on interpolation grids
'''

@njit(cache=True)
def evolve_parton(parton_in, log_mu2_start, log_mu2_finish,
                  log_mu2_grid, log_mu2_grid_uniform, log_mu2_grid_dx,
                  log_delta_q_grid, log_delta_g_grid, alphas_grid,
                  dmu2_sudakov_qq_grid, dmu2_sudakov_qg_grid, dmu2_sudakov_gq_grid, dmu2_sudakov_gg_grid, 
                  log_thresholds2, epsilon=1e-6, max_n_splitting=100, order=0):

    # Set starting configuration of the evolution
    log_mu2_step = log_mu2_start
    log_delta_i_aux = log_delta_g_grid if parton_in==0 else log_delta_q_grid
    log_sudakov_step = linear_interp(log_mu2_start, log_mu2_grid, log_delta_i_aux, log_mu2_grid_uniform, log_mu2_grid_dx)

    # Allocate arrays for history information according to max_n_splittings
    log_mu2_history = np.zeros(max_n_splitting)
    z_history  = np.zeros(max_n_splitting)
    w_history  = np.zeros(max_n_splitting)
    parton_history = np.empty(max_n_splitting)
    
    # Start loop 
    n_splittings = 0
    
    for i in range(max_n_splitting):

        # Generate new scale for splitting
        log_mu2_candidate, log_sudakov_candidate = generate_scale(log_sudakov_step, log_mu2_grid, log_delta_i_aux, False, log_mu2_grid_dx)
    
        # Conditions for candiate scale
        if log_mu2_candidate >= log_mu2_finish:
            break
        elif log_mu2_candidate <= log_mu2_step:
            break
    
        # Get alphas and number of flavours for the candidate scale
        alphas = linear_interp(log_mu2_candidate, log_mu2_grid, alphas_grid, log_mu2_grid_uniform, log_mu2_grid_dx)
        nf = get_nf(log_mu2_candidate, log_thresholds2)
    
        # Choose splitting function according to rates
        splitting_channel, w_rate = select_splitting( parton_in, log_mu2_candidate,
                                                      log_mu2_grid, log_mu2_grid_uniform, log_mu2_grid_dx,
                                                      dmu2_sudakov_qq_grid, dmu2_sudakov_qg_grid, 
                                                      dmu2_sudakov_gq_grid, dmu2_sudakov_gg_grid)
        
        # Generate z according 
        z_candidate, w_sampling = sample_z(splitting_channel, epsilon, 1.0 - epsilon)
        w_splitting = z_candidate*evaluate(splitting_channel, z_candidate, alphas, nf, order)
    


        # Choose new outgoing parton according to the splitting function
        if splitting_channel == 12 or splitting_channel == 11:
            parton_out = 0
        elif splitting_channel == 22:
            parton_out = parton_in
        else:
            sign = 1 if np.random.rand() < 0.5 else -1
            parton_out = sign * np.random.randint(1, nf + 1)
            w_sampling *= 2 * nf
        
        # Weight of the generated event
        w_event = w_rate * w_sampling * w_splitting

        # Save information on the history arrays
        log_mu2_history[i] = log_mu2_candidate
        z_history[i]       = z_candidate
        parton_history[i]  = parton_out
        w_history[i]       = w_event

        # Update step values for next iteration
        log_mu2_step     = log_mu2_candidate
        log_sudakov_step = log_sudakov_candidate
        if splitting_channel == 12 or splitting_channel == 21:
            log_delta_i_aux = log_delta_g_grid if parton_out==0 else log_delta_q_grid
            log_sudakov_step = linear_interp(log_mu2_step, log_mu2_grid, log_delta_i_aux, 
                                             log_mu2_grid_uniform, log_mu2_grid_dx)

        parton_in = parton_out
        log_delta_i_aux = log_delta_g_grid if parton_in==0 else log_delta_q_grid
        n_splittings += 1
       

    return log_mu2_history, z_history, parton_history, w_history, n_splittings



from numba import njit
import numpy as np
from numba_src.SudakovFormFactor import generate_scale, select_splitting
from numba_src.z_sampling import sample_z
from numba_src.AlphaQCD import get_nf
from numba_src.interpolation.linear import linear_interp

@njit(cache=True)
def evolve_parton_debug(parton_in, log_mu2_start, log_mu2_finish,
                  log_mu2_grid, log_mu2_grid_uniform, log_mu2_grid_dx,
                  log_delta_q_grid, log_delta_g_grid, alphas_grid,
                  dmu2_sudakov_qq_grid, dmu2_sudakov_qg_grid, dmu2_sudakov_gq_grid, dmu2_sudakov_gg_grid, 
                  log_thresholds2, epsilon=1e-6, max_n_splitting=100):

    log_mu2_step = log_mu2_start
    log_delta_i_aux = log_delta_g_grid if parton_in==0 else log_delta_q_grid
    log_sudakov_step = linear_interp(log_mu2_start, log_mu2_grid, log_delta_i_aux, log_mu2_grid_uniform, log_mu2_grid_dx)

    log_mu2_history   = np.zeros(max_n_splitting)
    z_history         = np.zeros(max_n_splitting)
    w_history         = np.zeros(max_n_splitting)
    w_splitting_hist  = np.zeros(max_n_splitting)   # <-- NEW: 1/integral factor
    w_sampling_hist   = np.zeros(max_n_splitting)   # <-- NEW: sample_z weight factor
    channel_hist      = np.zeros(max_n_splitting)   # <-- NEW: which splitting channel
    parton_history    = np.empty(max_n_splitting)

    n_splittings = 0
    for i in range(max_n_splitting):
        log_mu2_candidate, log_sudakov_candidate = generate_scale(log_sudakov_step, log_mu2_grid, log_delta_i_aux, False, log_mu2_grid_dx)

        if log_mu2_candidate >= log_mu2_finish:
            break
        elif log_mu2_candidate <= log_mu2_step:
            break

        alphas = linear_interp(log_mu2_candidate, log_mu2_grid, alphas_grid, log_mu2_grid_uniform, log_mu2_grid_dx)
        nf = get_nf(log_mu2_candidate, log_thresholds2)

        splitting_channel, w_splitting = select_splitting( parton_in, log_mu2_candidate,
                                           log_mu2_grid, log_mu2_grid_uniform, log_mu2_grid_dx,
                                           dmu2_sudakov_qq_grid, dmu2_sudakov_qg_grid, 
                                           dmu2_sudakov_gq_grid, dmu2_sudakov_gg_grid)

        z_candidate, w_sampling = sample_z(splitting_channel, epsilon, 1.0 - epsilon, alphas)

        if splitting_channel == 12 or splitting_channel == 11:
            parton_out = 0
        elif splitting_channel == 22:
            parton_out = parton_in
        else:
            sign = 1 if np.random.rand() < 0.5 else -1
            parton_out = sign * np.random.randint(1, nf + 1)
            w_sampling *= 2 * nf

        w_event = w_sampling * w_splitting

        log_mu2_history[i]  = log_mu2_candidate
        z_history[i]        = z_candidate
        parton_history[i]   = parton_out
        w_history[i]        = w_event
        w_splitting_hist[i] = w_splitting     # <-- NEW
        w_sampling_hist[i]  = w_sampling      # <-- NEW
        channel_hist[i]     = splitting_channel  # <-- NEW

        log_mu2_step     = log_mu2_candidate
        log_sudakov_step = log_sudakov_candidate
        parton_in        = parton_out
        log_delta_i_aux  = log_delta_g_grid if parton_in==0 else log_delta_q_grid
        n_splittings += 1

    return (log_mu2_history, z_history, parton_history, w_history,
            w_splitting_hist, w_sampling_hist, channel_hist, n_splittings)