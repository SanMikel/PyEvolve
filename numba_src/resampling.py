from numba import njit
import numpy as np
from numba_src.SudakovFormFactor import generate_scale, select_splitting
from numba_src.z_sampling import sample_z
from numba_src.AlphaQCD import get_nf
from numba_src.interpolation.linear import linear_interp

@njit(cache=True)
def advance_particle(parton_in, log_mu2_cur, log_sudakov_cur, z_cum, w_cum,
                     log_mu2_target,
                     log_mu2_grid, log_mu2_grid_uniform, log_mu2_grid_dx,
                     log_delta_q_grid, log_delta_g_grid, alphas_grid,
                     dmu2_sudakov_qq_grid, dmu2_sudakov_qg_grid,
                     dmu2_sudakov_gq_grid, dmu2_sudakov_gg_grid,
                     log_thresholds2, epsilon=1e-6, max_n_splitting=1000):
    """
    Advances ONE particle from its current state up to log_mu2_target,
    performing as many splittings as fall in between. Returns updated
    (parton, log_mu2, log_sudakov, z_cum, w_cum) - resumable on the next call.
    """
    log_delta_i_aux = log_delta_g_grid if parton_in == 0 else log_delta_q_grid

    for _ in range(max_n_splitting):
        log_mu2_candidate, log_sudakov_candidate = generate_scale(
            log_sudakov_cur, log_mu2_grid, log_delta_i_aux, False, log_mu2_grid_dx)

        if log_mu2_candidate >= log_mu2_target:
            break   # next splitting (if any) belongs to a later checkpoint
        if log_mu2_candidate <= log_mu2_cur:
            break   # no more resolvable splittings

        alphas = linear_interp(log_mu2_candidate, log_mu2_grid, alphas_grid,
                               log_mu2_grid_uniform, log_mu2_grid_dx)
        nf = get_nf(log_mu2_candidate, log_thresholds2)

        splitting_channel, w_splitting = select_splitting(
            parton_in, log_mu2_candidate,
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

        w_cum *= w_sampling * w_splitting
        z_cum *= z_candidate

        log_mu2_cur = log_mu2_candidate
        log_sudakov_cur = log_sudakov_candidate
        parton_in = parton_out
        log_delta_i_aux = log_delta_g_grid if parton_in == 0 else log_delta_q_grid

    return parton_in, log_mu2_cur, log_sudakov_cur, z_cum, w_cum


@njit(cache=True)
def systematic_resample(weights):
    n = len(weights)
    cumsum = np.cumsum(weights)
    cumsum /= cumsum[-1]
    positions = (np.random.rand() + np.arange(n)) / n
    indices = np.searchsorted(cumsum, positions)
    return indices

@njit(cache=True)
def advance_population(parton, log_mu2_cur, log_sudakov_cur, z_cum, w_cum,
                       log_mu2_target, log_mu2_grid, log_mu2_grid_uniform, log_mu2_grid_dx,
                       log_sudakov_q, log_sudakov_g, alphas_grid,
                       dmu2_sudakov_qq, dmu2_sudakov_qg, dmu2_sudakov_gq, dmu2_sudakov_gg,
                       log_thresholds2, epsilon):
    n = len(parton)
    for i in range(n):
        parton[i], log_mu2_cur[i], log_sudakov_cur[i], z_cum[i], w_cum[i] = advance_particle(
            parton[i], log_mu2_cur[i], log_sudakov_cur[i], z_cum[i], w_cum[i],
            log_mu2_target, log_mu2_grid, log_mu2_grid_uniform, log_mu2_grid_dx,
            log_sudakov_q, log_sudakov_g, alphas_grid,
            dmu2_sudakov_qq, dmu2_sudakov_qg, dmu2_sudakov_gq, dmu2_sudakov_gg,
            log_thresholds2, epsilon)