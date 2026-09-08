import numpy as np
from scipy.integrate import cumulative_trapezoid, quad_vec
from numba_src.SplittingFunctions import Pgg, Pqq, Pqg, Pgq
from numba_src.interpolation import Interpolator
from numba_src.interpolation.linear import linear_interp
from numba_src.AlphaQCD import get_nf
import time
import warnings
from numba import njit

@njit(cache=True)
def generate_scale(log_sudakov_start, x, y, uniform, dx):
    # Generate new scale
    sudakov_random = np.random.rand()   
    log_sudakov_new = log_sudakov_start - np.log(sudakov_random) 

    # Since the y side of the grid is the integral, the spacing is not unifom
    uniform=False
    # Check if new sudakov outside boundaries
    if log_sudakov_new >= y[-1]:
        return x[-1], y[-1]

    log_mu2 = linear_interp(log_sudakov_new, y, x, uniform, dx)
    return log_mu2, log_sudakov_new 


@njit(cache=True)
def select_splitting(parton_type, log_mu2, 
                     log_mu2_grid, log_mu2_grid_uniform, log_mu2_grid_dx, 
                     dmu2_sudakov_qq_grid, dmu2_sudakov_qg_grid,
                     dmu2_sudakov_gq_grid, dmu2_sudakov_gg_grid):
    if parton_type == 0:
        integral_gg = linear_interp(log_mu2, log_mu2_grid, dmu2_sudakov_gg_grid, log_mu2_grid_uniform, log_mu2_grid_dx)
        integral_qg = linear_interp(log_mu2, log_mu2_grid, dmu2_sudakov_qg_grid, log_mu2_grid_uniform, log_mu2_grid_dx) 

        integral_total = integral_gg + integral_qg
        rate_gg = integral_gg/integral_total
        rate_qg = integral_qg/integral_total

        rate_random = np.random.rand()

        if rate_random < rate_gg:
            return 11, 1.0/integral_gg
        else:
            return 21, 1.0/integral_qg 

        
    else:
        integral_qq = linear_interp(log_mu2, log_mu2_grid, dmu2_sudakov_qq_grid, log_mu2_grid_uniform, log_mu2_grid_dx)
        integral_gq = linear_interp(log_mu2, log_mu2_grid, dmu2_sudakov_gq_grid, log_mu2_grid_uniform, log_mu2_grid_dx)

        integral_total = integral_qq + integral_gq
        rate_qq = integral_qq/integral_total
        rate_gq = integral_gq/integral_total

        rate_random = np.random.rand()

        if rate_random < rate_qq:
            return 22, 1.0/integral_qq
        else:
            return 12, 1.0/integral_gq





class SudakovFormFactor():

    def __init__(self, alpha_qcd, mu_grid_param=(1.3, 14000, 10000), epsilon=1e-6, order=0):

        # Load alpha qcd into class
        self.alpha_qcd = alpha_qcd

        # build mu grid in log scale 
        self.mu_0 = mu_grid_param[0]
        self.mu_f = mu_grid_param[1] 
        self.log_mu2_grid = np.linspace(np.log(self.mu_0**2), np.log(self.mu_f**2), mu_grid_param[2])

        # Get grid values in mu2 and mu
        self.mu2_grid = np.exp(self.log_mu2_grid)
        self.mu_grid  = np.sqrt(self.mu2_grid)

        # Get grid spacing and uniformity
        dxs = np.diff(self.log_mu2_grid)
        self.uniform = bool(np.allclose(dxs, dxs[0]))
        self.dx = float(dxs[0]) if self.uniform else 0.0

        # Get integration limits of z
        self.epsilon = epsilon
        self.z_max = 1.0 - epsilon
        self.z_min = epsilon
        
        self.order = order

        # Allocate variables for sudakov dz integral
        self.dmu2_sudakov_gg = None
        self.dmu2_sudakov_gq = None
        self.dmu2_sudakov_qg = None
        self.dmu2_sudakov_qq = None

        # Allocate variables for sudakov dmu dz integral
        self.log_sudakov_qq = None
        self.log_sudakov_qg = None
        self.log_sudakov_gq = None
        self.log_sudakov_gg = None
        self.log_sudakov_q = None
        self.log_sudakov_g = None


    def tabulate(self, interpolation_order=0, plot_interpolation=False):
        start = time.time()
        # Theoretically alpha_qcd shares the same mu grid
        alphas_grid = self.alpha_qcd.alphas_grid
        # Warm up splitting functions:
        Pgg(0.1, 0.1, 3, 1); Pqq(0.1, 0.1, 3, 1); Pgq(0.1, 0.1, 3, 1); Pqg(0.1, 0.1, 1)  
        # Integrate z term of the Sudakov form factor int_{0}^{z_max}dz z*Pij(z,mu)
        z_max = 1-self.epsilon
        nf = get_nf(self.mu_grid, self.alpha_qcd.thresholds)
        self.dmu2_sudakov_gg = quad_vec( lambda z: z* Pgg(z, alphas_grid, nf, self.order), 0.0, z_max)[0]
        self.dmu2_sudakov_gq = quad_vec( lambda z: z* Pgq(z, alphas_grid, nf, self.order), 0.0, z_max)[0]
        self.dmu2_sudakov_qg = quad_vec( lambda z: z* Pqg(z, alphas_grid, self.order)*2*nf, 0.0, z_max)[0]
        self.dmu2_sudakov_qq = quad_vec( lambda z: z* Pqq(z, alphas_grid, nf, self.order), 0.0, z_max)[0]
        
        # We perform cumulative integral of in dmu2/mu2 of the Sudakov form factor
        # We make use of the log_mu2_grid hence we do not need 1/mu2
        self.log_sudakov_qq = cumulative_trapezoid(self.dmu2_sudakov_qq, self.log_mu2_grid, initial=0.0)
        self.log_sudakov_qg = cumulative_trapezoid(self.dmu2_sudakov_qg, self.log_mu2_grid, initial=0.0)
        self.log_sudakov_gq = cumulative_trapezoid(self.dmu2_sudakov_gq, self.log_mu2_grid, initial=0.0)
        self.log_sudakov_gg = cumulative_trapezoid(self.dmu2_sudakov_gg, self.log_mu2_grid, initial=0.0)


        np.savetxt( f"grids/sudakov_grids_epsilon{self.epsilon:.2E}.txt",
                     np.column_stack((self.mu_grid, np.exp(-self.log_sudakov_qq), np.exp(-self.log_sudakov_qg), np.exp(-self.log_sudakov_gq), np.exp(-self.log_sudakov_gg))),
                     header="mu Sudakov_qq Sudakov_qg Sudakov_gq Sudakov_gg", fmt="%.10e" )

        self.log_sudakov_g = self.log_sudakov_gg + self.log_sudakov_qg
        self.log_sudakov_q = self.log_sudakov_qq + self.log_sudakov_gq
        
        end = time.time()
        print(f"Sudakov form factor tabulated, time elapsed: {end-start:.3f} s") 
        self.interpolate_grids()

    def interpolate_grids(self):
        dummy =  Interpolator.FastInterp(self.log_mu2_grid, self.dmu2_sudakov_gg)


    def generate_scale(self, parton, log_sudakov_start):
        if parton == 0:
            return generate_scale(log_sudakov_start, self.log_mu2_grid, self.log_sudakov_g, self.unifom, self.dx)
        
        else:
            return generate_scale(log_sudakov_start, self.log_mu2_grid, self.log_sudakov_q, self.unifom, self.dx)


    def evaluate(self, parton, mu):
        log_mu2 = np.log(mu**2)
        if parton==0:
            return np.exp(-linear_interp(log_mu2, self.log_mu2_grid, self.log_sudakov_g, self.uniform, self.dx))
        
        else:
            return np.exp(-linear_interp(log_mu2, self.log_mu2_grid, self.log_sudakov_q, self.uniform, self.dx)) 
        
