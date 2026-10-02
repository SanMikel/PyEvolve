import numpy as np
from scipy.integrate import solve_ivp
from numba_src.interpolation import Interpolator
from numba_src.interpolation.linear import linear_interp
from scipy.interpolate import interp1d
from numba_src.qcdbetafunction import betaqcd, softgluon
import time
from numba import njit


@njit(cache=True)
def alpha_qcd_fast(mu,  x, y, uniform, dx):
    #log_mu2 = np.log(mu**2)
    return linear_interp(mu, x, y,
                         uniform, dx)

@njit(cache=True)
def get_nf(mu, thresholds):
    nf = np.searchsorted(thresholds, mu)
    return nf

class AlphaQCD():
    def __init__(self, mu_ref=91.1880, alphas_ref=0.130, thresholds=[0, 0, 0, 1.47, 4.5, 173], order=0):
        self.mu_ref = mu_ref
        self.alphas_ref = alphas_ref
        self.thresholds = np.array(thresholds)
        self.log_thresholds2 = np.log(self.thresholds**2+1e-6)
        self.order = order

        self.mu_grid = None
        self.log_mu2_grid = None
        self.alphas_grid = None

        self.alpha_interpolator = None
        self.alphas_interp_info = None


    
    def get_number_of_active_flavours(self, mu):
       mu = np.asarray(mu)
       mask = mu[..., None] >= self.thresholds
       nf = np.sum(mask, axis=-1)
       return nf

    def evolve_rge(self, mu_ref, alphas_ref, mu, z=0.0):
        nf = self.get_number_of_active_flavours(np.sqrt(mu*mu_ref))
        
        mu_eff = mu*(1-z) if mu*(1-z) > 1.0 else 1.0
        t_ref = 2*np.log(mu_ref)
        t     = 2*np.log(mu_eff) 
    
        # Define rge
        rge = lambda t, alpha: -sum(
            betaqcd(nf, i) / pow((4*np.pi), i+1) * alpha**(2+i)
            for i in range(self.order+1)
        )
        
        t_range = [t_ref, t ] #if t > t_ref else [t, t_ref]
        solution = solve_ivp(rge, t_range, [alphas_ref], method='RK45')
        return solution.y[0][-1]


    def evolve_alphas_scalar(self, mu, z=0.0):
        nf_ref = self.get_number_of_active_flavours(self.mu_ref) 
        nf     = self.get_number_of_active_flavours(mu)  
        n_transitions = nf_ref - nf
    
        if n_transitions!=0:
            up = n_transitions < 0
            if mu > self.mu_ref:
                thresholds_crossed = [th for th in self.thresholds if self.mu_ref < th < mu]
            else:
                thresholds_crossed = [th for th in self.thresholds if mu < th < self.mu_ref][::-1]
            mu_step = self.mu_ref
            alphas_step = self.alphas_ref
            for mu_threshold in thresholds_crossed:
                alphas_threshold = self.evolve_rge(mu_step, alphas_step, mu_threshold, z)
                mu_step = mu_threshold
                alphas_step = alphas_threshold
    
            return self.evolve_rge(mu_step, alphas_step, mu, z)
        else:
            return self.evolve_rge(self.mu_ref, self.alphas_ref, mu, z)
    
    def evolve_alphas(self, mu):
        mu = np.asarray(mu)

        if mu.ndim == 0:
            return self.evolve_alphas_scalar(float(mu))
        
        return np.array([self.evolve_alphas_scalar(m) for m in mu])

    def tabulate(self, mu_grid_param=(1.3, 14000, 10000)):
        start = time.time()
        mu_min = mu_grid_param[0]
        mu_max = mu_grid_param[1]
        n_mu_grid = mu_grid_param[2] 

        self.log_mu2_grid = np.linspace(np.log(mu_grid_param[0]**2), np.log(mu_grid_param[1]**2), mu_grid_param[2])
        self.mu2_grid = np.exp(self.log_mu2_grid)
        self.mu_grid  = np.sqrt(self.mu2_grid)

        alphas_values = self.evolve_alphas(self.mu_grid)
        
        self.alphas_grid = alphas_values
        self.alphas_interpolator = Interpolator.FastInterp(self.log_mu2_grid, alphas_values)
        self.alphas_interp_info = self.alphas_interpolator.as_args()
    
        
        end = time.time()
        print(f"Alpha QCD tabulated, time elapsed: {end-start:.4f} seconds")


    def evaluate(self, mu, fast=False):
        log_mu2 = np.log(mu**2)
        if self.alpha_grid is None:
            return self.evolve_alphas(mu)
        
        if fast==False:
            return np.interp(log_mu2, self.log_mu2_grid, self.alpha_grid)
        else:
            return linear_interp(log_mu2, *self.alphas_interp_info)

    def __call__(self, mu, fast=False):
        return self.evaluate(mu, fast)




'''
def alphas_nnll(mu, z=0.0):
    nf = self.get_number_of_active_flavours(mu)
    alphas_mu = evolve_alphas(mu*(1-z), order=1)
    k_term = softgluon(nf, 2)*(pow(alphas_mu / (2*np.pi), 2))
    return alphas_mu*(1+k_term)

def alphas_phys(mu, z=0.0, order=0):
    nf = self.get_number_of_active_flavours(mu)
    alphas_mu = evolve_alphas(mu*(1-z))
    term = 1.0
    for i in range(order+1):
        order_term = softgluon(nf, i)*(pow(alphas_mu / (2*np.pi), i))
        term += order_term

    return alphas_mu * term 
'''
#np.savetxt( "alpha_s_tab.txt",
#                     np.column_stack((self.mu_grid, self.alpha_grid)),
#                     header="mu alpha_s", fmt="%.10e" )
        
        





