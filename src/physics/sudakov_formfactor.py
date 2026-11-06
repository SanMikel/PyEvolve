from src.parameters import LO, NLO, G_TO_G, G_TO_Q, Q_TO_G, Q_TO_Q, Q_TO_QBAR, Q_TO_QPRIME, TIMELIKE, SPACELIKE
from src.physics.splittingfunctions.splittingfunctions import P_gg, P_qq, P_qg, P_gq, P_qbarq, P_qjqi, warmup
from numba_src.interpolation.linear import linear_interp
from numba_src.interpolation import Interpolator

from scipy.integrate import cumulative_trapezoid, quad_vec
import numpy as np
import time


class SudakovFormFactor():

    def __init__(self, alpha_qcd, scale_grid, epsilon=1e-6, qcd_order=LO, mode=SPACELIKE):

        # Load alpha qcd into class
        self.alpha_qcd = alpha_qcd

        # load scale grid object into class
        self.scale_grid = scale_grid
        self.log_mu2_grid = scale_grid.log_mu2_grid
        # Get integration limits of z
        self.epsilon = epsilon
        self.z_max = 1.0 - epsilon
        self.z_min = epsilon
        
        self.qcd_order = qcd_order
        self.mode = mode
    
        warmup()

        self.integrate()


    def integrate(self):
        start = time.time()
        # Integrate z term of the Sudakov form factor int_{0}^{z_max}dz z*Pij(z,mu)
        z_max = 1-self.epsilon
        nf = self.alpha_qcd.get_nf(self.scale_grid.mu2_grid)
        alphas_grid = self.alpha_qcd.alphas_grid
        self.dmu2_sudakov_gg = quad_vec( lambda z: z* P_gg(z, alphas_grid, nf, self.qcd_order, self.mode), 0.0, z_max)[0]
        self.dmu2_sudakov_gq = quad_vec( lambda z: z* P_gq(z, alphas_grid, nf, self.qcd_order, self.mode), 0.0, z_max)[0]
        self.dmu2_sudakov_qg = quad_vec( lambda z: z* P_qg(z, alphas_grid, self.qcd_order, self.mode), 0.0, z_max)[0]
        self.dmu2_sudakov_qq = quad_vec( lambda z: z* P_qq(z, alphas_grid, nf, self.qcd_order, self.mode), 0.0, z_max)[0]
        if self.qcd_order == 0:
            self.dmu2_sudakov_qqb = np.zeros_like(self.log_mu2_grid)
            self.dmu2_sudakov_qqp = np.zeros_like(self.log_mu2_grid)
        else:
            self.dmu2_sudakov_qqb = quad_vec( lambda z: z* P_qbarq(z, alphas_grid, nf, self.qcd_order, self.mode), 0.0, z_max)[0]
            self.dmu2_sudakov_qqp = quad_vec( lambda z: z* Pqjqi(z, alphas_grid, nf, self.qcd_order, self.mode)*(2*nf-2), 0.0, z_max)[0]

        
        
        # We perform cumulative integral of in dmu2/mu2 of the Sudakov form factor
        # We make use of the log_mu2_grid hence we do not need 1/mu2

        dmu2_sudakov_qqtotal = self.dmu2_sudakov_qq + self.dmu2_sudakov_qqb + self.dmu2_sudakov_qqp  
        self.log_sudakov_qq = cumulative_trapezoid(dmu2_sudakov_qqtotal, self.log_mu2_grid, initial=0.0)
        self.log_sudakov_qg = cumulative_trapezoid(self.dmu2_sudakov_qg, self.log_mu2_grid, initial=0.0)
        self.log_sudakov_gq = cumulative_trapezoid(self.dmu2_sudakov_gq, self.log_mu2_grid, initial=0.0)
        self.log_sudakov_gg = cumulative_trapezoid(self.dmu2_sudakov_gg, self.log_mu2_grid, initial=0.0)


        #np.savetxt( f"grids/sudakov_grids_epsilon{self.epsilon:.2E}.txt",
        #            np.column_stack((self.mu_grid, np.exp(-self.log_sudakov_qq), np.exp(-self.log_sudakov_qg), np.exp(-self.log_sudakov_gq), np.exp(-self.log_sudakov_gg))),
        #             header="mu Sudakov_qq Sudakov_qg Sudakov_gq Sudakov_gg", fmt="%.10e" )

        self.log_sudakov_g = self.log_sudakov_gg 
        self.log_sudakov_q = self.log_sudakov_qq
        if self.mode==SPACELIKE:
            self.log_sudakov_g += 2*nf*self.log_sudakov_qg
            self.log_sudakov_q += self.log_sudakov_gq
        else:
            self.log_sudakov_g += 2*nf*self.log_sudakov_gq
            self.log_sudakov_q += self.log_sudakov_qg
         

        end = time.time()
        print(f"Sudakov form factor tabulated, time elapsed: {end-start:.3f} s")

    def args_evolve_parton(self):
        args = (self.log_sudakov_q, self.log_sudakov_g, self.dmu2_sudakov_qq, 
                self.dmu2_sudakov_qg, self.dmu2_sudakov_gq, self.dmu2_sudakov_gg,
                self.dmu2_sudakov_qqb, self.dmu2_sudakov_qqp)

        return args


    



        
