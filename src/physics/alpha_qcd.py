import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from src.physics.betafunction_qcd import beta_qcd
import time
from numba import njit

@njit(cache=True)
def get_nf(log_mu2, log_quark_thresholds2):
    return 3 + np.searchsorted(log_quark_thresholds2, log_mu2, side="right")

class AlphaQCD():
    def __init__(self, mu_ref: float = 91.188, alphas_ref: float = 0.130, 
                       mc: float = 1.5, mb: float = 4.5, mt: float = 172.5,
                       qcd_order: int = 0):

        print(f"Heavy quark masses:\n  m_c = {mc} GeV \n  m_b = {mb} GeV\n  m_t = {mt} GeV")
        print(f"Reference values:\n  mu_ref = {mu_ref} GeV    alpha_s(mu_ref) = {alphas_ref}")

        self.mu_ref = mu_ref; self.mu2_ref = mu_ref**2
        self.qcd_order = qcd_order; self.alphas_ref = alphas_ref 
        self.quark_thresholds2 = np.array([mc, mb, mt], dtype=np.float64) ** 2
        self.log_quark_thresholds2 = np.log(self.quark_thresholds2) 


        # Generate anchor points for faster evolution
        self.nf_ref = self.get_nf(self.mu2_ref)
        if self.nf_ref < 4:
            self.alphas_mc = self.solve_rge(self.mu2_ref, self.alphas_ref, mc**2, 3)
            self.alphas_mb = self.solve_rge(mc**2, self.alphas_mc, mb**2, 4)
            self.alphas_mt = self.solve_rge(mb**2, self.alphas_mb, mt**2, 5)
        elif self.nf_ref < 5:
            self.alphas_mc = self.solve_rge(self.mu2_ref, self.alphas_ref, mc**2, 4)
            self.alphas_mb = self.solve_rge(self.mu2_ref, self.alphas_ref, mb**2, 4)
            self.alphas_mt = self.solve_rge(mb**2, self.alphas_mb, mt**2, 5)     
        elif self.nf_ref < 6:
            self.alphas_mb = self.solve_rge(self.mu2_ref, self.alphas_ref, mb**2, 5)
            self.alphas_mt = self.solve_rge(self.mu2_ref, self.alphas_ref, mt**2, 5)
            self.alphas_mc = self.solve_rge(mb**2, self.alphas_mb, mc**2, 4)
        else:
            self.alphas_mt = self.solve_rge(self.mu2_ref, self.alphas_ref, mt**2, 6)
            self.alphas_mb = self.solve_rge(mt**2, self.alphas_mt, mb**2, 5)
            self.alphas_mc = self.solve_rge(mb**2, self.alphas_mb, mc**2, 4) 
        
        self.anchors_alphas = np.array([self.alphas_mc, self.alphas_mc, self.alphas_mb, self.alphas_mt], dtype=np.float64)
        self.anchors_mu2    = np.array([mc, mc, mb, mt], dtype=np.float64)**2
        
        #assert abs(self.alphas_at(self.mu2_ref) - self.alphas_ref) < 1e-8 

    def get_nf(self, mu2):
        return 3 + np.searchsorted(self.quark_thresholds2, mu2, side="right")
    
    def solve_rge(self, mu2_ref, alphas_ref, mu2, nf):

        rge = lambda t, alphas: -sum(
            beta_qcd(nf, i) / (4* np.pi)**(i+1) * alphas**(2+i)
            for i in range(self.qcd_order + 1)
        )

        t_ref = np.log(mu2_ref); t_end = np.log(mu2)
        solution = solve_ivp(rge, (t_ref, t_end), [alphas_ref], rtol=1e-10, atol=1e-12)
        return solution.y[0][-1]

    def alphas_at(self, mu2):
        mu2 = np.asarray(mu2)
        nf_at_mu2 = self.get_nf(mu2)
        i = nf_at_mu2 - 3 
        mu2_anchors = self.anchors_mu2[i]
        alphas_anchors = self.anchors_alphas[i]

        if np.ndim(mu2) == 0: 
            return self.solve_rge(mu2_anchors, alphas_anchors, mu2, nf_at_mu2) 
        
        else:
            return np.array(
                     [self.solve_rge(mu2_anchors[k], alphas_anchors[k], mu2[k], nf_at_mu2[k]) 
                     for k in range(np.size(mu2)) ]
                   )    

    def tabulate(self, scale_grid):
        start = time.time()
        self.scale_grid = scale_grid
        self.alphas_grid = self.alphas_at(self.scale_grid.mu2_grid)
        end   = time.time()
        print(f"Alpha QCD tabulated {end - start:.2f} seconds elapsed")
        self.plot_alphas()
        return 
    
    def plot_alphas(self):
        mu = self.scale_grid.mu_grid
        alphas = self.alphas_grid

        fig, ax = plt.subplots(figsize=(6,5), tight_layout=True)
        ax.plot(mu, alphas)
        ax.set_xlabel("$\\mu$ [GeV]")
        ax.set_ylabel("$\\alpha_{s}(\\mu)$")
        quark_masses = np.sqrt(self.quark_thresholds2)
        j_points = np.searchsorted(mu, quark_masses) 
        for k, quark, in enumerate(quark_masses):
            lo, hi = max(j_points[k] - 10, 0 ), min(j_points[k] +10, mu.size)
            if hi - lo < 2:
                continue
            mu_zoom, alphas_zoom = mu[lo:hi], alphas[lo:hi]
            axin = ax.inset_axes([0.15+0.315*k, 0.7, 0.2, 0.2])
            axin.plot(mu_zoom, alphas_zoom, "o-", ms=2.5, )
            axin.axvline(quark, color="0.7", ls='--', lw=0.7)
            axin.set_title(f"$m_{{q}}={quark:.2f}$ GeV", fontsize=7, loc="left")
            pad = 0.1*(alphas_zoom.max() - alphas_zoom.min())
            axin.set_xlim(mu_zoom[0], mu_zoom[-1])
            axin.set_ylim(alphas_zoom.min()-pad, alphas_zoom.max()+pad)
            axin.tick_params(labelsize=7)
            ax.indicate_inset_zoom(axin, edgecolor="0.4")
        qcd_order = ["LO", "NLO"]
        ax.set_title(f"$\\mu_{{ref}}={self.mu_ref:.3f}$ GeV, $\\alpha_{{s}}(\\mu_{{ref}})={self.alphas_ref:.3f}$ at {qcd_order[self.qcd_order]}",
                     loc='left', fontsize=10)
            
        ax.set_xscale("log")
        path = "figures/alphas.pdf" 
        plt.savefig(path)
        print(f"  figure for tabulated alpha_qcd saved in {path}")
        return




        





