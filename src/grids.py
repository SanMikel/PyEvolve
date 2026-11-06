import numpy as np

class ScaleGrid:
    def __init__(self, mu_min: float=1.0, mu_max: float=14000.0, n: int=10000 ):
        self.mu_min = mu_min; self.mu_max = mu_max
        self.n = n

        self.log_mu2_grid = np.linspace(np.log(self.mu_min**2), np.log(self.mu_max**2), self.n)
        self.mu2_grid     = np.exp(self.log_mu2_grid)
        self.mu_grid      = np.sqrt(self.mu2_grid)

        self.dlog_mu2 = self.log_mu2_grid[1] - self.log_mu2_grid[0]

    def args_evolve_parton(self):
        return (self.log_mu2_grid, True, self.dlog_mu2) 