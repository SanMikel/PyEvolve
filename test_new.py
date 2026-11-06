from src.physics.alpha_qcd import AlphaQCD
from src.physics.sudakov_formfactor import SudakovFormFactor
from src.grids import ScaleGrid
from src.evolution.parton_evolution import evolve_parton
from src.evolution.sample_generation import generate_n_samples
import numpy as np
from time import time
from numba import prange, njit

alpha_qcd = AlphaQCD(qcd_order=0)

scale_grid = ScaleGrid()


start = time()
alpha_qcd.tabulate(scale_grid)
end = time()
print(f"{end-start:.3f} seconds")
sudakov = SudakovFormFactor(alpha_qcd, scale_grid)
start = time()
evolve_parton(0, np.log(1.3**2), np.log(10**2),
              *scale_grid.args_evolve_parton(),
              alpha_qcd.alphas_grid,
              *sudakov.args_evolve_parton(),
              alpha_qcd.log_quark_thresholds2)
end = time()
print(f"{end-start:.3f} warm up seconds")


grid_args = scale_grid.args_evolve_parton() 
sudakov_args =  sudakov.args_evolve_parton() 
alpha_grid = alpha_qcd.alphas_grid
thr = alpha_qcd.log_quark_thresholds2 

start = time()
generate_n_samples(0, np.log(1.3**2), np.log(10**2),
                   *scale_grid.args_evolve_parton(), 
                   alpha_qcd.alphas_grid, alpha_qcd.log_quark_thresholds2,
                   *sudakov.args_evolve_parton(), n_samples=10)
end = time()
print(f"{end-start:.3f} warm up seconds")

start = time()
generate_n_samples(0, np.log(1.3**2), np.log(10**2),
                   *scale_grid.args_evolve_parton(), 
                   alpha_qcd.alphas_grid, alpha_qcd.log_quark_thresholds2,
                   *sudakov.args_evolve_parton(), n_samples=100_000)
end = time()
print(f"{end-start:.3f} warm up seconds")