
from src.parameters import LO, NLO, TIMELIKE, SPACELIKE, ANGULAR_ORDERING, PT_ORDERING
from numba import njit, prange, set_num_threads
import numpy as np
import matplotlib.pyplot as plt
import time

class EvolutionKernel:

    def __init__(self, evolution_kernel_grid, max_n_splittings=100, 
                 order=LO, ordering=ANGULAR_ORDERING, n_cores=1):

        self.order=order
        self.ordering = ordering
        self.mu_0 = mu_grid_param[0]
        self.mu_f = mu_grid_param[1]
        self.epsilon = epsilon
        self.max_n_splittings = max_n_splittings
        print_banner()
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
        self.dmu2_sudakov_qqb = self.sudakov.dmu2_sudakov_qqb
        self.dmu2_sudakov_qqp = self.sudakov.dmu2_sudakov_qqp 


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
        self.k_log_kt2_grid = np.linspace(np.log(k_log_kt2_grid_param[0]**2), np.log(k_log_kt2_grid_param[1]**2), k_log_kt2_grid_param[2])
        #self.k_z_grid = np.logspace(np.log10(k_z_grid_param[0]), np.log10(k_z_grid_param[1]), k_z_grid_param[2])
  
        
        print("Warming up functions")
        if n_cores == 1:
            generate_samples(0, self.mu_0, self.mu_f,
                             self.sudakov.log_mu2_grid, self.sudakov.uniform, self.sudakov.dx,
                             self.log_sudakov_q, self.log_sudakov_g, self.alphas_interp_info[1],
                             self.dmu2_sudakov_qq, self.dmu2_sudakov_qg, 
                             self.dmu2_sudakov_gq, self.dmu2_sudakov_gg,
                             self.dmu2_sudakov_qqb, self.dmu2_sudakov_qqp, 
                             self.log_thresholds2, epsilon=self.epsilon, n_samples=1, order=self.order)
        else:
            set_num_threads(n_cores)
            generate_samples_batch(0, self.mu_0, self.mu_f,
                                   self.sudakov.log_mu2_grid, self.sudakov.uniform, self.sudakov.dx,
                                   self.log_sudakov_q, self.log_sudakov_g, self.alphas_interp_info[1],
                                   self.dmu2_sudakov_qq, self.dmu2_sudakov_qg, 
                                   self.dmu2_sudakov_gq, self.dmu2_sudakov_gg,
                                   self.dmu2_sudakov_qqb, self.dmu2_sudakov_qqp,
                                   self.log_thresholds2, epsilon=self.epsilon, n_samples=1, order=self.order)
    

    def generate_evolution_kernel_batch(self, parton_in, n_samples=1_000_000):
        print(f'Generating {n_samples} samples for {parton_in}')
        start = time.time()
        
        n_mu = len(self.k_log_mu2_grid)
        n_z = len(self.k_z_grid) - 1
        n_kt = len(self.k_log_kt2_grid) - 1

        kernel_w = np.zeros((n_mu, 6, n_z, n_kt))
        kernel_w2 =np.zeros_like(kernel_w)
        kernel_d = np.zeros((n_mu, 2))
        kernel_u = np.zeros((n_mu, 6, n_z, 2))

        
        if n_samples < 2_000_000:
            k_mu, k_z, k_parton, k_w, k_n =  generate_samples_batch(parton_in, self.mu_0, self.mu_f,
                                   self.sudakov.log_mu2_grid, self.sudakov.uniform, self.sudakov.dx,
                                   self.log_sudakov_q, self.log_sudakov_g, self.alphas_interp_info[1],
                                   self.dmu2_sudakov_qq, self.dmu2_sudakov_qg, 
                                   self.dmu2_sudakov_gq, self.dmu2_sudakov_gg,
                                   self.dmu2_sudakov_qqb, self.dmu2_sudakov_qqp,
                                   self.log_thresholds2, epsilon=self.epsilon, n_samples=n_samples, max_n_splittings=self.max_n_splittings,
                                   order=self.order)
            
            construct_evolution_kernel_new(parton_in, k_mu, k_z, k_parton, k_w, k_n, 
                                           kernel_w, kernel_w2, kernel_d, kernel_u,
                                           self.k_log_mu2_grid, self.k_z_grid, self.k_log_kt2_grid,
                                           n_events=n_events,n_samples=n_samples, ordering=self.ordering) 
            end = time.time()
            print(f'Time elapsed: {end-start:.2f}') 
            return kernel_w, kernel_w2, kernel_d, kernel_u

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
                                   self.dmu2_sudakov_qqb, self.dmu2_sudakov_qqp,
                                   self.log_thresholds2, epsilon=self.epsilon, n_samples=n_events, 
                                   max_n_splittings=self.max_n_splittings, order=self.order)
                time_batch_end = time.time()
                #print(f'Time elapsed for {n_events}: {time_batch_end-time_batch_start} secs')
                construct_evolution_kernel_new(parton_in, k_mu, k_z, k_parton, k_w, k_n,
                                               kernel_w, kernel_w2, kernel_d, kernel_u, 
                                               self.k_log_mu2_grid, self.k_z_grid, self.k_log_kt2_grid,
                                               n_events=n_events,n_samples=n_samples) 
               
                #print(f'Time elapsed for contructing Kernels: {time.time()-time_batch_start} secs')
            
                del k_mu, k_z, k_parton, k_w, k_n
        
        end = time.time()
        print(f'Time elapsed: {end-start:.2f}')

        return kernel_w, kernel_w2, kernel_d, kernel_u


    def generate_evolution_kernel(self, parton_in, n_samples=10_000):
        print(f'Generating evolution kernel K_i{parton_in} -> i')
        print(f'Generating {n_samples} samples')
        start = time.time()
        mu, z, parton, w, n =  generate_samples(parton_in, mu_0, mu_f,
                                                log_mu2_grid, log_mu2_grid_uniform, log_mu2_dx,
                                                log_delta_q_grid, log_delta_g_grid, alphas_grid,
                                                dmu2_sudakov_qq_grid, dmu2_sudakov_qg_grid, 
                                                dmu2_sudakov_gq_grid, dmu2_sudakov_gg_grid,
                                                self.dmu2_sudakov_qqb, self.dmu2_sudakov_qqp,
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
    

        


    

        
            

    
    






