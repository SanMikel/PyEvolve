def print_banner(self=None):
    banner = r"""
    ____        _____            _
   |  _ \ _   _| ____|_   _____ | |_   __
   | |_) | | | |  _| \ \ / / _ \| \ \ / /
   |  __/| |_| | |___ \ V / (_) | |\ V /
   |_|    \__, |_____| \_/ \___/|_| \_/
          |___/

        Parton Evolution via Monte Carlo Sudakov Sampling
    """
    equation = r"""
      f_i(x, mu) = Delta_i(mu0,mu) f_i(x, mu0)
                 + sum_j  Integral[x,1] dz/z  K_ij(z, mu) f_j(x/z, mu0)
    """

    print(banner)
    print("  " + "-" * 58)
    print(f"  PyEvolv v{getattr(self, '__version__', '0.1.0')}")
    print("  " + "-" * 58)
    if self is not None:
        print(f"  mu_0            : {self.mu_0:.4g} GeV")
        print(f"  mu_f            : {self.mu_f:.4g} GeV")
        print(f"  epsilon (z cut) : {self.epsilon:.2e}")
        if hasattr(self, "log_mu2_grid"):
            n_bins = len(self.log_mu2_grid) - 1
            print(f"  log(mu^2) grid  : {n_bins} bins "
                  f"[{self.log_mu2_grid[0]:.3f}, {self.log_mu2_grid[-1]:.3f}]")
    print("  " + "-" * 58)
    print("  DGLAP-equivalent evolution equation:")
    print(equation)
    print("  " + "-" * 58)
    print("  Ready to evolve. Try: model.generate_split_kernels('g', ...)")
    print("  " + "-" * 58)